from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4_3D,ParticleFile,ErrorCode,AdvectionRK4, Variable, VectorField,Field
from glob import glob
from datetime import timedelta as delta
from datetime import date
from parcels import plotTrajectoriesFile, logger, GeographicPolar, Geographic
import xarray as xr
import geopandas as gpd
import numpy as np
import sys
from shapely.geometry import Point
#from IPython.core.interactiveshell import InteractiveShell
#InteractiveShell.ast_node_interactivity = "all"
import numpy.ma as ma
from netCDF4 import Dataset
from scipy import interpolate
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from copy import copy
import os
#import cmocean

## ~~~~~~~~~~~~~~~~~~ ##
## ~     Functions  ~ ##
## ~~~~~~~~~~~~~~~~~~ ##
def seed_polygon_shape(shapefile_name, release_site_id, fieldset, num_particles, release_depth):
    try :
        print('Using shapefile!!!! ' + shapefile_name)
        data_shape = gpd.read_file(shapefile_name)
        value_index = list(data_shape.loc[data_shape['FID'] == release_site_id].index)
        value_index = int("".join(map(str, value_index)))
        area = data_shape['area'][value_index]
        polygon = data_shape['geometry'][value_index]
        base_polygon = gpd.GeoSeries([polygon])
        num_sites = data_shape.shape[0]
        print('Number of sites : ', num_sites)
    except Exception:
        print("Error while attempting to load the shapefile " + shapefile_name)
        sys.exit()
    bounds = polygon.bounds
    z = np.ones((num_particles, 1)) * release_depth
    min_lon = bounds[0]
    max_lon = bounds[2]
    del_lon = max_lon - min_lon
    min_lat = bounds[1]
    max_lat = bounds[3]
    del_lat = max_lat - min_lat
    num_in_polygon = 0
    num_attempts = 0
    x_in = []
    y_in = []
    fieldset.check_complete()
    fieldset.computeTimeChunk(0, 1)
    while num_in_polygon < num_particles:
        xc = min_lon + np.random.random(num_particles) * del_lon
        yc = min_lat + np.random.random(num_particles) * del_lat
        pts = gpd.GeoSeries([Point(x, y) for x, y in zip(xc, yc)])
        # Are these points inside the polygon?
        p = base_polygon.apply(lambda x: pts.within(x))
        m = p.to_numpy().reshape(num_particles, 1)
        valid_indices = np.argwhere(m == True)
        valid_indices = valid_indices[:, 0]

        if len(valid_indices) > 0:
            xc = xc[valid_indices]
            yc = yc[valid_indices]

            for index in range(0, len(valid_indices)):
               # Be sure that we are seeding in water
                # try:
                #     (u, v) = fieldset.UV[0, release_depth, yc[index], xc[index]]
                # except FieldSamplingError:
                #     print('FieldSamplingError')
                #     continue
                x_in.append(xc[index])
                y_in.append(yc[index])
                num_in_polygon = len(x_in)
                if num_in_polygon >= num_particles:
                    break
        num_attempts = num_attempts + 1
        if num_attempts > 2 and num_in_polygon == 0:
            break
        if num_attempts > 10:
            break
    if num_attempts > 10:
        print('\nfailed to find valid points for ' + str(release_site_id))
        return
    # have we ended up with too many particles?
    if num_in_polygon > num_particles:
        x_in = x_in[0:num_particles]
        y_in = y_in[0:num_particles]
        num_in_polygon = len(x_in)
    if num_in_polygon == num_particles:
        print('Sucessfully seeded particles\n')
        print('num_attempts = ' + str(num_attempts))
    else:
        print('\nfailed to find valid points for ' + str(release_site_id))
        print('found ' + str(num_in_polygon) + ' locations')
    return(x_in, y_in, z, area)

def release_times_per_day(time_origin, num_particles_per_day, release_start_hour, release_end_hour, start_day, end_day, time_zone):
## conversion
    simulation_start_day = np.datetime64(release_start_day,'D')
    simulation_end_day   = np.datetime64(release_end_day,'D')
    new_time_origin      = np.datetime64(time_origin)
    release_times=[]
    number_days = (1 + simulation_end_day-simulation_start_day).astype(int)
    for day in range(0, number_days):
        release_start=np.datetime64(release_start_day + release_start_hour) + np.timedelta64(day,'D')
        release_end=np.datetime64(release_start_day +  release_end_hour) + np.timedelta64(day,'D')
        if time_zone == 'GMT+10' :
            release_start= release_start - np.timedelta64(10,'h')
            release_end = release_end - np.timedelta64(10,'h')
        diff = release_end - release_start
        offset = (release_start - new_time_origin)
        offset = np.timedelta64(offset, 's') / np.timedelta64(1, 's')
        q = int(np.timedelta64(diff, 's') / np.timedelta64(1, 's'))
        p = np.random.randint(0, q, num_particles_per_day)
        p = (np.round(p/3600) * 3600).astype(int)
        release_times=np.append(release_times, offset + p)
    num_particles = num_particles_per_day * number_days
    return release_times, p, num_particles

def DeleteParticle(particle, fieldset, time):
    print("Particle [%d] lost !! (%g %g %g %g)" % (particle.id, particle.lon, particle.lat, particle.depth, particle.time))
    particle.delete()


def GBRVerticalMovement(particle, fieldset, time):
    ## different depth for each of the phases
    if particle.age >= fieldset.first_change_depth:
        particle.depth = -10
    elif particle.age >= fieldset.second_change_depth:
        particle.depth = -15

def ageing(particle, fieldset, time):
     particle.age += math.fabs(particle.dt)

def FollowSurface(particle, fieldset, time) :
    if(particle.age < fieldset.first_change_depth):
        ## if the particle is at another depth than the initial depth of release,  it should comeback to the initial value
        if particle.depth < fieldset.release_depth:
            particle.depth = fieldset.release_depth
            ## checking velocity field at currrenth depth
            (u, v) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
            if fabs(u) < 1e-14 and fabs(v) < 1e-14 and particle.beached  == 0:
                new_depth = particle.depth
                while fabs(u) < 1e-14 and fabs(v) < 1e-14 and particle.beached  == 0:
                    new_depth = new_depth - 0.1
                    (u, v) = fieldset.UV[time, new_depth, particle.lat, particle.lon]
                    if(new_depth < -5):
                        particle.beached = 1
                particle.depth = new_depth
    else:
        (u, v) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
        if fabs(u) < 1e-14 and fabs(v) < 1e-14 and particle.beached  == 0:
            particle.beached = 1

def WindAdvectionRK4(particle, fieldset, time):
    """Advection of particles using fourth-order Runge-Kutta integration.
     Function needs to be converted to Kernel object before execution"""
    if particle.beached == 0 and particle.age < fieldset.first_change_depth:
        wp = fieldset.wind_percentage ## this need to be add to the fieldset
        if wp > 0:
            (wind_u1, wind_v1) = fieldset.UVwind[time, particle.depth, particle.lat, particle.lon]
            wind_u1 = wind_u1 * wp
            wind_v1 = wind_v1 * wp
            wind_lon1, wind_lat1 = (particle.lon + wind_u1*.5*particle.dt, particle.lat + wind_v1*.5*particle.dt)
            (wind_u2 ,wind_v2) = fieldset.UVwind[time + .5 * particle.dt, particle.depth, wind_lat1, wind_lon1]
            wind_u2 = wind_u2 * wp
            wind_v2 = wind_v2 * wp
            wind_lon2, wind_lat2 = (particle.lon + wind_u2*.5*particle.dt, particle.lat + wind_v2*.5*particle.dt)
            (wind_u3, wind_v3) = fieldset.UVwind[time + .5 * particle.dt, particle.depth, wind_lat2, wind_lon2]
            wind_u3 = wind_u3 * wp
            wind_v3 = wind_v3 * wp
            wind_lon3, wind_lat3 = (particle.lon + wind_u3*particle.dt, particle.lat + wind_v3*particle.dt)
            (wind_u4, wind_v4) = fieldset.UVwind[time + particle.dt, particle.depth, wind_lat3, wind_lon3]
            wind_u4 = wind_u4 * wp
            wind_v4 = wind_v4 * wp
            particle.lon += (wind_u1 + 2*wind_u2 + 2*wind_u3 + wind_u4) / 6. * particle.dt
            particle.lat += (wind_v1 + 2*wind_v2 + 2*wind_v3 + wind_v4) / 6. * particle.dt

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## ~            Displacement functions        ~ ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
def make_landmask(fielddata):
    """Returns landmask where land = 1 and ocean = 0
    fielddata is a netcdf file.
    """
    datafile = Dataset(fielddata)
    landmask = datafile.variables['u'][1, 39]
    landmask[landmask > 10] = np.nan
    landmask = np.ma.masked_invalid(landmask)
    landmask = landmask.mask.astype('int')
    return landmask

def moving_box_avg(fielddata):
    New_fielddata = np.full([(fielddata.shape[0]-1), (fielddata.shape[1]-1)], None)
    for rows in range(1, fielddata.shape[0]):
        for cols in range(1, fielddata.shape[1]):
            New_fielddata[rows-1, cols-1] = (fielddata[rows-1, cols-1] + fielddata[rows-1, cols] +
                                         fielddata[rows, cols] + fielddata[rows, cols-1]) * 0.25
    return New_fielddata

def get_coastal_nodes(landmask):
    """Function that detects the coastal nodes, i.e. the ocean nodes directly
    next to land. Computes the Laplacian of landmask.

    - landmask: the land mask built using `make_landmask`, where land cell = 1
                and ocean cell = 0.

    Output: 2D array array containing the coastal nodes, the coastal nodes are
            equal to one, and the rest is zero.
    """
    mask_lap = np.roll(landmask, -1, axis=0) + np.roll(landmask, 1, axis=0)
    mask_lap += np.roll(landmask, -1, axis=1) + np.roll(landmask, 1, axis=1)
    mask_lap -= 4*landmask
    coastal = np.ma.masked_array(landmask, mask_lap > 0)
    coastal = coastal.mask.astype('int')

    return coastal

def get_shore_nodes(landmask):
    """Function that detects the shore nodes, i.e. the land nodes directly
    next to the ocean. Computes the Laplacian of landmask.

    - landmask: the land mask built using `make_landmask`, where land cell = 1
                and ocean cell = 0.

    Output: 2D array array containing the shore nodes, the shore nodes are
            equal to one, and the rest is zero.
    """
    mask_lap = np.roll(landmask, -1, axis=0) + np.roll(landmask, 1, axis=0)
    mask_lap += np.roll(landmask, -1, axis=1) + np.roll(landmask, 1, axis=1)
    mask_lap -= 4*landmask
    shore = np.ma.masked_array(landmask, mask_lap < 0)
    shore = shore.mask.astype('int')

    return shore


def get_coastal_nodes_diagonal(landmask):
    """Function that detects the coastal nodes, i.e. the ocean nodes where
    one of the 8 nearest nodes is land. Computes the Laplacian of landmask
    and the Laplacian of the 45 degree rotated landmask.

    - landmask: the land mask built using `make_landmask`, where land cell = 1
                and ocean cell = 0.

    Output: 2D array array containing the coastal nodes, the coastal nodes are
            equal to one, and the rest is zero.
    """
    mask_lap = np.roll(landmask, -1, axis=0) + np.roll(landmask, 1, axis=0)
    mask_lap += np.roll(landmask, -1, axis=1) + np.roll(landmask, 1, axis=1)
    mask_lap += np.roll(landmask, (-1,1), axis=(0,1)) + np.roll(landmask, (1, 1), axis=(0,1))
    mask_lap += np.roll(landmask, (-1,-1), axis=(0,1)) + np.roll(landmask, (1, -1), axis=(0,1))
    mask_lap -= 8*landmask
    coastal = np.ma.masked_array(landmask, mask_lap > 0)
    coastal = coastal.mask.astype('int')
    return coastal

def get_shore_nodes_diagonal(landmask):
    """Function that detects the shore nodes, i.e. the land nodes where
    one of the 8 nearest nodes is ocean. Computes the Laplacian of landmask
    and the Laplacian of the 45 degree rotated landmask.
    - landmask: the land mask built using `make_landmask`, where land cell = 1
                and ocean cell = 0.
    Output: 2D array array containing the shore nodes, the shore nodes are
            equal to one, and the rest is zero.
    """
    mask_lap = np.roll(landmask, -1, axis=0) + np.roll(landmask, 1, axis=0)
    mask_lap += np.roll(landmask, -1, axis=1) + np.roll(landmask, 1, axis=1)
    mask_lap += np.roll(landmask, (-1,1), axis=(0,1)) + np.roll(landmask, (1, 1), axis=(0,1))
    mask_lap += np.roll(landmask, (-1,-1), axis=(0,1)) + np.roll(landmask, (1, -1), axis=(0,1))
    mask_lap -= 8*landmask
    shore = np.ma.masked_array(landmask, mask_lap < 0)
    shore = shore.mask.astype('int')
    return shore

def create_displacement_field(landmask, double_cell=False):
    """Function that creates a displacement field 1 m/s away from the shore.
    - landmask: the land mask dUilt using `make_landmask`.
    - double_cell: Boolean for determining if you want a double cell.
      Default set to False.
    Output: two 2D arrays, one for each camponent of the velocity.
    """
    shore = get_shore_nodes(landmask)
    shore_d = get_shore_nodes_diagonal(landmask) # bordering ocean directly and diagonally
    shore_c = shore_d - shore                    # corner nodes that only border ocean diagonally
    Ly = np.roll(landmask, -1, axis=0) - np.roll(landmask, 1, axis=0) # Simple derivative
    Lx = np.roll(landmask, -1, axis=1) - np.roll(landmask, 1, axis=1)
    Ly_c = np.roll(landmask, -1, axis=0) - np.roll(landmask, 1, axis=0)
    Ly_c += np.roll(landmask, (-1,-1), axis=(0,1)) + np.roll(landmask, (-1,1), axis=(0,1)) # Include y-component of diagonal neighbours
    Ly_c += - np.roll(landmask, (1,-1), axis=(0,1)) - np.roll(landmask, (1,1), axis=(0,1))
    Lx_c = np.roll(landmask, -1, axis=1) - np.roll(landmask, 1, axis=1)
    Lx_c += np.roll(landmask, (-1,-1), axis=(1,0)) + np.roll(landmask, (-1,1), axis=(1,0)) # Include x-component of diagonal neighbours
    Lx_c += - np.roll(landmask, (1,-1), axis=(1,0)) - np.roll(landmask, (1,1), axis=(1,0))
    v_x = -Lx*(shore)
    v_y = -Ly*(shore)
    v_x_c = -Lx_c*(shore_c)
    v_y_c = -Ly_c*(shore_c)
    v_x = v_x + v_x_c
    v_y = v_y + v_y_c
    magnitude = np.sqrt(v_y**2 + v_x**2)
    # the coastal nodes between land create a problem. Magnitude there is zero
    # I force it to be 1 to avoid problems when normalizing.
    ny, nx = np.where(magnitude == 0)
    magnitude[ny, nx] = 1
    v_x = v_x/magnitude
    v_y = v_y/magnitude
    return v_x, v_y

def distance_to_shore(landmask, dx=1):
    """Function that computes the distance to the shore. It is based in the
    the `get_coastal_nodes` algorithm.
    - landmask: the land mask dUilt using `make_landmask` function.
    - dx: the grid cell dimension. This is a crude approxsimation of the real
    distance (be careful).
    Output: 2D array containing the distances from shore.
    """
    ci = get_coastal_nodes(landmask) # direct neighbours
    dist = ci*dx                     # 1 dx away
    ci_d = get_coastal_nodes_diagonal(landmask) # diagonal neighbours
    dist_d = (ci_d - ci)*np.sqrt(2*dx**2)       # sqrt(2) dx away
    return dist+dist_d


class GBRParticle_displacement(JITParticle):
    age = Variable('age', dtype=np.float32, initial=0.)
    dU = Variable('dU')
    dV = Variable('dV')
    d2s = Variable('d2s', initial=1e3)
    beached = Variable('beached', dtype=np.int32, initial=0.)


def set_displacement(particle, fieldset, time):
    particle.d2s = fieldset.distance2shore[time, particle.depth,
                               particle.lat, particle.lon]
    if  particle.d2s < 0.5:
        dispUab = fieldset.dispU[time, particle.depth, particle.lat,
                               particle.lon]
        dispVab = fieldset.dispV[time, particle.depth, particle.lat,
                               particle.lon]
        particle.dU = dispUab
        particle.dV = dispVab
    else:
        particle.dU = 0.
        particle.dV = 0.

def displace(particle, fieldset, time):
    if  particle.d2s < 0.5:
        particle.lon += particle.dU*particle.dt
        particle.lat += particle.dV*particle.dt

def find_particle_index(latitudes, longitudes, part_lat, part_lon):
    """"This function looks for the indexes of each particle in the Xi-Yi
    In which grid cell each particle is located.
    Use this function after setting the particles and before executing the run"""
    diff_grid = np.abs(part_lat - latitudes) + np.abs(part_lon - longitudes)
    XiYi = np.where(diff_grid == diff_grid.min())
    indexes = [XiYi[0].item(), XiYi[1].item()]
    return(indexes)



## Files and shapefiles
polygon_id = int(sys.argv[1])
data_path = '/scratch2/por07g/Data/GBR1_Simple/'
files = sorted(glob(data_path+'gbr1_simple_*.nc'))
mesh_mask =  'Data/coordinates.nc'
today = date.today()
folderShape="Shape_files/"
originalfile = "gbr1_coral_1m_merged.shp"
shapefile = folderShape+originalfile

## Variables to get the boundaries
grid    =  xr.open_dataset(mesh_mask)
grd_lat = grid['latitude']
grd_lon = grid['longitude']

## Setting the displacement field
landmask = make_landmask(files[0])
u_displacement, v_displacement = create_displacement_field(landmask)
d_2_s = distance_to_shore(landmask)
avg_landmask=moving_box_avg(landmask)
lmask = np.ma.masked_values(avg_landmask, 1) # land when interpolated value == 1



## Dimensions and variables
filenames = {'U': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': files[0], 'data': files},
             'V': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': files[0], 'data': files}}

variables = {'U': 'u',
             'V': 'v'}

dimensions = {'U': {'lon': 'longitude', 'lat': 'latitude', 'depth': 'zc', 'time': 'time'},
              'V': {'lon': 'longitude', 'lat': 'latitude', 'depth': 'zc', 'time': 'time'}}


## Setting the fielset
fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, allow_time_extrapolation = True, chunksize='auto')

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## ~            Creating the Wind Field         ~ ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
wind_filenames = {'lon': mesh_mask, 'lat': mesh_mask,'data': files}
wind_dimensions = {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}
Uwind_field = Field.from_netcdf(wind_filenames, ('Uwind', 'wspeed_u'),
                                     wind_dimensions, fieldtype='U',
                                     allow_time_extrapolation=True,
                                     transpose=False,deferred_load=True)
Vwind_field = Field.from_netcdf(wind_filenames, ('Vwind', 'wspeed_v'),
                                     wind_dimensions,
                                     fieldtype='V',
                                     allow_time_extrapolation=True,
                                     transpose=False,
                                     deferred_load=True)

fieldset.add_field(Uwind_field)
fieldset.add_field(Vwind_field)
wind_field = VectorField('UVwind', Uwind_field,  Vwind_field)
fieldset.add_vector_field(wind_field)
## Effect of wind
# I use a 3% effect of wind over the trajectory of the particles
wind_percentage = 3
fieldset.add_constant('wind_percentage', wind_percentage/100.0)
## change in depth
fieldset.add_constant('first_change_depth', .5 * 86400) ## after 12 hours the particle goes down to 10 m
fieldset.add_constant('second_change_depth', 2.5 * 86400) ## after +2.5 days the particle goes down to 15 m

## Displacement field
fieldset.add_field(Field('dispU', data=u_displacement,
                         lon=fieldset.U.grid.lon, lat=fieldset.U.grid.lat,
                         mesh='spherical'))
fieldset.add_field(Field('dispV', data=v_displacement,
                         lon=fieldset.U.grid.lon, lat=fieldset.U.grid.lat,
                         mesh='spherical'))
fieldset.dispU.units = GeographicPolar()
fieldset.dispV.units = Geographic()

## shore and distante to shore fields
fieldset.add_field(Field('landmask', landmask,
                         lon=fieldset.U.grid.lon, lat=fieldset.U.grid.lat,
                         mesh='spherical'))
fieldset.add_field(Field('distance2shore', d_2_s,
                         lon=fieldset.U.grid.lon, lat=fieldset.U.grid.lat,
                         mesh='spherical'))

release_depth = -0.1
fieldset.add_constant('release_depth', release_depth)
release_start_day = sys.argv[2]
release_start_hour= sys.argv[3]
release_end_day = sys.argv[4]
release_end_hour = sys.argv[5]
num_particles_per_day = int(sys.argv[6])
time_zone ='GMT+10'

time_origin = fieldset.U.grid.time_origin.time_origin ## reading the first time value from the entired dataset

[release_times, p, num_particles]= release_times_per_day(time_origin, num_particles_per_day, release_start_hour,
                                               release_end_hour, release_start_day, release_end_day, time_zone)

[x, y, z, area] = seed_polygon_shape(shapefile, polygon_id, fieldset, num_particles, release_depth)

pset = ParticleSet.from_list(fieldset, pclass=GBRParticle_displacement,
                                 lon = x,
                                 lat = y,
                                 depth = z,
                                 time = release_times)
## find grid cell number where particle is
for p in pset:
    yi, xi = find_particle_index(grd_lat, grd_lon, p.lat, p.lon)
    p.xi = np.array([xi], dtype=np.int32)
    p.yi = np.array([yi], dtype=np.int32)

## Creating the output file
# Parent Directory path
parent_dir = "/datasets/work/oa-coconet/work/OceanParcels_outputs/Coral"
# Path
path = os.path.join(parent_dir, release_start_day)
try:
    os.makedirs(path, exist_ok = True)
    print("Directory '%s' created successfully" % release_start_day)
except OSError as error:
    print("Directory '%s' can not be created" % release_start_day)

kernels = pset.Kernel(displace) + pset.Kernel(AdvectionRK4) + pset.Kernel(WindAdvectionRK4) + pset.Kernel(set_displacement) + pset.Kernel(FollowSurface) + pset.Kernel(ageing)
output_file = path + "/GBR1_H2p0_Coral_Release_" + release_start_day + "_Polygon_" +  str(polygon_id) + '_Wind_' + str(wind_percentage)+ '_percent_displacement_field'

#folder_temp_out = os.environ.get('MEMDIR') + '/out_temp'
#pfile = pset.ParticleFile(output_file, outputdt = delta(hours=1), tempwritedir = folder_temp_out)
pfile = pset.ParticleFile(output_file, outputdt = delta(hours=1))
strategy= "Coral: release time " + str(release_start_day + release_start_hour) + "; Particles per day: " + str(num_particles_per_day) + "; Number of days " +  str(num_particles/num_particles_per_day)+ " days; Using displacement field for particles reaching the coast"
pfile.add_metadata("Model_version", "GBR1_H2p0")
pfile.add_metadata("Data_accessed_from", "https://dapds00.nci.org.au/thredds/catalog/fx3/gbr1_2.0/catalog.html")
pfile.add_metadata("Date_created", today.strftime("%B %d, %Y"))
pfile.add_metadata("eReefs_data_frequency","hourly")
pfile.add_metadata("Wind_effect", str(wind_percentage)+ '_percent')
pfile.add_metadata("Seeding_strategy", strategy)
pfile.add_metadata("Recording_time_step", '3600 seconds')
pfile.add_metadata("GBR_Shapefiles", originalfile)
pfile.add_metadata("GBR_polygon_ID", str(polygon_id))
pfile.add_metadata("GBR_polygon_area[m2]", str(area))
pfile.add_metadata("Release_time", str(release_start_day + release_start_hour))
pfile.add_metadata("Particles_per_day", str(num_particles_per_day))
pfile.add_metadata("Spawning_days", str(num_particles/num_particles_per_day))
pfile.add_metadata("Author", "Javier Porobic; javier.porobicgarate@csiro.au")
pfile.add_metadata("Comments", "Coral larval dispersal tracks on the Great Barrier Reef. These simulations include the effect of wind in the first 12 hours after larvae spawn. They also have an impact of a displacement field when the larvae reach areas close to the coast.")

#pset.execute(kernels, runtime=delta(days=35), dt=delta(hours=1), output_file=pfile,
#             recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})
pset.execute(kernels, runtime=delta(hours=10), dt=delta(hours=1), output_file=pfile,
             recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})

## Safe the file
pfile.export()  # export the trajectory data to a netcdf file
pfile.close()

## Clean the file and remove the unnecesary variables
with xr.open_dataset(output_file + '.nc') as ds:
    data = ds.load()
#track_nc = xr.open_dataset(output_file + '.nc')
new_nc = data.drop(['dU','dV','d2s'])
#track_nc.close()
new_nc.to_netcdf(path = output_file + '.nc',  mode = 'w')
