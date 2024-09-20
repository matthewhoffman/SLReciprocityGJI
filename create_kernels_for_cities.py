# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pyshtools as pysh
import SLmod as SL
import RFmod as RF
from numpy import pi as pi
import netCDF4
from scipy.interpolate import RegularGridInterpolator
import xarray as xr

# set densities and constants

# set the truncation degree
L=2048

# read in the present day sea level and ice thickness
#print('read standard dataset')
#sl0,ice0 = SL.get_sl_ice_data(L)

print('init SL object')
sl = pysh.SHGrid.from_zeros(lmax=L, grid='GLQ')
ice = sl.copy()
#print(sl.data.shape, sl.nlat, sl.nlon, sl.lmax, sl.grid, sl.units, sl.extend)

# read in sea level and ice thickness from our datasets
print('read our data')
sl_file='sl.nc'
f = netCDF4.Dataset(sl_file, 'r')
f.set_auto_mask(False)
sl.data = f.variables['sl'][:]
f.close()
print('sl min/max:', sl.data.max(), sl.data.min())

ice_file='ice.nc'
f = netCDF4.Dataset(ice_file, 'r')
f.set_auto_mask(False)
ice.data = f.variables['ice'][:]
f.close()
print('ice min/max:', ice.data.max(), ice.data.min())

cities = np.loadtxt('/usr/projects/climate/mhoffman/SLE-E3SM/ISMIP6_processing/cities_lat_long.txt',
                    skiprows=1, delimiter=',',
                    dtype={'names': ('city', 'lat', 'lon'),
                           'formats': ('U15', 'f8', 'f8')})

# compute the ocean function
print('compute ocean fn')
#C = SL.ocean_function(sl, ice)

C = sl.copy()
C_file='C.nc'
f = netCDF4.Dataset(C_file, 'r')
f.set_auto_mask(False)
C.data = f.variables['C'][:]
f.close()
print('C min/max:', C.data.max(), C.data.min())


# ice mask
print('calculating ice mask')
#ice_mask = SL.ice_mask(sl, ice, val=0.)
ice_mask = sl.copy()
ice_mask_file='ice_mask.nc'
f = netCDF4.Dataset(ice_mask_file, 'r')
f.set_auto_mask(False)
ice_mask.data = f.variables['ice_mask'][:]
f.close()
print('ice_mask min/max:', ice_mask.data.max(), ice_mask.data.min())

# loop over cities
for c in cities:
   # set the observation point to Boston
   #lat =  42.3601
   #lon = -71.0589
   lat,lon = (c[1], c[2])
   if lon > 180.0:
       lon = lon - 360.0
   print(f'city={c[0]}, {lat}, {lon}')

   # compute the adjoint load for a SMOOTHED point load
   print('compute smoothed load')
   zeta_d,_,_,_ = SL.sea_level_load(L, lat, lon, angle=1.)

   # solve the sea level equation for SL^{\dagger}
   print('calculating sl_d')
   sl_d,_,_,_,_ = SL.fingerprint(C, zeta_d)
   print('done with sl_d')

   # set and plot the kernel projected onto regions of grounded ice
   K = SL.rhoi * (1-C) * sl_d * ice_mask
   #SL.plot(ice_mask*K,label = r'ice kernel (m$^{-2}$)',marker = [lat,lon])

   out_data_vars = {'K': (['lat', 'lon'], K.data.astype('float64'))}
   out_coords = {'lat': (['lat'], K.lats()),
                 'lon': (['lon'], K.lons())}
   dataOut = xr.Dataset(data_vars=out_data_vars, coords=out_coords)
   output_file = f'K_{c[0].replace(" ", "")}.nc'
   dataOut.to_netcdf(output_file, mode='w')



   # now get the value by integrating the sensitivity kernel against the direct load
   #rhs = SL.surface_integral(sl_d*zeta)
   #SL.plot(sl_d*zeta, label = r'sl_d * zeta', marker = [lat,lon])
   #print(f'SLC from kernel={rhs}')
