# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pyshtools as pysh
import SLmod as SL
import RFmod as RF
from numpy import pi as pi
import netCDF4
from scipy.interpolate import RegularGridInterpolator, griddata
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
print(sl.data.shape, sl.nlat, sl.nlon, sl.lmax, sl.grid, sl.units, sl.extend)
print(sl.lats())
print(sl.lons())

# read in sea level and ice thickness from our datasets
print('read our data')
topg_file='/lustre/scratch5/mhoffman/SLM_Processing_2024-09-11/elastic-only/AIS/UCIJPL_ISSM/exp05/reformatted/topo_initial.nc'
f = netCDF4.Dataset(topg_file, 'r')
f.set_auto_mask(False)
reftopo = f.variables['topo_initial'][:]
f.close()

ice_file='/lustre/scratch5/mhoffman/SLM_Processing_2024-09-11/elastic-only/AIS/UCIJPL_ISSM/exp05/reformatted/grdice0.nc'
f = netCDF4.Dataset(ice_file, 'r')
f.set_auto_mask(False)
refice = f.variables['grdice'][:]
print(refice.max(), refice.min())
refice[np.isnan(refice)] = 0.0
f.close()
print(refice.max(), refice.min())

latfile = '/usr/projects/climate/mhoffman/SLE-E3SM/SeaLevelModel_Standard_Inputs/others/GLlat_2048.txt'
reflat = np.loadtxt(latfile)
lonfile = '/usr/projects/climate/mhoffman/SLE-E3SM/SeaLevelModel_Standard_Inputs/others/GLlon_2048.txt'
reflon = np.loadtxt(lonfile)

print('interpolating topo data')
fun = RegularGridInterpolator((reflat, reflon), np.flipud(reftopo), method = 'linear',
                              bounds_error=False, fill_value=None)
xy_grid = np.meshgrid(sl.lats(), sl.lons(), indexing='ij')
xy_list = np.reshape(xy_grid, (2, -1), order='C').T
sl.data[:,:] = fun(xy_list).reshape(sl.data.shape)
print('done')

# use griddata
#refx, refy = np.meshgrid(reflat, reflon)
#xi,yi = np.meshgrid(sl.lats(), sl.lons())
#sl.data[:,:] = griddata((refx.flatten(), refy.flatten()), np.transpose(reftopo, (1,0)).flatten(), (xi,yi), method='linear')
#print('done')

out_data_vars = {'sl': (['lat', 'lon'], sl.data.astype('float64'))}
out_coords = {'lat': (['lat'], sl.lats()),
              'lon': (['lon'], sl.lons())}
dataOut = xr.Dataset(data_vars=out_data_vars, coords=out_coords)
output_file = 'sl.nc'
dataOut.to_netcdf(output_file, mode='w')


print('interpolating ice data')
fun = RegularGridInterpolator((reflat, reflon), np.flipud(refice), method = 'linear',
                              bounds_error=False, fill_value=None)
ice.data[:,:] = fun(xy_list).reshape(ice.data.shape)
# interpolate onto sl grid
#for ilat,llat in enumerate(ice.lats()):
#    for ilon,llon in enumerate(ice.lons()):
#        ice.data[ilat,ilon] = fun((llat,llon))
out_data_vars = {'ice': (['lat', 'lon'], ice.data.astype('float64'))}
out_coords = {'lat': (['lat'], ice.lats()),
              'lon': (['lon'], ice.lons())}
dataOut = xr.Dataset(data_vars=out_data_vars, coords=out_coords)
output_file = 'ice.nc'
dataOut.to_netcdf(output_file, mode='w')



# compute the ocean function
print('compute ocean fn')
C = SL.ocean_function(sl, ice)

out_data_vars = {'C': (['lat', 'lon'], C.data.astype('int'))}
out_coords = {'lat': (['lat'], C.lats()),
              'lon': (['lon'], C.lons())}
dataOut = xr.Dataset(data_vars=out_data_vars, coords=out_coords)
output_file = 'C.nc'
dataOut.to_netcdf(output_file, mode='w')



# ice mask
print('calculating ice mask')
ice_mask = SL.ice_mask(sl, ice, val=0.)

out_data_vars = {'ice_mask': (['lat', 'lon'], ice_mask.data.astype('int'))}
out_coords = {'lat': (['lat'], ice_mask.lats()),
              'lon': (['lon'], ice_mask.lons())}
dataOut = xr.Dataset(data_vars=out_data_vars, coords=out_coords)
output_file = 'ice_mask.nc'
dataOut.to_netcdf(output_file, mode='w')

