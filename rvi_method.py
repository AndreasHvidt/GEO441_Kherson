#%%
import numpy as np
import rasterio as rio
from datetime import datetime, timedelta, date
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#%% file structure
def get_filenames(dir_str):
    directory = os.fsencode(dir_str)
    filenames = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        filenames.append(filename)
    return filenames

optical_files = get_filenames(r'S:\course\geo441\data\2023_Ukraine\S2\GEO441_Kherson_large_S2_10d_composites')


sar_folder = r'S:\course\geo441\data\2023_Ukraine\s1\Kherson_30m'
vv_folder = sar_folder + r'\12d_12d_vv'
vh_folder = sar_folder + r'\12d_12d_vh'
vv_filenames = get_filenames(vv_folder)
vh_filenames = get_filenames(vh_folder)


mask_files = get_filenames(r'P:\windows\Documents\GEO441 - RS Seminar\masks')


#%% optical data

ndvi_data = defaultdict(lambda: defaultdict(dict))
bsi_data = defaultdict(lambda: defaultdict(dict))
nir_data = defaultdict(lambda: defaultdict(dict))
r_data = defaultdict(lambda: defaultdict(dict))
g_data = defaultdict(lambda: defaultdict(dict))
b_data = defaultdict(lambda: defaultdict(dict))

aoi_indices = [1850, 2010, 5700, 6100]
count = 0
for file in optical_files:
    raster = rio.open(r'S:\course\geo441\data\2023_Ukraine\S2\GEO441_Kherson_large_S2_10d_composites' + '/' + file).read().astype(float)
    raster[raster==0] = np.nan
    ndvi = (raster[3]-raster[2])/(raster[3]+raster[2])
    bsi = ((raster[4]+raster[2]) - (raster[3]+raster[0])) / ((raster[4]+raster[2]) + (raster[3]+raster[0]))
    dat = datetime.strptime(file[file.find('UA_Kherson'):].removeprefix('UA_Kherson_S2_30m_').removesuffix('_composite_filt.tif'),'%Y-%m-%d')
    ndvi_data[dat.year][dat.month][dat.day] = ndvi
    bsi_data[dat.year][dat.month][dat.day] = bsi
    nir_data[dat.year][dat.month][dat.day] = raster[3]
    r_data[dat.year][dat.month][dat.day] = raster[2]
    g_data[dat.year][dat.month][dat.day] = raster[1]
    b_data[dat.year][dat.month][dat.day] = raster[0]

#%% sar data

rvi_data = defaultdict(lambda: defaultdict(dict))
rfdi_data = defaultdict(lambda: defaultdict(dict))

for i in range(len(vv_filenames)):
    vv_im = rio.open(vv_folder + '/' + vv_filenames[i]).read()[0]
    vh_im = rio.open(vh_folder + '/' + vh_filenames[i]).read()[0]
    dat = datetime.strptime(vv_filenames[i][12:20],'%Y%m%d') + timedelta(days=6)
    q = vh_im / vv_im
    N = q*(q+3)
    D = (q+1)*(q+1)
    RVI = N/D
    rfdi = (vv_im - vh_im) / (vv_im + vh_im)
    rvi_data[dat.year][dat.month][dat.day] = RVI
    rfdi_data[dat.year][dat.month][dat.day] = rfdi

#%% masks
mask_2021 = rio.open(r'P:\windows\Documents\GEO441 - RS Seminar\masks\Crops_mask_2021.tif').read()[0]
mask_2022 = rio.open(r'P:\windows\Documents\GEO441 - RS Seminar\masks\Crops_mask_2022.tif').read()[0]
mask_2023 = rio.open(r'P:\windows\Documents\GEO441 - RS Seminar\masks\Crops_mask_2023.tif').read()[0]
#%% rvi classification
grass_agr_classification = defaultdict(lambda: defaultdict(dict))

for year in rvi_data:
    grass_agr_mask = np.zeros((3670,6762))
    yearly_data = []
    for month in rvi_data[year]:
        for day in rvi_data[year][month]:
            yearly_data.append(rvi_data[year][month][day] * mask_2021[:-2,:])
    grass_agr_mask[(np.nanmax(yearly_data, axis = 0) - np.nanmin(yearly_data, axis = 0)) > 0.25] = 1
    grass_agr_mask[((np.nanmax(yearly_data, axis = 0) - np.nanmin(yearly_data, axis = 0)) < 0.25) & ((np.nanmax(yearly_data, axis = 0) - np.nanmin(yearly_data, axis = 0)) > 0)]  = 2
    grass_agr_mask[grass_agr_mask==0] = np.nan
    grass_agr_classification[year] = grass_agr_mask
            

# %% plotting of classification yearly differences
x_ticks = np.arange(399.930, 399.930 + mask_2021.shape[1] * .30, .30)
y_ticks = np.flip(np.arange(5200.050 - (mask_2021.shape[0]-2) * .30, 5200.050, .30))
for year in range(2021,2024):
    plt.figure(figsize=(20,12))
    plt.imshow(grass_agr_classification[year-1]-grass_agr_classification[year], vmin = -1, vmax = 1, cmap = 'RdYlGn',extent=[x_ticks[0], x_ticks[-1], y_ticks[-1], y_ticks[0]])
    plt.title(f'Classification change Agriculture-Grassland from {year-1} to {year} using RVI classification')
    plt.xlabel('X [km] (EPSG:32636)')
    plt.ylabel('Y [km] (EPSG:32636)')
    plt.savefig(r'\\geofiles.d.uzh.ch\private\ahvidt\windows\Documents\GEO441 - RS Seminar\Plots\rvi_classification\difference_maps' + '/' + f'{year-1}' + '-' + f'{year}' + '_rvi_classification_diffmap.png')
    plt.close()

# %%
for year in range(2021,2024):
    plt.figure(figsize=(20,12))
    plt.imshow(np.nan_to_num(grass_agr_classification[year], nan = 3).astype(int), cmap = 'Paired',extent=[x_ticks[0], x_ticks[-1], y_ticks[-1], y_ticks[0]])
    plt.title(f'Classification change Agriculture-Grassland from {year-1} to {year} using RVI classification')
    plt.xlabel('X [km] (EPSG:32636)')
    plt.ylabel('Y [km] (EPSG:32636)')
    plt.savefig(r'\\geofiles.d.uzh.ch\private\ahvidt\windows\Documents\GEO441 - RS Seminar\Plots\rvi_classification\classification_maps' + '/' + f'{year}' + '_rvi_classification_map.png')
    plt.close()

# %%
