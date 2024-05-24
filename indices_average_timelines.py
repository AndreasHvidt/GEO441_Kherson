#%% package loading and global parameter setting

from eodal.core.raster import RasterCollection
from eodal.core.band import Band
from eodal.core.band import GeoInfo
from eodal.core.scene import SceneCollection
from eodal.utils.sentinel2 import get_S2_processing_level
import os
import rasterio as rio
from rasterio.features import rasterize
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import geopandas as gpd


plt.rcParams['figure.figsize'] = [15, 15]

#%% define functions for data loading
def create_collection(file_path, band_names, band_aliases, geo_info, mask = None):
    # import data from file
    data = rio.open(file_path).read()
    data = data.astype(float)
    data[data==0] = np.nan
    
    #check for same length of input data and band names/aliases:
    if (len(data) != len(band_names)) or (len(data) != len(band_aliases)):
        raise ValueError("number of bands in data and band names/aliases must be the same")
    
    if mask is not None:
        if isinstance(mask, str):
            mask = rio.open(mask).read()
            data = mask * data
            data[data==0] = np.nan
        else:
            data = mask * data
            data[data==0] = np.nan

    #create raster collection with first band
    raster = RasterCollection(
             band_constructor=Band,
             band_name = band_names[0],
             band_alias = band_aliases[0],
             values=data[0],
             geo_info=geo_info
    )
    
    #add further bands
    for i in range(1,len(data)):
        band = Band(values = data[i], band_name = band_names[i], band_alias = band_aliases[i], geo_info = geo_info)
        raster.add_band(band_constructor=band)
    
    #add scene properties
    raster.scene_properties.acquisition_time = datetime.strptime(file_path[file_path.find('UA_Kherson'):].removeprefix('UA_Kherson_S2_30m_').removesuffix('_composite_filt.tif'),'%Y-%m-%d')
    raster.scene_properties.processing_level = get_S2_processing_level(dot_safe_name='S2A_MSIL2A_20190524T101031_N0212_R022_T32UPU_20190524T130304.SAFE')
    raster.scene_properties.platform = 'S2A'
    raster.scene_properties.sensor = 'MSI'
    
    return raster

def get_filenames(dir_str):
    directory = os.fsencode(dir_str)
    filenames = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        filenames.append(filename)
    return filenames
#%% get all available image names
data_folder = r'S:\course\geo441\data\2023_Ukraine\S2\GEO441_Kherson_large_S2_10d_composites'
filenames = get_filenames(data_folder)
mask_folder = r'P:\windows\Documents\GEO441 - RS Seminar\masks'
masknames = get_filenames(mask_folder)

#%% define geoinfo for raster collections
epsg = 32636
ulx, uly = 399930, 5200050
pixres_x, pixres_y = 30, -30
geo_info = GeoInfo(epsg = epsg, ulx = ulx, uly = uly, pixres_x = pixres_x, pixres_y = pixres_y)

del epsg, ulx, uly, pixres_x, pixres_y

#%% set band names
band_names = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
band_aliases = ['blue', 'green', 'red', 'nir_1', 'swir_1', 'swir_2']

#%% calc average ndvi, evi, msavi, bsi with a single starting mask
avg_list = []
median_list = []
date_list = []
si_list = ['ndvi','evi','msavi','bsi']
for filename in filenames:
    collection = create_collection(data_folder + '/' + filename, band_names, band_aliases, geo_info, mask = mask_folder + '/' + masknames[0])
    for si in si_list:
        collection.calc_si(si, inplace = True)
    if np.all(np.isnan(collection.get_values())):
        avg_list.append(np.array([np.nan, np.nan, np.nan, np.nan]))
        median_list.append(np.array([np.nan, np.nan, np.nan, np.nan]))
        date_list.append(collection.scene_properties.acquisition_time)
        continue
    avg_list.append(collection.band_summaries(band_selection= si_list, method = ['mean'])['mean'].values)
    medians = []
    for si in si_list:
        medians.append(np.nanmedian(collection.get_values(band_selection = [si])))
    median_list.append(medians)
    date_list.append(collection.scene_properties.acquisition_time)


#%% save timeline plots of the single starting mask
for i in range(len(si_list)):
    avgs = []
    medians = []
    for entry in avg_list:
        avgs.append(entry[i])
    
    for entry in median_list:
        medians.append(entry[i])
    plt.plot(date_list, avgs)
    plt.title(si_list[i]+'-averages')
    plt.xlabel('date')
    plt.ylabel('index-value')
    plt.savefig(r'P:\windows\Documents\GEO441 - RS Seminar\Plots\timelines_single_mask' + 
                '/' + si_list[i] + '_avg_timeline.png')
    plt.clf()
    plt.plot(date_list, medians)
    plt.title(si_list[i]+'-medians')
    plt.xlabel('date')
    plt.ylabel('index-value')
    plt.savefig(r'P:\windows\Documents\GEO441 - RS Seminar\Plots\timelines_single_mask' + 
                '/' + si_list[i] + '_median_timeline.png')
    plt.clf()

#%% calc average ndvi, evi, msavi, bsi with different masks
avg_list = []
median_list = []
date_list = []
si_list = ['ndvi','evi','msavi','bsi']
for filename in filenames:
    if datetime.strptime(filename.removeprefix('UA_Kherson_S2_30m_').removesuffix('_composite_filt.tif'),'%Y-%m-%d').year == (2020 or 2021):
        collection = create_collection(data_folder + '/' + filename, band_names, band_aliases, geo_info, mask = mask_folder + '/' + masknames[0])
    elif datetime.strptime(filename.removeprefix('UA_Kherson_S2_30m_').removesuffix('_composite_filt.tif'),'%Y-%m-%d').year == 2022:
        collection = create_collection(data_folder + '/' + filename, band_names, band_aliases, geo_info, mask = mask_folder + '/' + masknames[1])
    else:
        collection = create_collection(data_folder + '/' + filename, band_names, band_aliases, geo_info, mask = mask_folder + '/' + masknames[2])
    if np.all(np.isnan(collection.get_values())):
        avg_list.append(np.array([np.nan, np.nan, np.nan, np.nan]))
        median_list.append(np.array([np.nan, np.nan, np.nan, np.nan]))
        date_list.append(collection.scene_properties.acquisition_time)
        continue
    for si in si_list:
        collection.calc_si(si, inplace = True)
    avg_list.append(collection.band_summaries(band_selection= si_list, method = ['mean'])['mean'].values)
    medians = []
    for si in si_list:
        medians.append(np.nanmedian(collection.get_values(band_selection = [si])))
    median_list.append(medians)
    date_list.append(collection.scene_properties.acquisition_time)


#%% save timeline plots of the different starting masks
for i in range(len(si_list)):
    avgs = []
    medians = []
    for entry in avg_list:
        avgs.append(entry[i])
    
    for entry in median_list:
        medians.append(entry[i])
    plt.plot(date_list, avgs)
    plt.title(si_list[i]+'-averages')
    plt.xlabel('date')
    plt.ylabel('index-value')
    plt.savefig(r'P:\windows\Documents\GEO441 - RS Seminar\Plots\timelines_mult_mask' + 
                '/' + si_list[i] + '_avg_timeline.png')
    plt.clf()
    plt.plot(date_list, medians)
    plt.title(si_list[i]+'-medians')
    plt.xlabel('date')
    plt.ylabel('index-value')
    plt.savefig(r'P:\windows\Documents\GEO441 - RS Seminar\Plots\timelines_mult_mask' + 
                '/' + si_list[i] + '_median_timeline.png')
    plt.clf()
# %% create a mask from the frontline area of interest (10km buffer)
kherson_area = gpd.read_file(r'P:\windows\Documents\GEO441 - RS Seminar\qgis_files\kherson_area.geojson')
frontline_area = gpd.read_file(r'P:\windows\Documents\GEO441 - RS Seminar\qgis_files\frontline_area.geojson')
frontline_area_poly = frontline_area.geometry.iloc[0]

bbox = kherson_area.total_bounds
width = int((bbox[2] - bbox[0]) / 30)
height = int((bbox[3] - bbox[1]) / 30)

transform = rio.transform.from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], width, height)

mask = np.zeros((height, width), dtype = np.uint8)

mask = rasterize(
    [(frontline_area_poly, 1)],
    out_shape = (height, width),
    transform = transform,
    fill = 0,
    all_touched=True,
    dtype = np.uint8
)
mask = np.vstack((mask, np.zeros((2, width), dtype=np.uint8)))

meta = {
    'driver': 'GTiff',
    'dtype': rio.uint8,
    'count': 1,
    'width': width,
    'height': height + 2,
    'crs': kherson_area.crs,
    'transform': transform
}

# Write the mask array to a GeoTIFF file
with rio.open(r'P:\windows\Documents\GEO441 - RS Seminar\masks\frontline_mask.tif', 'w', **meta) as dst:
    dst.write(mask, 1)

# %% averages using frontline mask
frontline_mask = rio.open(r'P:\windows\Documents\GEO441 - RS Seminar\masks\frontline_mask.tif').read()
agr_mask = rio.open(r'P:\windows\Documents\GEO441 - RS Seminar\masks\Crops_mask_2021.tif').read()
mask = agr_mask * frontline_mask

avg_list = []
median_list = []
date_list = []
si_list = ['ndvi','evi','msavi','bsi']
for filename in filenames:
    collection = create_collection(data_folder + '/' + filename, band_names, band_aliases, geo_info, mask)
    if np.all(np.isnan(collection.get_values())):
        avg_list.append(np.array([np.nan, np.nan, np.nan, np.nan]))
        median_list.append(np.array([np.nan, np.nan, np.nan, np.nan]))
        date_list.append(collection.scene_properties.acquisition_time)
        continue
    for si in si_list:
        collection.calc_si(si, inplace = True)
    avg_list.append(collection.band_summaries(band_selection= si_list, method = ['mean'])['mean'].values)
    medians = []
    for si in si_list:
        medians.append(np.nanmedian(collection.get_values(band_selection = [si])))
    median_list.append(medians)
    date_list.append(collection.scene_properties.acquisition_time)

#%% plots of front_masks
for i in range(len(si_list)):
    avgs = []
    medians = []
    for entry in avg_list:
        if len(entry) == 4:
            avgs.append(entry[i])
        else:
            avgs.append(np.nan)
    
    for entry in median_list:
        if len(entry) == 4:
            medians.append(entry[i])
        else:
            medians.append(np.nan)
    plt.plot(date_list, avgs)
    plt.title(si_list[i]+'-averages')
    plt.xlabel('date')
    plt.ylabel('index-value')
    plt.savefig(r'P:\windows\Documents\GEO441 - RS Seminar\Plots\timelines_front_mask' + 
                '/' + si_list[i] + '_avg_timeline.png')
    plt.clf()
    plt.plot(date_list, medians)
    plt.title(si_list[i]+'-medians')
    plt.xlabel('date')
    plt.ylabel('index-value')
    plt.savefig(r'P:\windows\Documents\GEO441 - RS Seminar\Plots\timelines_front_mask' + 
                '/' + si_list[i] + '_median_timeline.png')
    plt.clf()

# %% create masks of ua and ru occupied regions
occupied_areas = gpd.read_file(r'P:\windows\Documents\GEO441 - RS Seminar\qgis_files\ua_ru_zones.geojson')
ru_area = occupied_areas.geometry.iloc[0]
ua_area = occupied_areas.geometry.iloc[1]

ru_mask = rasterize(
    [(ru_area, 1)],
    out_shape = (height, width),
    transform = transform,
    fill = 0,
    all_touched=True,
    dtype = np.uint8
)
ru_mask = np.vstack((ru_mask, np.ones((2, width), dtype=np.uint8)))

ua_mask = rasterize(
    [(ua_area, 1)],
    out_shape = (height, width),
    transform = transform,
    fill = 0,
    all_touched=True,
    dtype = np.uint8
)
ua_mask = np.vstack((ua_mask, np.zeros((2, width), dtype=np.uint8)))

with rio.open(r'P:\windows\Documents\GEO441 - RS Seminar\masks\ru_mask.tif', 'w', **meta) as dst:
    dst.write(ru_mask, 1)

with rio.open(r'P:\windows\Documents\GEO441 - RS Seminar\masks\ua_mask.tif', 'w', **meta) as dst:
    dst.write(ua_mask, 1)

#%% averages of ua_area
ua_mask = rio.open(r'P:\windows\Documents\GEO441 - RS Seminar\masks\ua_mask.tif').read()
agr_mask = rio.open(r'P:\windows\Documents\GEO441 - RS Seminar\masks\Crops_mask_2021.tif').read()
mask = agr_mask * ua_mask

avg_list = []
median_list = []
date_list = []
si_list = ['ndvi','evi','msavi','bsi']
for filename in filenames:
    collection = create_collection(data_folder + '/' + filename, band_names, band_aliases, geo_info, mask)
    for si in si_list:
        collection.calc_si(si, inplace = True)
    if np.all(np.isnan(collection.get_values())):
        avg_list.append(np.array([np.nan, np.nan, np.nan, np.nan]))
        median_list.append(np.array([np.nan, np.nan, np.nan, np.nan]))
        date_list.append(collection.scene_properties.acquisition_time)
        continue
    avg_list.append(collection.band_summaries(band_selection= si_list, method = ['mean'])['mean'].values)
    medians = []
    for si in si_list:
        medians.append(np.nanmedian(collection.get_values(band_selection = [si])))
    median_list.append(medians)
    date_list.append(collection.scene_properties.acquisition_time)

#%% plots of ua_area
for i in range(len(si_list)):
    avgs = []
    medians = []
    for entry in avg_list:
        if len(entry) == 4:
            avgs.append(entry[i])
        else:
            avgs.append(np.nan)
    
    for entry in median_list:
        if len(entry) == 4:
            medians.append(entry[i])
        else:
            medians.append(np.nan)
    plt.plot(date_list, avgs)
    plt.title(si_list[i]+'-averages')
    plt.xlabel('date')
    plt.ylabel('index-value')
    plt.savefig(r'P:\windows\Documents\GEO441 - RS Seminar\Plots\timelines_ua_mask' + 
                '/' + si_list[i] + '_avg_timeline.png')
    plt.clf()
    plt.plot(date_list, medians)
    plt.title(si_list[i]+'-medians')
    plt.xlabel('date')
    plt.ylabel('index-value')
    plt.savefig(r'P:\windows\Documents\GEO441 - RS Seminar\Plots\timelines_ua_mask' + 
                '/' + si_list[i] + '_median_timeline.png')
    plt.clf()

#%% averages of ru_area
ru_mask = rio.open(r'P:\windows\Documents\GEO441 - RS Seminar\masks\ru_mask.tif').read()
agr_mask = rio.open(r'P:\windows\Documents\GEO441 - RS Seminar\masks\Crops_mask_2021.tif').read()
mask = agr_mask * ru_mask

avg_list = []
median_list = []
date_list = []
si_list = ['ndvi','evi','msavi','bsi']
for filename in filenames:
    collection = create_collection(data_folder + '/' + filename, band_names, band_aliases, geo_info, mask)
    for si in si_list:
        collection.calc_si(si, inplace = True)
    if np.all(np.isnan(collection.get_values())):
        avg_list.append(np.array([np.nan, np.nan, np.nan, np.nan]))
        median_list.append(np.array([np.nan, np.nan, np.nan, np.nan]))
        date_list.append(collection.scene_properties.acquisition_time)
        continue
    avg_list.append(collection.band_summaries(band_selection= si_list, method = ['mean'])['mean'].values)
    medians = []
    for si in si_list:
        medians.append(np.nanmedian(collection.get_values(band_selection = [si])))
    median_list.append(medians)
    date_list.append(collection.scene_properties.acquisition_time)

#%% plots of ru_area
for i in range(len(si_list)):
    avgs = []
    medians = []
    for entry in avg_list:
        if len(entry) == 4:
            avgs.append(entry[i])
        else:
            avgs.append(np.nan)
    
    for entry in median_list:
        if len(entry) == 4:
            medians.append(entry[i])
        else:
            medians.append(np.nan)
    plt.plot(date_list, avgs)
    plt.title(si_list[i]+'-averages')
    plt.xlabel('date')
    plt.ylabel('index-value')
    plt.savefig(r'P:\windows\Documents\GEO441 - RS Seminar\Plots\timelines_ru_mask' + 
                '/' + si_list[i] + '_avg_timeline.png')
    plt.clf()
    plt.plot(date_list, medians)
    plt.title(si_list[i]+'-medians')
    plt.xlabel('date')
    plt.ylabel('index-value')
    plt.savefig(r'P:\windows\Documents\GEO441 - RS Seminar\Plots\timelines_ru_mask' + 
                '/' + si_list[i] + '_median_timeline.png')
    plt.clf()

# ----------------------------------------------------------------- finish optical data -------------------------------------------------------------------

# %% sar data

sar_folder = r'S:\course\geo441\data\2023_Ukraine\s1\Kherson_30m'
vv_folder = sar_folder + r'\12d_12d_vv'
vh_folder = sar_folder + r'\12d_12d_vh'

vv_filenames = get_filenames(vv_folder)
vh_filenames = get_filenames(vh_folder)


# %% sar dates
sar_date_list = []
for name in vv_filenames:
    date = datetime.strptime(name[12:20],'%Y%m%d') + timedelta(days=6)
    sar_date_list.append(date)

#%% sar averages
masknames = ['frontline_mask', 'single_mask', 'ru_mask', 'ua_mask']

for maskname in masknames:
    avg_list = []
    median_list = []
    if maskname == 'single_mask':
        mask = rio.open(mask_folder + '/Crops_mask_2021.tif').read()[0][:-2]
    else:
        mask = rio.open(mask_folder + '/' + maskname + '.tif').read()[0][:-2]
    
    for i in range(len(vv_filenames)):
        vv_im = rio.open(vv_folder + '/' + vv_filenames[i]).read()[0]
        vh_im = rio.open(vh_folder + '/' + vh_filenames[i]).read()[0]

        rfdi = (vv_im - vh_im) / (vv_im + vh_im)
        rfdi = rfdi * mask
        rfdi[rfdi==0] = np.nan

        avg_list.append(np.nanmean(rfdi))
        median_list.append(np.nanmedian(rfdi))
    
    plt.plot(sar_date_list, avg_list)
    plt.title(maskname + '-rfdi-averages')
    plt.xlabel('date')
    plt.ylabel('rfdi-value')
    plt.savefig(r'P:\windows\Documents\GEO441 - RS Seminar\Plots\sar_plots' + '/' + maskname + '/rfdi_avgs.png')
    plt.clf()
    plt.plot(sar_date_list, median_list)
    plt.title(maskname + '-rfdi-medians')
    plt.xlabel('date')
    plt.ylabel('rfdi-value')
    plt.savefig(r'P:\windows\Documents\GEO441 - RS Seminar\Plots\sar_plots' + '/' + maskname + '/rfdi_medians.png')
    plt.clf()


