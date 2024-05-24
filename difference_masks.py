#%% import
import numpy as np
import rasterio as rio
from datetime import datetime, timedelta
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd

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

#%% optical data loading

ndvi_data = defaultdict(lambda: defaultdict(list))
bsi_data = defaultdict(lambda: defaultdict(list))
mask = rio.open(r'P:\windows\Documents\GEO441 - RS Seminar\masks\Crops_mask_2021.tif').read()[0]
for file in optical_files:
    raster = rio.open(r'S:\course\geo441\data\2023_Ukraine\S2\GEO441_Kherson_large_S2_10d_composites' + '/' + file).read().astype(float)
    raster = mask[np.newaxis, :, :] * raster
    raster[raster==0] = np.nan
    ndvi = (raster[3]-raster[2])/(raster[3]+raster[2])
    bsi = ((raster[4]+raster[2]) - (raster[3]+raster[0])) / ((raster[4]+raster[2]) + (raster[3]+raster[0]))
    date = datetime.strptime(file[file.find('UA_Kherson'):].removeprefix('UA_Kherson_S2_30m_').removesuffix('_composite_filt.tif'),'%Y-%m-%d')
    ndvi_data[date.year][date.month].append(ndvi)
    bsi_data[date.year][date.month].append(bsi)

#%% optical means
ndvi_means = defaultdict(lambda: defaultdict(list))
bsi_means = defaultdict(lambda: defaultdict(list))

# calculate monthly pixelwise means
for year in ndvi_data:
    for month in ndvi_data[year]:
        ndvi_means[year][month] = np.nanmean(np.stack(ndvi_data[year][month], axis = 0), axis = 0)
        bsi_means[year][month] = np.nanmean(np.stack(bsi_data[year][month], axis = 0), axis = 0)
        

#%% calculate & plot monthly differences
ndvi_difference_maps = defaultdict(dict)
bsi_difference_maps = defaultdict(dict)

ndvi_diff_pixels = defaultdict(dict)
bsi_diff_pixels = defaultdict(dict)

month_names = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
}

x_ticks = np.arange(399.930, 399.930 + mask.shape[1] * .30, .30)
y_ticks = np.flip(np.arange(5200.050 - mask.shape[0] * .30, 5200.050, .30))

for year in range(2021,2024):
    for month in ndvi_means[year]:
        if len(ndvi_means[2020][month]) != 0:
            ndvi_difference_maps[year][month] = ndvi_means[year][month] - ndvi_means[2020][month]
            plt.figure(figsize=(18,9.774))
            plt.imshow(ndvi_difference_maps[year][month], cmap='RdYlGn', vmin=-1, vmax=1, extent=[x_ticks[0], x_ticks[-1], y_ticks[-1], y_ticks[0]])
            plt.colorbar(label='Index-difference', fraction = 0.047)
            plt.title(f'NDVI difference {month_names[month]} 2020 - {year}')
            plt.xlabel('X [km] (EPSG:32636)')
            plt.ylabel('Y [km] (EPSG:32636)')
            plt.savefig(r'\\geofiles.d.uzh.ch\private\ahvidt\windows\Documents\GEO441 - RS Seminar\Plots\difference_maps\monthly\ndvi' + '/' + f'{year}' + '_' + f'{month}' + '_ndvi_diffmask.png')
            plt.close()
        if len(bsi_means[2020][month]) != 0:
            bsi_difference_maps[year][month] = bsi_means[year][month] - bsi_means[2020][month]
            bsi_diff_pixels[year][month] = [np.sum(bsi_difference_maps[year][month] > 0)*9e-4, np.sum(bsi_difference_maps[year][month] < 0)*9e-4, np.sum(bsi_difference_maps[year][month] == 0)*9e-4]
            plt.figure(figsize=(18,9.774))
            plt.imshow(bsi_difference_maps[year][month], cmap='RdYlGn_r', vmin=-1, vmax=1, extent=[x_ticks[0], x_ticks[-1], y_ticks[-1], y_ticks[0]])
            plt.colorbar(label='Index-difference', fraction = 0.047)
            plt.title(f'BSI difference {month_names[month]} 2020 - {year}')
            plt.xlabel('X [km] (EPSG:32636)')
            plt.ylabel('Y [km] (EPSG:32636)')
            plt.savefig(r'\\geofiles.d.uzh.ch\private\ahvidt\windows\Documents\GEO441 - RS Seminar\Plots\difference_maps\monthly\bsi' + '/' + f'{year}' + '_' + f'{month}' + '_bsi_diffmask.png')
            plt.close()



# %% calculate & plot yearly differences
ndvi_difference_yearly = defaultdict(dict)
bsi_difference_yearly = defaultdict(dict)

for year in range(2021,2024):
    ndvi_all_months = []
    bsi_all_months = []
    for i in range(1,13):
        ndvi_all_months.append(ndvi_difference_maps[year][i])
        bsi_all_months.append(bsi_difference_maps[year][i])
    ndvi_difference_yearly[year] = np.nanmean(ndvi_all_months, axis = 0)
    bsi_difference_yearly[year] = np.nanmean(bsi_all_months, axis = 0)
    plt.figure(figsize=(18,9.774))
    plt.imshow(ndvi_difference_yearly[year], cmap='RdYlGn', vmin=-1, vmax=1, extent=[x_ticks[0], x_ticks[-1], y_ticks[-1], y_ticks[0]])
    plt.colorbar(label='Index-difference', fraction = 0.047)
    plt.title(f'NDVI difference 2020 - {year}')
    plt.xlabel('X [km] (EPSG:32636)')
    plt.ylabel('Y [km] (EPSG:32636)')
    plt.savefig(r'\\geofiles.d.uzh.ch\private\ahvidt\windows\Documents\GEO441 - RS Seminar\Plots\difference_maps\yearly' + '/' + f'{year}' + '_ndvi_diffmask.png')
    plt.close()
    plt.figure(figsize=(18,9.774))
    plt.imshow(bsi_difference_yearly[year], cmap='RdYlGn_r', vmin=-1, vmax=1, extent=[x_ticks[0], x_ticks[-1], y_ticks[-1], y_ticks[0]])
    plt.colorbar(label='Index-difference', fraction = 0.047)
    plt.title(f'BSI difference 2020 - {year}')
    plt.xlabel('X [km] (EPSG:32636)')
    plt.ylabel('Y [km] (EPSG:32636)')
    plt.savefig(r'\\geofiles.d.uzh.ch\private\ahvidt\windows\Documents\GEO441 - RS Seminar\Plots\difference_maps\yearly' + '/' + f'{year}' + '_bsi_diffmask.png')
    plt.close()





# %% sar data loading & pixel mean calculation
rvi_data = defaultdict(lambda: defaultdict(list))
rfdi_data = defaultdict(lambda: defaultdict(list))

for i in range(len(vv_filenames)):
    vv_im = rio.open(vv_folder + '/' + vv_filenames[i]).read()[0]
    vh_im = rio.open(vh_folder + '/' + vh_filenames[i]).read()[0]
    date = datetime.strptime(vv_filenames[i][12:20],'%Y%m%d') + timedelta(days=6)
    q = vh_im / vv_im
    N = q*(q+3)
    D = (q+1)*(q+1)
    rvi = N/D
    rfdi = (vv_im - vh_im) / (vv_im + vh_im)
    rvi = mask[:-2, :] * rvi
    rvi[rvi==0] = np.nan
    rfdi = mask[:-2, :] * rfdi
    rfdi[rfdi==0] = np.nan
    rvi_data[date.year][date.month].append(rvi)
    rfdi_data[date.year][date.month].append(rfdi)

rvi_means = defaultdict(lambda: defaultdict(list))
rfdi_means = defaultdict(lambda: defaultdict(list))

for year in rvi_data:
    for month in rvi_data[year]:
        rvi_means[year][month] = np.nanmean(np.stack(rvi_data[year][month], axis = 0), axis = 0)
        rfdi_means[year][month] = np.nanmean(np.stack(rfdi_data[year][month], axis = 0), axis = 0)


# %% sar monthly difference maps
rvi_difference_maps = defaultdict(dict)
rfdi_difference_maps = defaultdict(dict)

x_ticks = np.arange(399.930, 399.930 + mask.shape[1] * .30, .30)[:-2]
y_ticks = np.flip(np.arange(5200.050 - mask.shape[0] * .30, 5200.050, .30))[:-2]

for year in range(2021,2024):
    for month in rvi_means[year]:
        if len(rvi_means[2020][month]) != 0:
            rvi_difference_maps[year][month] = rvi_means[year][month] - rvi_means[2020][month]
            plt.figure(figsize=(18,9.774))
            plt.imshow(rvi_difference_maps[year][month], cmap='RdYlGn', vmin=-1, vmax=1, extent=[x_ticks[0], x_ticks[-1], y_ticks[-1], y_ticks[0]])
            plt.colorbar(label='Index-difference', fraction = 0.047)
            plt.title(f'RVI difference {month_names[month]} 2020 - {year}')
            plt.xlabel('X [km] (EPSG:32636)')
            plt.ylabel('Y [km] (EPSG:32636)')
            plt.savefig(r'\\geofiles.d.uzh.ch\private\ahvidt\windows\Documents\GEO441 - RS Seminar\Plots\difference_maps\monthly\rvi' + '/' + f'{year}' + '_' + f'{month}' + '_rvi_diffmask.png')
            plt.close()
            rfdi_difference_maps[year][month] = rfdi_means[year][month] - rfdi_means[2020][month]
            plt.figure(figsize=(18,9.774))
            plt.imshow(rfdi_difference_maps[year][month], cmap='RdYlGn_r', vmin=-1, vmax=1, extent=[x_ticks[0], x_ticks[-1], y_ticks[-1], y_ticks[0]])
            plt.colorbar(label='Index-difference', fraction = 0.047)
            plt.title(f'RFDI difference {month_names[month]} 2020 - {year}')
            plt.xlabel('X [km] (EPSG:32636)')
            plt.ylabel('Y [km] (EPSG:32636)')
            plt.savefig(r'\\geofiles.d.uzh.ch\private\ahvidt\windows\Documents\GEO441 - RS Seminar\Plots\difference_maps\monthly\rfdi' + '/' + f'{year}' + '_' + f'{month}' + '_rfdi_diffmask.png')
            plt.close()



# %% sar yearly difference maps
rvi_difference_yearly = defaultdict(dict)
rfdi_difference_yearly = defaultdict(dict)

for year in range(2021,2024):
    rvi_all_months = []
    rfdi_all_months = []
    for i in range(1,13):
        rvi_all_months.append(rvi_difference_maps[year][i])
        rfdi_all_months.append(rfdi_difference_maps[year][i])
    rvi_difference_yearly[year] =  np.nanmean(rvi_all_months, axis = 0)
    rfdi_difference_yearly[year] = np.nanmean(rfdi_all_months, axis = 0)
    plt.figure(figsize=(18,9.774))
    plt.imshow(rvi_difference_yearly[year], cmap='RdYlGn', vmin=-1, vmax=1, extent=[x_ticks[0], x_ticks[-1], y_ticks[-1], y_ticks[0]])
    plt.colorbar(label='Index-difference', fraction = 0.047)
    plt.title(f'RVI difference 2020 - {year}')
    plt.xlabel('X [km] (EPSG:32636)')
    plt.ylabel('Y [km] (EPSG:32636)')
    plt.savefig(r'\\geofiles.d.uzh.ch\private\ahvidt\windows\Documents\GEO441 - RS Seminar\Plots\difference_maps\yearly' + '/' + f'{year}' + '_rvi_diffmask.png')
    plt.close()
    plt.figure(figsize=(18,9.774))
    plt.imshow(rfdi_difference_yearly[year], cmap='RdYlGn_r', vmin=-1, vmax=1, extent=[x_ticks[0], x_ticks[-1], y_ticks[-1], y_ticks[0]])
    plt.colorbar(label='Index-difference', fraction = 0.047)
    plt.title(f'RFDI difference 2020 - {year}')
    plt.xlabel('X [km] (EPSG:32636)')
    plt.ylabel('Y [km] (EPSG:32636)')
    plt.savefig(r'\\geofiles.d.uzh.ch\private\ahvidt\windows\Documents\GEO441 - RS Seminar\Plots\difference_maps\yearly' + '/' + f'{year}' + '_rfdi_diffmask.png')
    plt.close()

#%% monthly difference area calculation & saving into excel sheet
rownames = []
diff_values = []

for year in ndvi_difference_maps:
    for month in ndvi_difference_maps[year]:
        rownames.append(f'{year}_{month}')
        diff_values.append([np.sum(ndvi_difference_maps[year][month] > 0.01)*9e-4,
                      np.sum(ndvi_difference_maps[year][month] < -0.01)*9e-4,
                      np.sum((ndvi_difference_maps[year][month] < 0.01)&(ndvi_difference_maps[year][month]>-0.01) )*9e-4,
                      np.sum(bsi_difference_maps[year][month] > 0.01)*9e-4,
                      np.sum(bsi_difference_maps[year][month] < -0.01)*9e-4,
                      np.sum((bsi_difference_maps[year][month] < 0.01)&(bsi_difference_maps[year][month]>-0.01) )*9e-4,
                      np.sum(rvi_difference_maps[year][month] > 0.01)*9e-4,
                      np.sum(rvi_difference_maps[year][month] < -0.01)*9e-4,
                      np.sum((rvi_difference_maps[year][month] < 0.01)&(rvi_difference_maps[year][month]>-0.01) )*9e-4,
                      np.sum(rfdi_difference_maps[year][month] > 0.01)*9e-4,
                      np.sum(rfdi_difference_maps[year][month] < -0.01)*9e-4,
                      np.sum((rfdi_difference_maps[year][month] < 0.01)&(rfdi_difference_maps[year][month]>-0.01) )*9e-4
                      ])

diff_values = np.array(diff_values)        

df = pd.DataFrame({
'Year_Month': rownames,
'ndvi_pos': diff_values[:,0],
'ndvi_neg': diff_values[:,1],
'ndvi_neutr': diff_values[:,2],
'bsi_pos': diff_values[:,3],
'bsi_neg': diff_values[:,4],
'bsi_neutr': diff_values[:,5],
'rvi_pos': diff_values[:,6],
'rvi_neg': diff_values[:,7],
'rvi_neutr': diff_values[:,8],
'rfdi_pos': diff_values[:,9],
'rfdi_neg': diff_values[:,10],
'rfdi_neutr': diff_values[:,11]
})

df.to_excel(r'P:\windows\Documents\GEO441 - RS Seminar\monthly_diff_areas.xlsx', index = False)

# %% yearly difference area calculation & saving into excel sheet
rownames = []
diff_values = []

for year in ndvi_difference_yearly:
    rownames.append(f'{year}')
    diff_values.append([np.sum(ndvi_difference_yearly[year] > 0.01)*9e-4,
                    np.sum(ndvi_difference_yearly[year] < -0.01)*9e-4,
                    np.sum((ndvi_difference_yearly[year] < 0.01)&(ndvi_difference_yearly[year]>-0.01) )*9e-4,
                    np.sum(bsi_difference_yearly[year] > 0.01)*9e-4,
                    np.sum(bsi_difference_yearly[year] < -0.01)*9e-4,
                    np.sum((bsi_difference_yearly[year] < 0.01)&(bsi_difference_yearly[year]>-0.01) )*9e-4,
                    np.sum(rvi_difference_yearly[year] > 0.01)*9e-4,
                    np.sum(rvi_difference_yearly[year] < -0.01)*9e-4,
                    np.sum((rvi_difference_yearly[year] < 0.01)&(rvi_difference_yearly[year]>-0.01) )*9e-4,
                    np.sum(rfdi_difference_yearly[year] > 0.01)*9e-4,
                    np.sum(rfdi_difference_yearly[year] < -0.01)*9e-4,
                    np.sum((rfdi_difference_yearly[year] < 0.01)&(rfdi_difference_yearly[year]>-0.01) )*9e-4
                    ])

        

diff_values = np.array(diff_values)        

df = pd.DataFrame({
'Year_Month': rownames,
'ndvi_pos': diff_values[:,0],
'ndvi_neg': diff_values[:,1],
'ndvi_neutr': diff_values[:,2],
'bsi_pos': diff_values[:,3],
'bsi_neg': diff_values[:,4],
'bsi_neutr': diff_values[:,5],
'rvi_pos': diff_values[:,6],
'rvi_neg': diff_values[:,7],
'rvi_neutr': diff_values[:,8],
'rfdi_pos': diff_values[:,9],
'rfdi_neg': diff_values[:,10],
'rfdi_neutr': diff_values[:,11]
})

df.to_excel(r'P:\windows\Documents\GEO441 - RS Seminar\yearly_diff_areas.xlsx', index = False)
# %% calculation of areas where monthly ndvi > 0.5

rownames = []
agr_area = []

for year in ndvi_means:
    for month in ndvi_means[year]:
        rownames.append(f'{year}_{month}')
        agr_area.append(np.sum(ndvi_means[year][month] > 0.5))

df = pd.DataFrame({
    'Year_Month': rownames,
    '#pixel ndvi > 0.5': agr_area
})

df.to_excel(r'P:\windows\Documents\GEO441 - RS Seminar\area_ndvi_threshold.xlsx', index = False)



# %% masks
mask_2021 = rio.open(r'P:\windows\Documents\GEO441 - RS Seminar\masks\Crops_mask_2021.tif').read()[0].astype(int)
mask_2022 = rio.open(r'P:\windows\Documents\GEO441 - RS Seminar\masks\Crops_mask_2022.tif').read()[0].astype(int)
mask_2023 = rio.open(r'P:\windows\Documents\GEO441 - RS Seminar\masks\Crops_mask_2023.tif').read()[0].astype(int)
mask_list = [mask_2021, mask_2022, mask_2023]
#%% masks diffmaps
x_ticks = np.arange(399.930, 399.930 + mask_2021.shape[1] * .30, .30)
y_ticks = np.flip(np.arange(5200.050 - (mask_2021.shape[0]-2) * .30, 5200.050, .30))
year_list = ['2021', '2022','2023']
for i in range(2):
    plt.figure(figsize=(20,12))
    plt.imshow(mask_list[i]-mask_list[i+1], vmin = -1, vmax = 1, cmap = 'RdYlGn_r',extent=[x_ticks[0], x_ticks[-1], y_ticks[-1], y_ticks[0]])
    plt.title(f'Dynamic world Agriculture-classification changes from {year_list[i]} to {year_list[i+1]}')
    plt.xlabel('X [km] (EPSG:32636)')
    plt.ylabel('Y [km] (EPSG:32636)')
    plt.savefig(r'\\geofiles.d.uzh.ch\private\ahvidt\windows\Documents\GEO441 - RS Seminar\Plots\difference_maps\mask_differences' + '/' + f'{year_list[i]}' + '-' + f'{year_list[i+1]}' + '_dynamworld_classification_diffmap.png')
    plt.close()
# %% 2021-2023 mask difference plot
plt.figure(figsize=(20,12))
plt.imshow(mask_list[0]-mask_list[2], vmin = -1, vmax = 1, cmap = 'RdYlGn_r',extent=[x_ticks[0], x_ticks[-1], y_ticks[-1], y_ticks[0]])
plt.title(f'Dynamic world Agriculture-classification changes from {year_list[0]} to {year_list[2]}')
plt.xlabel('X [km] (EPSG:32636)')
plt.ylabel('Y [km] (EPSG:32636)')
plt.savefig(r'\\geofiles.d.uzh.ch\private\ahvidt\windows\Documents\GEO441 - RS Seminar\Plots\difference_maps\mask_differences' + '/' + f'{year_list[0]}' + '-' + f'{year_list[2]}' + '_dynamworld_classification_diffmap.png')
plt.close()

# %% quantitative area change
neg_values = [np.sum((mask_2021-mask_2022) == 1), np.sum((mask_2022-mask_2023)== 1), np.sum((mask_2021-mask_2023)== 1)]
pos_values = [np.sum((mask_2021-mask_2022)== -1), np.sum((mask_2022-mask_2023)== -1), np.sum((mask_2021-mask_2023)== -1)]
neutr_values = [np.sum((mask_2021-mask_2022)== 0), np.sum((mask_2022-mask_2023) == 0), np.sum((mask_2021-mask_2023)== 0)]
rownames = ['2021-2022', '2022-2023', '2021-2023']
df = pd.DataFrame({
    'diff_years': rownames,
    'negative_change[#pixel]': neg_values,
    'positive_change[#pixel]': pos_values,
    'no_change[#pixel]': neutr_values,
})

df.to_excel(r'P:\windows\Documents\GEO441 - RS Seminar\mask_diff_areas.xlsx', index = False)

# %%
diffmask = mask_2021-mask_2022


# %%
