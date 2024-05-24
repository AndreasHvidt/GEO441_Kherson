#%% import
import numpy as np
import rasterio as rio
from datetime import datetime, timedelta
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

ndvi_data = defaultdict(lambda: defaultdict(list))
bsi_data = defaultdict(lambda: defaultdict(list))
r_data = defaultdict(lambda: defaultdict(list))
g_data = defaultdict(lambda: defaultdict(list))
b_data = defaultdict(lambda: defaultdict(list))


count = 0
for file in optical_files:
    #count += 1
    #if count > 37:
    #    break
    raster = rio.open(r'S:\course\geo441\data\2023_Ukraine\S2\GEO441_Kherson_large_S2_10d_composites' + '/' + file).read().astype(float)[:,1850:2010,5700:6100]
    raster[raster==0] = np.nan
    ndvi = (raster[3]-raster[2])/(raster[3]+raster[2])
    bsi = ((raster[4]+raster[2]) - (raster[3]+raster[0])) / ((raster[4]+raster[2]) + (raster[3]+raster[0]))
    date = datetime.strptime(file[file.find('UA_Kherson'):].removeprefix('UA_Kherson_S2_30m_').removesuffix('_composite_filt.tif'),'%Y-%m-%d')
    ndvi_data[date.year][date.month].append(ndvi)
    bsi_data[date.year][date.month].append(bsi)
    r_data[date.year][date.month].append(raster[2])
    g_data[date.year][date.month].append(raster[1])
    b_data[date.year][date.month].append(raster[0])

#%% sar data

rvi_data = defaultdict(lambda: defaultdict(list))
rfdi_data = defaultdict(lambda: defaultdict(list))

for i in range(31):
    vv_im = rio.open(vv_folder + '/' + vv_filenames[i]).read()[0][1850:2010,5700:6100]
    vh_im = rio.open(vh_folder + '/' + vh_filenames[i]).read()[0][1850:2010,5700:6100]
    date = datetime.strptime(vv_filenames[i][12:20],'%Y%m%d') + timedelta(days=6)
    q = vh_im / vv_im
    N = q*(q+3)
    D = (q+1)*(q+1)
    RVI = N/D
    rfdi = (vv_im - vh_im) / (vv_im + vh_im)
    rvi_data[date.year][date.month].append(RVI)
    rfdi_data[date.year][date.month].append(rfdi)


#%% optical pixel mean calculation
ndvi_means = defaultdict(lambda: defaultdict(list))
bsi_means = defaultdict(lambda: defaultdict(list))
r_means = defaultdict(lambda: defaultdict(list))
g_means = defaultdict(lambda: defaultdict(list))
b_means = defaultdict(lambda: defaultdict(list))

for year in ndvi_data:
    for month in ndvi_data[year]:
        ndvi_means[year][month] = np.nanmean(np.stack(ndvi_data[year][month], axis = 0), axis = 0)
        bsi_means[year][month] = np.nanmean(np.stack(bsi_data[year][month], axis = 0), axis = 0)
        r_means[year][month] = np.nanmean(np.stack(r_data[year][month], axis = 0), axis = 0)
        g_means[year][month] = np.nanmean(np.stack(g_data[year][month], axis = 0), axis = 0)
        b_means[year][month] = np.nanmean(np.stack(b_data[year][month], axis = 0), axis = 0)

#%% sar pixel mean calculation
rvi_means = defaultdict(lambda: defaultdict(list))
rfdi_means = defaultdict(lambda: defaultdict(list))

for year in rvi_data:
    for month in rvi_data[year]:
        rvi_means[year][month] = np.nanmean(np.stack(rvi_data[year][month], axis = 0), axis = 0)
        rfdi_means[year][month] = np.nanmean(np.stack(rfdi_data[year][month], axis = 0), axis = 0)

#%% generate animation with side by side indices/rgb
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
    12: "December"}

fig3, (ax1,ax2,ax3,ax4) = plt.subplots(1,4, figsize = (20,10))
plt.subplots_adjust(wspace=0.3)

artists = []

for year in ndvi_data:
    for month in ndvi_data[year]:
        r_dat = ((r_means[year][month] - np.nanmin(r_means[year][month])) * (1 / (np.nanmax(r_means[year][month]) - np.nanmin(r_means[year][month])) * 255)).astype("uint8")
        g_dat = ((g_means[year][month] - np.nanmin(g_means[year][month])) * (1 / (np.nanmax(g_means[year][month]) - np.nanmin(g_means[year][month])) * 255)).astype("uint8")
        b_dat = ((b_means[year][month] - np.nanmin(b_means[year][month])) * (1 / (np.nanmax(b_means[year][month]) - np.nanmin(b_means[year][month])) * 255)).astype("uint8")
        stack = np.dstack([r_dat,g_dat,b_dat])
        vmin = np.nanquantile(stack, 0.1)
        vmax = np.nanquantile(stack, 0.9)
        im1 = ax1.imshow(stack, vmin=vmin, vmax=vmax, animated=True)
        ax1.title.set_text('RGB')
        im2 = ax2.imshow(ndvi_means[year][month], vmin = -1, vmax = 1, cmap = 'RdYlGn', animated=True)
        ax2.title.set_text('NDVI')
        im3 = ax3.imshow(bsi_means[year][month], vmin = -1, vmax = 1, cmap = 'RdYlGn_r', animated=True)
        ax3.title.set_text('BSI')
        if len(rvi_means[year][month]) != 0:
            im4 = ax4.imshow(rvi_means[year][month], vmin = 0, vmax = 1, cmap = 'Greens', animated=True)
            ax4.title.set_text('RVI')
        else:
            im4 = ax4.imshow(np.zeros((400, 300)), animated=True)
            ax4.title.set_text('RVI')

        text = ax4.text(0.95, 0.01, f'{year}-{month_names[month]}',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax4.transAxes,
        color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))

        artists.append(([im1,im2,im3,im4,text]))


cbar1 = fig3.colorbar(im2, ax = ax2, fraction= 0.047)
cbar2 = fig3.colorbar(im3, ax = ax3, fraction= 0.047)
cbar2 = fig3.colorbar(im4, ax = ax4, fraction= 0.047)
ani = animation.ArtistAnimation(fig=fig3, artists=artists, interval=200)

writervideo = animation.FFMpegWriter(fps=2)
ani.save(r'P:\windows\Documents\GEO441 - RS Seminar\index_inspection_animation.mp4', writer = writervideo)
plt.close()

# %% generate singular image with side by side indices/rgb
year = 2020
month = 6
fig4, (ax1,ax2,ax3,ax4) = plt.subplots(1,4, figsize = (20,10))


r_dat = ((r_means[year][month] - np.nanmin(r_means[year][month])) * (1 / (np.nanmax(r_means[year][month]) - np.nanmin(r_means[year][month])) * 255)).astype("uint8")
g_dat = ((g_means[year][month] - np.nanmin(g_means[year][month])) * (1 / (np.nanmax(g_means[year][month]) - np.nanmin(g_means[year][month])) * 255)).astype("uint8")
b_dat = ((b_means[year][month] - np.nanmin(b_means[year][month])) * (1 / (np.nanmax(b_means[year][month]) - np.nanmin(b_means[year][month])) * 255)).astype("uint8")
stack = np.dstack([r_dat,g_dat,b_dat])
vmin = np.nanquantile(stack, 0.1)
vmax = np.nanquantile(stack, 0.9)
im1 = ax1.imshow(stack, vmin=vmin, vmax=vmax, animated=True)
ax1.title.set_text('RGB')
im2 = ax2.imshow(ndvi_means[year][month], vmin = -1, vmax = 1, cmap = 'RdYlGn', animated=True)
ax2.title.set_text('NDVI')
im3 = ax3.imshow(bsi_means[year][month], vmin = -1, vmax = 1, cmap = 'RdYlGn_r', animated=True)
ax3.title.set_text('BSI')
im4 = ax4.imshow(rvi_means[year][month], vmin = 0, vmax = 1, cmap = 'Greens', animated=True)
ax4.title.set_text('RVI')


text = ax4.text(0.95, 0.01, f'{year}-{month_names[month]}',
verticalalignment='bottom', horizontalalignment='right',
transform=ax4.transAxes,
color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))
cbar1 = fig4.colorbar(im2, ax = ax2, fraction= 0.047)
cbar2 = fig4.colorbar(im3, ax = ax3, fraction= 0.047)
cbar2 = fig4.colorbar(im4, ax = ax4, fraction= 0.047)
plt.subplots_adjust(wspace=0.3)
plt.savefig(r'P:\windows\Documents\GEO441 - RS Seminar\Plots' + '/' + 'index_comparison_december.png')
plt.clf()

# %%
