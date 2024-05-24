#%% import
import numpy as np
import rasterio as rio
from datetime import datetime, timedelta, date
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from geojson import Polygon
import pyproj
import geojson

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

#%% optical mean values
ndvi_means = defaultdict(lambda: defaultdict(list))
bsi_means = defaultdict(lambda: defaultdict(list))
r_means = defaultdict(lambda: defaultdict(list))
g_means = defaultdict(lambda: defaultdict(list))
b_means = defaultdict(lambda: defaultdict(list))
nir_means = defaultdict(lambda: defaultdict(list))

for year in ndvi_data:
    for month in ndvi_data[year]:
        ndvi_vals = []
        bsi_vals = []
        r_vals = []
        g_vals = []
        b_vals = []
        nir_vals = []
        for day in ndvi_data[year][month]:
            ndvi_vals.append(ndvi_data[year][month][day])
            bsi_vals.append(bsi_data[year][month][day])
            r_vals.append(r_data[year][month][day])
            g_vals.append(g_data[year][month][day])
            b_vals.append(b_data[year][month][day])
            nir_vals.append(nir_data[year][month][day])
        ndvi_means[year][month] = np.nanmean(np.stack(ndvi_vals, axis = 0), axis = 0)
        bsi_means[year][month] = np.nanmean(np.stack(bsi_vals, axis = 0), axis = 0)
        r_means[year][month] = np.nanmean(np.stack(r_vals, axis = 0), axis = 0)
        g_means[year][month] = np.nanmean(np.stack(g_vals, axis = 0), axis = 0)
        b_means[year][month] = np.nanmean(np.stack(b_vals, axis = 0), axis = 0)
        nir_means[year][month] = np.nanmean(np.stack(nir_vals, axis = 0), axis = 0)
# %% sar mean values
rvi_means = defaultdict(lambda: defaultdict(list))
rfdi_means = defaultdict(lambda: defaultdict(list))

for year in rvi_data:
    for month in rvi_data[year]:
        rvi_vals = []
        rfdi_vals = []
        for day in rvi_data[year][month]:
            rvi_vals.append(rvi_data[year][month][day])
            rfdi_vals.append(rvi_data[year][month][day])
        rvi_means[year][month] = np.nanmean(np.stack(rvi_vals, axis = 0), axis = 0)
        rfdi_means[year][month] = np.nanmean(np.stack(rfdi_vals, axis = 0), axis = 0)

# %% index monthly growing season plots
ulx, uly = 399930 + (5700 * 30), 5200050 - (1850 * 30)
x_ticks = np.arange(ulx * 1e-3, (ulx * 1e-3) + ndvi_means[2020][1].shape[1] * .30, .30)
y_ticks = np.flip(np.arange((uly * 1e-3) - ndvi_means[2020][1].shape[0] * .30, (uly * 1e-3), .30))

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

for year in ndvi_means:
    for month in ndvi_means[year]:
        plt.figure(figsize=(18,9.774))
        plt.imshow(ndvi_means[year][month], cmap='RdYlGn', vmin=-1, vmax=1, extent=[x_ticks[0], x_ticks[-1], y_ticks[-1], y_ticks[0]])
        plt.colorbar(label='Index-value', fraction = 0.02)
        plt.title(f'NDVI {month_names[month]} {year}')
        plt.xlabel('X [km] (EPSG:32636)')
        plt.ylabel('Y [km] (EPSG:32636)')
        plt.savefig(r'\\geofiles.d.uzh.ch\private\ahvidt\windows\Documents\GEO441 - RS Seminar\Plots\indices_evolution\ndvi' + '/' + f'{year}' + '_' + f'{month}' + '.png')
        plt.close()
        plt.figure(figsize=(18,9.774))
        plt.imshow(bsi_means[year][month], cmap='RdYlGn', vmin=-1, vmax=1, extent=[x_ticks[0], x_ticks[-1], y_ticks[-1], y_ticks[0]])
        plt.colorbar(label='Index-value', fraction = 0.02)
        plt.title(f'BSI {month_names[month]} {year}')
        plt.xlabel('X [km] (EPSG:32636)')
        plt.ylabel('Y [km] (EPSG:32636)')
        plt.savefig(r'\\geofiles.d.uzh.ch\private\ahvidt\windows\Documents\GEO441 - RS Seminar\Plots\indices_evolution\bsi' + '/' + f'{year}' + '_' + f'{month}' + '.png')
        plt.close()
        plt.figure(figsize=(18,9.774))
        plt.imshow(ndvi_means[year][month], cmap='RdYlGn', vmin=-1, vmax=1, extent=[x_ticks[0], x_ticks[-1], y_ticks[-1], y_ticks[0]])
        plt.colorbar(label='Index-value', fraction = 0.02)
        plt.title(f'NDVI {month_names[month]} {year}')
        plt.xlabel('X [km] (EPSG:32636)')
        plt.ylabel('Y [km] (EPSG:32636)')
        plt.savefig(r'\\geofiles.d.uzh.ch\private\ahvidt\windows\Documents\GEO441 - RS Seminar\Plots\indices_evolution\ndvi' + '/' + f'{year}' + '_' + f'{month}' + '.png')
        plt.close()
        r_dat = ((r_means[year][month] - np.nanmin(r_means[year][month])) * (1 / (np.nanmax(r_means[year][month]) - np.nanmin(r_means[year][month])) * 255)).astype("uint8")
        g_dat = ((g_means[year][month] - np.nanmin(g_means[year][month])) * (1 / (np.nanmax(g_means[year][month]) - np.nanmin(g_means[year][month])) * 255)).astype("uint8")
        b_dat = ((b_means[year][month] - np.nanmin(b_means[year][month])) * (1 / (np.nanmax(b_means[year][month]) - np.nanmin(b_means[year][month])) * 255)).astype("uint8")
        stack = np.dstack([r_dat,g_dat,b_dat])
        vmin = np.nanquantile(stack, 0.1)
        vmax = np.nanquantile(stack, 0.9)
        plt.figure(figsize=(18,9.774))
        plt.imshow(stack, vmin=vmin, vmax=vmax, extent=[x_ticks[0], x_ticks[-1], y_ticks[-1], y_ticks[0]])
        plt.title(f'RGB {month_names[month]} {year}')
        plt.xlabel('X [km] (EPSG:32636)')
        plt.ylabel('Y [km] (EPSG:32636)')
        plt.savefig(r'\\geofiles.d.uzh.ch\private\ahvidt\windows\Documents\GEO441 - RS Seminar\Plots\indices_evolution\rgb' + '/' + f'{year}' + '_' + f'{month}' + '.png')
        plt.close()
        plt.figure(figsize=(18,9.774))
        plt.imshow(rvi_means[year][month], cmap='Greens', vmin=0, vmax=1, extent=[x_ticks[0], x_ticks[-1], y_ticks[-1], y_ticks[0]])
        plt.colorbar(label='Index-value', fraction = 0.02)
        plt.title(f'RVI {month_names[month]} {year}')
        plt.xlabel('X [km] (EPSG:32636)')
        plt.ylabel('Y [km] (EPSG:32636)')
        plt.savefig(r'\\geofiles.d.uzh.ch\private\ahvidt\windows\Documents\GEO441 - RS Seminar\Plots\indices_evolution\rvi' + '/' + f'{year}' + '_' + f'{month}' + '.png')
        plt.close()
        plt.figure(figsize=(18,9.774))
        plt.imshow(rfdi_means[year][month], cmap='Greens_r', vmin=0, vmax=1, extent=[x_ticks[0], x_ticks[-1], y_ticks[-1], y_ticks[0]])
        plt.colorbar(label='Index-value', fraction = 0.02)
        plt.title(f'RFDI {month_names[month]} {year}')
        plt.xlabel('X [km] (EPSG:32636)')
        plt.ylabel('Y [km] (EPSG:32636)')
        plt.savefig(r'\\geofiles.d.uzh.ch\private\ahvidt\windows\Documents\GEO441 - RS Seminar\Plots\indices_evolution\rfdi' + '/' + f'{year}' + '_' + f'{month}' + '.png')
        plt.close()


# %% masks

mask_2021 = rio.open(r'P:\windows\Documents\GEO441 - RS Seminar\masks\Crops_mask_2021.tif').read()[0]
mask_2022 = rio.open(r'P:\windows\Documents\GEO441 - RS Seminar\masks\Crops_mask_2022.tif').read()[0]
mask_2023 = rio.open(r'P:\windows\Documents\GEO441 - RS Seminar\masks\Crops_mask_2023.tif').read()[0]

# %% agriculture change mask
agr_change_mask = (mask_2021.astype(int) - mask_2022.astype(int))[1850:2010,5700:6100]
agr_change_mask[agr_change_mask != 1] = 0

# %% data lineplots
flag = True
np.random.seed(5)
lines_per_cat = 30
lines = []
data_list = [rfdi_data]
for i in range(2*lines_per_cat+1):
    lines.append([])
for year in range(2021,2023):
    for month in data_list[0][year]:
        for day in data_list[0][year][month]:
            if flag:
                non_agr_coords = np.argwhere(agr_change_mask == 1)
                agr_coords = np.argwhere(mask_2021[1850:2010,5700:6100] + agr_change_mask == 1)
                selected_non_agr_coords = non_agr_coords[np.random.choice(non_agr_coords.shape[0], lines_per_cat, replace = False)]
                selected_agr_coords = agr_coords[np.random.choice(agr_coords.shape[0], lines_per_cat, replace = False)]
                flag = False
            lines[0].append(date(year,month,day))
            for i in range(lines_per_cat*2):
                if i < lines_per_cat:
                    lines[i+1].append(data_list[0][year][month][day][selected_agr_coords[i][0],selected_agr_coords[i][1]])
                else:
                    lines[i+1].append(data_list[0][year][month][day][selected_non_agr_coords[i-lines_per_cat][0],selected_non_agr_coords[i-lines_per_cat][1]])
            
plt.figure(figsize=(10,6))
for i, values in enumerate(lines[1:]):
    if i < lines_per_cat:
        plt.plot(lines[0],values, label=f'agr_{i+1}', color = 'Green')
    else:
        plt.plot(lines[0],values, label=f'non_agr_{i-lines_per_cat+1}', color = 'Red', alpha = 0.5)


# %% coordinates of data line plots on rgb image
year = 2021
month = 1
day = 6
artists = []
fig, ax = plt.subplots(1,1)
for year in r_data:
    for month in r_data[year]:
        for day in r_data[year][month]:
            r_dat = ((r_data[year][month][day] - np.nanmin(r_data[year][month][day])) * (1 / (np.nanmax(r_data[year][month][day]) - np.nanmin(r_data[year][month][day])) * 255)).astype("uint8")
            g_dat = ((g_data[year][month][day] - np.nanmin(g_data[year][month][day])) * (1 / (np.nanmax(g_data[year][month][day]) - np.nanmin(g_data[year][month][day])) * 255)).astype("uint8")
            b_dat = ((b_data[year][month][day] - np.nanmin(b_data[year][month][day])) * (1 / (np.nanmax(b_data[year][month][day]) - np.nanmin(b_data[year][month][day])) * 255)).astype("uint8")
            stack = np.dstack([r_dat,g_dat,b_dat])
            vmin = np.nanquantile(stack, 0.1)
            vmax = np.nanquantile(stack, 0.9)
            im = ax.imshow(stack, vmin=vmin, vmax=vmax, animated=True)
            text = ax.text(0.95, 0.01, f'{year}-{month_names[month]}',
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,
            color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))
            ax.scatter(selected_agr_coords[:,1],selected_agr_coords[:,0], color = 'yellow')
            ax.scatter(selected_non_agr_coords[:,1],selected_non_agr_coords[:,0], color = 'red')
            
            artists.append([im, text])
            

ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=200)
writervideo = animation.FFMpegWriter(fps=2)
ani.save(r'P:\windows\Documents\GEO441 - RS Seminar\rgb_agr_inspection_animation.mp4', writer = writervideo)
plt.close()

# %% grass plot 
grassfield_indices_x = np.array([2716,2736])
grassfield_indices_y = np.array([1032,1038])
agr_indices_x = np.array([2069,2130])
agr_indices_y = np.array([753,779])
agr2_indices_x = np.array([4550,4586])
agr2_indices_y = np.array([1077,1110])
agr3_indices_x = np.array([4418,4435])
agr3_indices_y = np.array([1059,1079])
gra2_indices_x = np.array([2269,2273])
gra2_indices_y = np.array([1170,1177])
gra3_indices_x = np.array([2187,2199])
gra3_indices_y = np.array([1115,1128])


points = []
for i in range(2):
    for j in range(2):
        points.append([grassfield_indices_x[i],grassfield_indices_y[j]])
points = np.array(points)

year = 2021
month = 1
day = 16
r_dat = ((r_data[year][month][day] - np.nanmin(r_data[year][month][day])) * (1 / (np.nanmax(r_data[year][month][day]) - np.nanmin(r_data[year][month][day])) * 255)).astype("uint8")[1000:1050,2700:2800]
g_dat = ((g_data[year][month][day] - np.nanmin(g_data[year][month][day])) * (1 / (np.nanmax(g_data[year][month][day]) - np.nanmin(g_data[year][month][day])) * 255)).astype("uint8")[1000:1050,2700:2800]
b_dat = ((b_data[year][month][day] - np.nanmin(b_data[year][month][day])) * (1 / (np.nanmax(b_data[year][month][day]) - np.nanmin(b_data[year][month][day])) * 255)).astype("uint8")[1000:1050,2700:2800]
stack = np.dstack([r_dat,g_dat,b_dat])
vmin = np.nanquantile(stack, 0.1)
vmax = np.nanquantile(stack, 0.9)
plt.imshow(stack, vmin=vmin, vmax=vmax)
plt.scatter(points[:,0]-2700, points[:,1]-1000, color = 'yellow')

# %% grass - agr lineplots
grass_lines = []
agr_lines = []
agr2_lines = []
agr3_lines = []
gra2_lines = []
gra3_lines = []

for i in range(((grassfield_indices_x[1]-grassfield_indices_x[0]) * (grassfield_indices_y[1]-grassfield_indices_y[0]))+1):
    grass_lines.append([])
for i in range(((agr_indices_x[1]-agr_indices_x[0]) * (agr_indices_y[1]-agr_indices_y[0]))+1):
    agr_lines.append([])
for i in range(((agr2_indices_x[1]-agr2_indices_x[0]) * (agr2_indices_y[1]-agr2_indices_y[0]))+1):
    agr2_lines.append([])
for i in range(((agr3_indices_x[1]-agr3_indices_x[0]) * (agr3_indices_y[1]-agr3_indices_y[0]))+1):
    agr3_lines.append([])
for i in range(((gra2_indices_x[1]-gra2_indices_x[0]) * (gra2_indices_y[1]-gra2_indices_y[0]))+1):
    gra2_lines.append([])
for i in range(((gra3_indices_x[1]-gra3_indices_x[0]) * (gra3_indices_y[1]-gra3_indices_y[0]))+1):
    gra3_lines.append([])

data_list = [rvi_data]
for year in range(2020,2023):
    for month in data_list[0][year]:
        for day in data_list[0][year][month]:
            grass_lines[0].append(date(year,month,day))
            agr_lines[0].append(date(year,month,day))
            for i in range(grassfield_indices_y[1]-grassfield_indices_y[0]):
                for j in range(grassfield_indices_x[1]-grassfield_indices_x[0]):
                    grass_lines[(i*(grassfield_indices_x[1]-grassfield_indices_x[0]))+j+1].append(data_list[0][year][month][day][grassfield_indices_y[0]+i,grassfield_indices_x[0]+j])
            for i in range(agr_indices_y[1]-agr_indices_y[0]):
                for j in range(agr_indices_x[1]-agr_indices_x[0]):
                    agr_lines[(i*(agr_indices_x[1]-agr_indices_x[0]))+j+1].append(data_list[0][year][month][day][agr_indices_y[0]+i,agr_indices_x[0]+j])
            for i in range(agr2_indices_y[1]-agr2_indices_y[0]):
                for j in range(agr2_indices_x[1]-agr2_indices_x[0]):
                    agr2_lines[(i*(agr2_indices_x[1]-agr2_indices_x[0]))+j+1].append(data_list[0][year][month][day][agr2_indices_y[0]+i,agr2_indices_x[0]+j])
            for i in range(agr3_indices_y[1]-agr3_indices_y[0]):
                for j in range(agr3_indices_x[1]-agr3_indices_x[0]):
                    agr3_lines[(i*(agr3_indices_x[1]-agr3_indices_x[0]))+j+1].append(data_list[0][year][month][day][agr3_indices_y[0]+i,agr3_indices_x[0]+j])
            for i in range(gra2_indices_y[1]-gra2_indices_y[0]):
                for j in range(gra2_indices_x[1]-gra2_indices_x[0]):
                    gra2_lines[(i*(gra2_indices_x[1]-gra2_indices_x[0]))+j+1].append(data_list[0][year][month][day][gra2_indices_y[0]+i,gra2_indices_x[0]+j])
            for i in range(gra3_indices_y[1]-gra3_indices_y[0]):
                for j in range(gra3_indices_x[1]-gra3_indices_x[0]):
                    gra3_lines[(i*(gra3_indices_x[1]-gra3_indices_x[0]))+j+1].append(data_list[0][year][month][day][gra3_indices_y[0]+i,gra3_indices_x[0]+j])

#%% grass - agr plotting
plt.figure(figsize=(10,6))
for i, values in enumerate(grass_lines[1:]):
    plt.plot(grass_lines[0], values, color = 'Green')
#for i, values in enumerate(agr_lines[1:]):
    #plt.plot(agr_lines[0], values, color = 'red', alpha = 0.2)
#for i, values in enumerate(agr3_lines[1:]):
    #plt.plot(agr_lines[0], values, color = 'blue', alpha = 0.2)
for i, values in enumerate(agr2_lines[1:]):
    plt.plot(agr_lines[0], values, color = 'pink', alpha = 0.1)
for i, values in enumerate(gra2_lines[1:]):
    plt.plot(grass_lines[0], values, color = 'red', alpha = 0.1)
for i, values in enumerate(gra3_lines[1:]):
    plt.plot(grass_lines[0], values, color = 'black', alpha = 0.1)




# %% grass - agr plotting means
plt.figure(figsize=(10,6))
plt.plot(grass_lines[0], np.nanmean(grass_lines[1:], axis = 0), label = 'grassland', color = 'Purple')
plt.plot(agr_lines[0], np.nanmean(agr_lines[1:], axis = 0), label = 'agricultural', color = 'Green')
plt.plot(agr_lines[0], np.nanmean(agr2_lines[1:], axis = 0), color = 'Green')
plt.plot(agr_lines[0], np.nanmean(agr3_lines[1:], axis = 0), color = 'Green')
plt.plot(agr_lines[0], np.nanmean(gra2_lines[1:], axis = 0), color = 'Purple')
plt.plot(agr_lines[0], np.nanmean(gra3_lines[1:], axis = 0), color = 'Purple')
plt.xlabel('Date')
plt.ylabel('RVI')
plt.title('RVI for Agricultural- and Grassland-areas')
plt.legend()
plt.savefig(r'P:\windows\Documents\GEO441 - RS Seminar\Plots' + '/' + 'RVI_agr_vs_grass.png')


# %% calculate maximum rvi differences for different areas
print('Gra1 maxdiff 2020 =', np.nanmax(np.nanmean(grass_lines[1:], axis = 0))-np.nanmin(np.nanmean(grass_lines[1:], axis = 0)))
print('Gra2 maxdiff 2020 =', np.nanmax(np.nanmean(gra2_lines[1:], axis = 0))-np.nanmin(np.nanmean(gra2_lines[1:], axis = 0)))
print('Gra3 maxdiff 2020 =', np.nanmax(np.nanmean(gra3_lines[1:], axis = 0))-np.nanmin(np.nanmean(gra3_lines[1:], axis = 0)))
print('Agr1 maxdiff 2020 =', np.nanmax(np.nanmean(agr_lines[1:], axis = 0))-np.nanmin(np.nanmean(agr_lines[1:], axis = 0)))
print('Agr1 maxdiff 2020 =', np.nanmax(np.nanmean(agr2_lines[1:], axis = 0))-np.nanmin(np.nanmean(agr2_lines[1:], axis = 0)))
print('Agr1 maxdiff 2020 =', np.nanmax(np.nanmean(agr3_lines[1:], axis = 0))-np.nanmin(np.nanmean(agr3_lines[1:], axis = 0)))

# %% save geometries as geojson for visualization
def create_coords(x,y):
    x_coords = [x[0], x[1], x[1], x[0], x[0]]
    y_coords = [y[0], y[0], y[1], y[1], y[0]]
    return x_coords, y_coords

gra1_x, gra1_y = create_coords((grassfield_indices_x * 30) + 399930, 5200050 - (grassfield_indices_y * 30))
gra2_x, gra2_y = create_coords((gra2_indices_x * 30) + 399930, 5200050 - (gra2_indices_y* 30))
gra3_x, gra3_y = create_coords((gra3_indices_x * 30) + 399930, 5200050 - (gra3_indices_y* 30))
agr1_x, agr1_y = create_coords((agr_indices_x * 30) + 399930, 5200050 - (agr_indices_y* 30))
agr2_x, agr2_y = create_coords((agr2_indices_x * 30) + 399930, 5200050 - (agr2_indices_y* 30))
agr3_x, agr3_y = create_coords((agr3_indices_x * 30) + 399930, 5200050 - (agr3_indices_y* 30))

transformer = pyproj.Transformer.from_crs("EPSG:32636", "EPSG:4326", always_xy=True)

gra1_coords = [(int(x),int(y)) for x,y in zip(gra1_x, gra1_y)]
gra2_coords = [(int(x),int(y)) for x,y in zip(gra2_x, gra2_y)]
gra3_coords = [(int(x),int(y)) for x,y in zip(gra3_x, gra3_y)]
agr1_coords = [(int(x),int(y)) for x,y in zip(agr1_x, agr1_y)]
agr2_coords = [(int(x),int(y)) for x,y in zip(agr2_x, agr2_y)]
agr3_coords = [(int(x),int(y)) for x,y in zip(agr3_x, agr3_y)]

coord_list = [gra1_coords, gra2_coords, gra3_coords, agr1_coords, agr2_coords, agr3_coords]
name_list = ['gra1','gra2','gra3','agr1','agr2','agr3',]
feature_list = []
for i, coords in enumerate(coord_list):
    polygon = Polygon([coords])
    feature_list.append(geojson.Feature(geometry=polygon, id = name_list[i]))
#polygon = Polygon()
#feature = geojson.Feature(geometry=polygon)
feature_collection = geojson.FeatureCollection(feature_list)

with open(r'P:\windows\Documents\GEO441 - RS Seminar\qgis_files\rvi_aois.geojson', 'w') as f:
    geojson.dump(feature_collection, f)



# %%
