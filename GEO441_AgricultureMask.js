// Defining geometry
var geometry = ee.Geometry({type: 'Polygon', coordinates: [[[399930+15, 5200050-15], [399930+15, 5089950+15], [602790-15, 5089950-15], [602790-15, 5200050+15],  [399930+15, 5200050-15]]]}, "EPSG:32636", false);
Map.addLayer(geometry, {color: "lightblue"}, "Base extent")

var START = ee.Date("2021-01-31")
var END = START.advance(365, "day")

//Defining visual palette
var VIS_PALETTE = [
    'blue', 'darkgreen', 'green', 'red', 'yellow', 'lightgreen', 'gray',
    'brown', 'white'];
    

// Image classification of 2021
var dw21 = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
    .filterDate("2021-11-01", "2021-12-31")
    .filterBounds(geometry)
    .select("label")
    .median()
Map.addLayer(dw21, {min: 0, max: 8, palette: VIS_PALETTE}, "Classification 2021")


// Image classification of 2022
var dw22 = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
    .filterDate("2022-01-01", "2022-12-31")
    .filterBounds(geometry)
    .select("label")
    .median()
Map.addLayer(dw22, {min: 0, max: 8, palette: VIS_PALETTE}, "Classification 2022")


// Image classification of 2023
var dw23 = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
    .filterDate("2023-01-01", "2023-12-31")
    .filterBounds(geometry)
    .select("label")
    .median()
Map.addLayer(dw23, {min: 0, max: 8, palette: VIS_PALETTE}, "Classification 2023")

//Set image Center
Map.centerObject(geometry, 6);

//Define Mask for 2021 ---------------------------------------------------------

var cropmask = dw21
    .select("label").eq(4)

// Define the new region of interest (ROI)
var ROI = geometry;
var clippedCropMask21 = cropmask.clip(ROI);

// Add the clipped crop mask to the map
Map.addLayer(clippedCropMask21, {min: 0, max: 1, palette: ["black", "white"]}, "Crop Mask 2021");



//Define Mask for 2022

var cropmask = dw22
    .select("label").eq(4)

// Define the new region of interest (ROI)
var ROI = geometry;
var clippedCropMask22 = cropmask.clip(ROI);
// Add the clipped crop mask to the map
Map.addLayer(clippedCropMask22, {min: 0, max: 1, palette: ["black", "white"]}, "Crop Mask 2022");


//Define Mask for 2023

var cropmask = dw23
    .select("label").eq(4)

// Define the new region of interest (ROI)
var ROI = geometry;
var clippedCropMask23 = cropmask.clip(ROI);

// Add the clipped crop mask to the map
Map.addLayer(clippedCropMask23, {min: 0, max: 1, palette: ["black", "white"]}, "Crop Mask 2023");