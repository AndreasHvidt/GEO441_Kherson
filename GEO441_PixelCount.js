// Pixel Counts ----------------------------------------------------------------

// Defining geometry
var geometry = ee.Geometry({type: 'Polygon', coordinates: [[[399930+15, 5200050-15], [399930+15, 5089950+15], [602790-15, 5089950-15], [602790-15, 5200050+15],  [399930+15, 5200050-15]]]}, "EPSG:32636", false);
Map.addLayer(geometry, {color: "lightblue"}, "Base extent")

// Define the time range
var startDate = '2023-01-01';
var endDate = '2023-12-31';

// Load the image collection
var collection = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')  // Update 'SOME_DATASET' with the appropriate dataset
  .filterBounds(geometry)
  .filterDate(startDate, endDate);

// Function to mask out pixels not belonging to class "label04"
var maskLabel04 = function(image) {
  var label04_mask = image.select('label').eq(4);  // Assuming 'class_band' contains class labels
  return image.updateMask(label04_mask);
};

// Apply the mask to each image in the collection
var maskedCollection = collection.map(maskLabel04);

// Function to calculate pixel count
var countPixels = function(image) {
  var count = image.reduceRegion({
    reducer: ee.Reducer.count(),
    geometry: geometry,
    scale: 30, // Update with appropriate scale
    maxPixels: 1e9
  });
  return image.set('pixel_count', count.get('label'));
};

// Map over the masked collection to calculate pixel count for each image
var countedCollection = maskedCollection.map(countPixels);

// Get the pixel count for each image as a list
var pixelCounts = countedCollection.aggregate_array('pixel_count');

// Get the corresponding dates as a list
var dates = countedCollection.aggregate_array('system:time_start');

// Plot the pixel count over time
print(ui.Chart.array.values(pixelCounts, 0, dates)
  .setChartType('LineChart')
  .setOptions({
    title: 'Pixel Count of Crops 2023',
    hAxis: {title: 'Date'},
    vAxis: {title: 'Pixel Count'},
    lineWidth: 2,  // Increase line width for better visibility
    pointSize: 0,  // Remove points
  }));


// Define the time range
var startDate = '2022-01-01';
var endDate = '2022-12-31';

// Load the image collection
var collection = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')  // Update 'SOME_DATASET' with the appropriate dataset
  .filterBounds(geometry)
  .filterDate(startDate, endDate);

// Function to mask out pixels not belonging to class "label04"
var maskLabel04 = function(image) {
  var label04_mask = image.select('label').eq(4);  // Assuming 'class_band' contains class labels
  return image.updateMask(label04_mask);
};

// Apply the mask to each image in the collection
var maskedCollection = collection.map(maskLabel04);

// Function to calculate pixel count
var countPixels = function(image) {
  var count = image.reduceRegion({
    reducer: ee.Reducer.count(),
    geometry: geometry,
    scale: 30, // Update with appropriate scale
    maxPixels: 1e9
  });
  return image.set('pixel_count', count.get('label'));
};

// Map over the masked collection to calculate pixel count for each image
var countedCollection = maskedCollection.map(countPixels);

// Get the pixel count for each image as a list
var pixelCounts = countedCollection.aggregate_array('pixel_count');

// Get the corresponding dates as a list
var dates = countedCollection.aggregate_array('system:time_start');

// Plot the pixel count over time
print(ui.Chart.array.values(pixelCounts, 0, dates)
  .setChartType('LineChart')
  .setOptions({
    title: 'Pixel Count of Crops 2022',
    hAxis: {title: 'Date'},
    vAxis: {title: 'Pixel Count'},
    lineWidth: 2,  // Increase line width for better visibility
    pointSize: 0,  // Remove points
  }));
  
  
  // Define the time range
var startDate = '2021-01-01';
var endDate = '2021-12-31';

// Load the image collection
var collection = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')  // Update 'SOME_DATASET' with the appropriate dataset
  .filterBounds(geometry)
  .filterDate(startDate, endDate);

// Function to mask out pixels not belonging to class "label04"
var maskLabel04 = function(image) {
  var label04_mask = image.select('label').eq(4);  // Assuming 'class_band' contains class labels
  return image.updateMask(label04_mask);
};

// Apply the mask to each image in the collection
var maskedCollection = collection.map(maskLabel04);

// Function to calculate pixel count
var countPixels = function(image) {
  var count = image.reduceRegion({
    reducer: ee.Reducer.count(),
    geometry: geometry,
    scale: 30, // Update with appropriate scale
    maxPixels: 1e9
  });
  return image.set('pixel_count', count.get('label'));
};

// Map over the masked collection to calculate pixel count for each image
var countedCollection = maskedCollection.map(countPixels);

// Get the pixel count for each image as a list
var pixelCounts = countedCollection.aggregate_array('pixel_count');

// Get the corresponding dates as a list
var dates = countedCollection.aggregate_array('system:time_start');

// Plot the pixel count over time
print(ui.Chart.array.values(pixelCounts, 0, dates)
  .setChartType('LineChart')
  .setOptions({
    title: 'Pixel Count of Crops 2021',
    hAxis: {title: 'Date'},
    vAxis: {title: 'Pixel Count'},
    lineWidth: 2,  // Increase line width for better visibility
    pointSize: 0,  // Remove points
  }));



  // Pixel Counts per masks -------------------------------------------------------

// Load the Dynamic World classification collection and create a median composite for 2021
var dw21 = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
.filterDate('2021-11-01', '2021-12-31')
.filterBounds(geometry)
.select('label')
.median();

// Define the mask for pixels labeled as "label04" (value 4)
var label04Mask = dw21.eq(4);

// Count the number of pixels labeled as "label04"
var label04Count = label04Mask.reduceRegion({
reducer: ee.Reducer.sum(),
geometry: geometry,
scale: 10,  // Adjust the scale as needed
maxPixels: 1e9
});

// Print the count of pixels labeled as "label04" and convert to square kilometers
label04Count.getInfo(function(result) {
var pixelCount = result.label;
pixelCount= pixelCount.toFixed(3);
var areaSqKm = pixelCount * 10 * 10 / 1e6; // 10m x 10m pixels converted to square kilometers
areaSqKm = areaSqKm.toFixed(3);
print('Total pixels labeled as "label04" in 2021:', pixelCount);
print('Total area labeled as "label04" in 2021(sq km):', areaSqKm);
});



// Load the Dynamic World classification collection and create a median composite for 2021
var dw21 = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
.filterDate('2022-11-01', '2022-12-31')
.filterBounds(geometry)
.select('label')
.median();

// Define the mask for pixels labeled as "label04" (value 4)
var label04Mask = dw21.eq(4);

// Count the number of pixels labeled as "label04"
var label04Count = label04Mask.reduceRegion({
reducer: ee.Reducer.sum(),
geometry: geometry,
scale: 10,  // Adjust the scale as needed
maxPixels: 1e9
});

// Print the count of pixels labeled as "label04" and convert to square kilometers
label04Count.getInfo(function(result) {
var pixelCount = result.label;
pixelCount= pixelCount.toFixed(3);
var areaSqKm = pixelCount * 10 * 10 / 1e6; // 10m x 10m pixels converted to square kilometers
areaSqKm = areaSqKm.toFixed(3);
print('Total pixels labeled as "label04"in 2022:', pixelCount);
print('Total area labeled as "label04" in 2022 (sq km):', areaSqKm);
});



// Load the Dynamic World classification collection and create a median composite for 2021
var dw21 = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
.filterDate('2023-11-01', '2023-12-31')
.filterBounds(geometry)
.select('label')
.median();

// Define the mask for pixels labeled as "label04" (value 4)
var label04Mask = dw21.eq(4);

// Count the number of pixels labeled as "label04"
var label04Count = label04Mask.reduceRegion({
reducer: ee.Reducer.sum(),
geometry: geometry,
scale: 10,  // Adjust the scale as needed
maxPixels: 1e9
});

// Print the count of pixels labeled as "label04" and convert to square kilometers
label04Count.getInfo(function(result) {
var pixelCount = result.label;
pixelCount= pixelCount.toFixed(3);
var areaSqKm = pixelCount * 10 * 10 / 1e6; // 10m x 10m pixels converted to square kilometers
areaSqKm = areaSqKm.toFixed(3);
print('Total pixels labeled as "label04" in 2023:', pixelCount);
print('Total area labeled as "label04" in 2023 (sq km):', areaSqKm);
});