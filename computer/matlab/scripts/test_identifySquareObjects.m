% Load one image from folderPath into Workspace
[filename, pathname] = uigetfile({'*.png;*.jpg;*.jpeg', 'Image Files (*.png, *.jpg, *.jpeg)'}, 'Select an image');
fileName = fullfile(pathname, filename);

% Read an image
inputImage = imread(fileName);

% Identify square objects
squareObjects = identifySquareObjects(inputImage);