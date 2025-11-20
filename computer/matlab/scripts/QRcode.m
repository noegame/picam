% Load one image from folderPath into Workspace
[filename, pathname] = uigetfile({'*.png;*.jpg;*.jpeg', 'Image Files (*.png, *.jpg, *.jpeg)'}, 'Select an image');
fileName = fullfile(pathname, filename);

% Read the image containing QR codes
img = imread(fileName);

% Detect and decode QR codes
[decodedData, locs] = vision.readQRCode(img);

% Display the results
for i = 1:length(decodedData)
    fprintf('QR Code %d: %s\n', i, decodedData{i});
    % Optionally, you can mark the location of the QR codes in the image
    rectangle('Position', [locs(i,1), locs(i,2), locs(i,3), locs(i,4)], ...
              'EdgeColor', 'r', 'LineWidth', 2);
end

% Show the image with detected QR codes
imshow(img);
