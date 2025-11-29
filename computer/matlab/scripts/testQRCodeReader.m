% Read an image containing a QR code
img = imread('C:\HOME\WORK\raspberrypi\ZOD\picam\matlab\data1\1761484171196.jpg');

% Create a QR code reader object
qrReader = vision.QRCodeReader;

% Decode the QR code
[decodedData, locs] = qrReader(img);

% Display the results
disp('Decoded Data:');
disp(decodedData);
disp('Locations:');
disp(locs);
