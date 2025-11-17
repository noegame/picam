% Count blue and yellow squares on a picture.
% Run withparameters specific to the image.
% 1. removes unwanted colors from the image
% 2. make the image black and white 
% 3. remove area that are to small to be square
% 4. count square

% === SETTINGS ===

% T = [minRed maxRed minGreen maxGreen minBlue Maxblue]
thresholdsB = [000.000 100.000 000.000 175.000 150.000 255.000];
thresholdsY = [100.000 255.000 140.000 255.000 000.000 050.000];

% Minimum size of objects to keep, objects with fewer pixels will be removed.
minObjectSize = 250;

% === IMAGE PROCESSING ===

% Load one image from folderPath into Workspace
[filename, pathname] = uigetfile({'*.png;*.jpg;*.jpeg', 'Image Files (*.png, *.jpg, *.jpeg)'}, 'Select an image');
fileName = fullfile(pathname, filename);

img = imread(fileName);

% Apply mask to the image
maskedImgB = createMaskedImage(img, thresholdsB);
maskedImgY = createMaskedImage(img, thresholdsY);

% Convert RGB masked image to gray image
grayImgB = rgb2gray(maskedImgB);
grayImgY = rgb2gray(maskedImgY);

% Convert the grey masked images to binary image
binaryImgB = imbinarize(grayImgB);
binaryImgY = imbinarize(grayImgY);

% Remove small objects
binaryImgCleanB = bwareaopen(binaryImgB, minObjectSize);
binaryImgCleanY = bwareaopen(binaryImgY, minObjectSize);

% Remove non-square objects based on their properties 
binaryImgCleanOnlyRectB = removeNonSquareObjects(binaryImgCleanB); % NOT WORKING PROPERLY
binaryImgCleanOnlyRectY = removeNonSquareObjects(binaryImgCleanY); % NOT WORKING PROPERLY

% Count rectangle on the binary image with function countRotatedRectangles
countB = countRotatedRectangles(binaryImgCleanB);
countY = countRotatedRectangles(binaryImgCleanY);

% Count rectangles in the cleaned binary images
countB2 = countRotatedRectangles(binaryImgCleanOnlyRectB);
countY2 = countRotatedRectangles(binaryImgCleanOnlyRectY);

% Count the box. A box is 2 adjacent squares
countBoxB = countBoxes(binaryImgCleanB);
countBoxY = countBoxes(binaryImgCleanY);

% === SHOW RESULT ===

disp("")
disp("rectangle bleue compté méthode 1 : " + string(countB))
disp("rectangle jaune compté méthode 1 : " + string(countY))
disp("rectangle bleue compté méthode 2 : " + string(countB2))
disp("rectangle jaune compté méthode 2 : " + string(countY2))

% figure;
% subplot(4, 2, 1); imshow(img); title('original image')
% subplot(4, 2, 3); imshow(maskedImgB); title('blue filtered image');
% subplot(4, 2, 4); imshow(maskedImgY); title('yellow filtered image');
% subplot(4, 2, 5); imshow(binaryImgB); title('blue binary image');
% subplot(4, 2, 6); imshow(binaryImgY); title('yellow binary image');
% subplot(4, 2, 7); imshow(binaryImgCleanB); title('blue cleaned from small object binary image');
% subplot(4, 2, 8); imshow(binaryImgCleanY); title('yellow cleaned from small object binary image');

imgList = {binaryImgCleanB,binaryImgCleanOnlyRectB};
montage(imgList)

% === SAVE RESULT ===

outputFolder = fullfile('output');
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

imwrite(img, fullfile(outputFolder, 'originalImg.png'));
imwrite(binaryImgB, fullfile(outputFolder, 'binaryImgB.png'));
imwrite(binaryImgY, fullfile(outputFolder, 'binaryImgY.png'));
imwrite(binaryImgCleanB, fullfile(outputFolder, 'binaryImgCleanB.png'));
imwrite(binaryImgCleanY, fullfile(outputFolder, 'binaryImgCleanY.png'));
imwrite(binaryImgCleanOnlyRectB, fullfile(outputFolder, 'binaryImgCleanOnlyRectB.png'));
imwrite(binaryImgCleanOnlyRectY, fullfile(outputFolder, 'binaryImgCleanYOnlyRectY.png'));



