function squareObjects = identifySquareObjects(image)
    % identifySquareObjects identifies square-like objects in a given image.
    %
    % Input:
    %   image - Input image (can be grayscale or RGB)
    %
    % Output:
    %   squareObjects - Binary image with detected square objects

    % Convert to grayscale if the image is RGB
    if size(image, 3) == 3
        grayImage = rgb2gray(image);
    else
        grayImage = image;
    end

    % Binarize the image
    binaryImage = imbinarize(grayImage);

    % Use edge detection
    edges = edge(binaryImage, 'Canny');

    % Find contours
    [B, ~] = bwboundaries(edges, 'noholes');

    % Initialize output image
    squareObjects = false(size(binaryImage));

    % Loop through each boundary
    for k = 1:length(B)
        boundary = B{k};
        % Approximate the contour to a polygon
        poly = approxPolyDP(boundary, 0.02 * arcLength(boundary, true), true);

        % Check if the polygon has 4 corners (indicating a square)
        if length(poly) == 4
            % Calculate aspect ratio
            rect = boundingRect(poly);
            aspectRatio = rect(3) / rect(4); % width / height

            % Check if aspect ratio is close to 1 (indicating a square)
            if aspectRatio > 0.8 && aspectRatio < 1.2
                % Fill the detected square in the output image
                squareObjects = squareObjects | poly2mask(boundary(:,2), boundary(:,1), size(binaryImage, 1), size(binaryImage, 2));
            end
        end
    end

    % Display the results
    imshowpair(binaryImage, squareObjects, 'montage');
    title('Original Binary Image and Detected Square Objects');
end

function poly = approxPolyDP(contour, epsilon, closed)
    % Approximate a contour to a polygon
    poly = cv.approxPolyDP(contour, epsilon, closed);
end

function rect = boundingRect(poly)
    % Get the bounding rectangle of a polygon
    rect = cv.boundingRect(poly);
end

function length = arcLength(contour, closed)
    % Calculate the arc length of a contour
    length = cv.arcLength(contour, closed);
end
