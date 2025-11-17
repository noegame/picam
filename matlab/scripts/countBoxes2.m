function count = countBoxes(imgBW)
    % countRotatedRectangles rectangular objects from a binary image.
    % 
    %
    % Input:
    %   imgBW - Binary image (logical or numeric array)
    %
    % Output:
    %   count - number of counted rectangles

    minArea = 400;
    ratioThresh = 0.50; % FilledArea / (w*h) attendu proche de 1 pour rectangle plein

    imgClean = bwareaopen(imgBW, minArea);
    cc = bwconncomp(imgClean);
    props = regionprops(cc, 'FilledArea', 'BoundingBox', 'Orientation', 'Image', 'Centroid');

    isRect = false(numel(props),1);
    for k = 1:numel(props)
        % extraire l'image binaire de la région (tight crop)
        regionImg = props(k).Image; % logical small image of region
        orient = props(k).Orientation;

        % Recenter & rotate the region so rectangle becomes axis-aligned
        % pad to allow rotation without cropping
        padsize = round(max(size(regionImg))*1.2);
        padded = false(padsize);
        % place region centré
        offr = floor((size(padded,1)-size(regionImg,1))/2)+1;
        offc = floor((size(padded,2)-size(regionImg,2))/2)+1;
        padded(offr:offr+size(regionImg,1)-1, offc:offc+size(regionImg,2)-1) = regionImg;

        % rotate by -orientation to align major axis horizontally
        rot = imrotate(padded, -orient, 'bilinear', 'crop');
        rotBin = rot > 0.5;
        % get bounding box of rotated component
        stats2 = regionprops(rotBin, 'BoundingBox', 'FilledArea');
        if isempty(stats2)
            continue;
        end
        bb = stats2(1).BoundingBox;
        areaRect = bb(3) * bb(4); % w*h
        filled = stats2(1).FilledArea;

        if areaRect > 0 && (filled / areaRect) >= ratioThresh && filled >= minArea
            isRect(k) = true;
        end
    end
    count = sum(isRect);

    % Display the original image
    figure;
    imshow(imgBW);
    hold on;

    % Display points at the centroids of the rectangles
    for k = 1:numel(props)
        if isRect(k)
            centroid = props(k).Centroid; % Get the centroid
            plot(centroid(1), centroid(2), 'go', 'MarkerSize', 3, 'MarkerFaceColor', 'g'); % Draw green point
        end
    end

    title(['Count of Rectangles: ', num2str(sum(isRect))]);
    hold off;

end
