function filteredBW = removeNonSquareObjects(BW)
    % removeNonSquareObjects removes non-rectangular objects from a binary image.
    % keep the distorted squares
    %
    % Input:
    %   BW - Binary image (logical or numeric array)
    %
    % Output:
    %   BW_cleaned - Binary image with non-rectangular objects removed

    
    % Label connected components in the binary image
    [L, num] = bwlabel(BW);
    
    % Initialize a binary image for cleaned output
    filteredBW = false(size(BW));
    
    % Loop through each labeled object
    for k = 1:num
        % Create a mask for the current object
        currentObject = (L == k);
        
        % Get the properties of the current object
        stats = regionprops(currentObject, 'BoundingBox', 'Area');
        
        % Calculate the aspect ratio
        width = stats.BoundingBox(3);
        height = stats.BoundingBox(4);
        aspectRatio = width / height;
        
        % Check if the object is approximately square (aspect ratio close to 1)
        if aspectRatio >= 0.4 && aspectRatio <= 1.6
            % Keep the object in the cleaned image
            filteredBW = filteredBW | currentObject;
        end
    end
end
