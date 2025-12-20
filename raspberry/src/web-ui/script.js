function formatTime(date) {
    return date.toLocaleTimeString();
}

function drawPlayground(arucoTags = []) {
    const canvas = document.getElementById('playground-canvas');
    if (!canvas) {
        console.error('Canvas element not found');
        return;
    }
    
    const ctx = canvas.getContext('2d');
    const img = new Image();
    
    img.onload = function() {
        // Draw the image on canvas
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        console.log('Playground image drawn successfully');
        
        // Draw ArUco tags on the canvas
        drawArucoTagsOnCanvas(ctx, arucoTags, canvas.width, canvas.height);
    };
    
    img.onerror = function() {
        console.error('Failed to load playground image');
        // Fill with white as fallback
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Draw ArUco tags on the canvas
        drawArucoTagsOnCanvas(ctx, arucoTags, canvas.width, canvas.height);
    };
    
    img.src = '/playground.png';
}

function drawArucoTagsOnCanvas(ctx, tags, canvasWidth, canvasHeight) {
    const squareSize = 100; // Size of the colored square for each tag
    
    tags.forEach(tag => {
        // Skip beacon tags (20, 21, 22, 23)
        if ([20, 21, 22, 23].includes(tag.aruco_id)) {
            return;
        }
        
        const x = tag.x;
        const y = tag.y;
        
        // Draw colored square at tag position
        const bgColor = getTagColor(tag.aruco_id);
        ctx.fillStyle = bgColor;
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 2;
        
        // Draw square centered at the tag position
        ctx.fillRect(x - squareSize / 2, y - squareSize / 2, squareSize, squareSize);
        ctx.strokeRect(x - squareSize / 2, y - squareSize / 2, squareSize, squareSize);
        
        // Draw tag ID text in the center
        const textColor = getTagTextColor(tag.aruco_id);
        ctx.fillStyle = textColor;
        ctx.font = 'bold 18px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(tag.aruco_id, x, y);
        
        // Draw angle indicator if angle is available
        if (tag.angle !== null && tag.angle !== undefined) {
            ctx.strokeStyle = textColor;
            ctx.lineWidth = 2;
            ctx.beginPath();
            const angleRad = (tag.angle * Math.PI) / 180;
            const lineLength = squareSize / 2;
            const endX = x + lineLength * Math.cos(angleRad);
            const endY = y + lineLength * Math.sin(angleRad);
            ctx.moveTo(x, y);
            ctx.lineTo(endX, endY);
            ctx.stroke();
        }
    });
}

function getTagColor(arucoId) {
    const colorMap = {
        '36': '#4169E1',      // blue
        '47': '#FFD700',      // yellow
        '41': '#000000',      // black
        '20': '#ddd',      // grey (beacon)
        '21': '#ddd',      // grey (beacon)
        '22': '#ddd',      // grey (beacon)
        '23': '#ddd'       // grey (beacon)
    };
    return colorMap[arucoId.toString()] || '#f0f0f0';
}

function getTagTextColor(arucoId) {
    const textColorMap = {
        '36': '#ffffff',      // white text for blue
        '47': '#000000',      // black text for yellow
        '41': '#ffffff',      // white text for black
        '20': '#000000',      // black text for grey
        '21': '#000000',      // black text for grey
        '22': '#000000',      // black text for grey
        '23': '#000000'       // black text for grey
    };
    return textColorMap[arucoId.toString()] || '#000000';
}

function fetchArucoData() {
    fetch('/aruco_data')
        .then(response => response.json())
        .then(data => {
            const tagsList = document.getElementById('tags-list');
            const tagCount = document.getElementById('tag-count');
            const lastUpdate = document.getElementById('last-update');

            // Redraw playground with current ArUco tags
            drawPlayground(data);

            if (data && data.length > 0) {
                tagsList.innerHTML = '';
                data.forEach(tag => {
                    const tagItem = document.createElement('div');
                    tagItem.className = 'tag-item';
                    const bgColor = getTagColor(tag.aruco_id);
                    const textColor = getTagTextColor(tag.aruco_id);
                    tagItem.style.backgroundColor = bgColor;
                    tagItem.style.color = textColor;
                    
                    // Create a wrapper for image and text
                    tagItem.innerHTML = `
                        <div class="tag-content">
                            <img class="tag-image" src="aruco_tags/4x4_1000-${tag.aruco_id}.svg" alt="Tag ${tag.aruco_id}">
                            <div class="tag-info">
                                <strong>ID: ${tag.aruco_id}</strong>
                                <br>X: ${Math.round(tag.x)} 
                                <br>Y: ${Math.round(tag.y)}
                                ${tag.angle !== null ? `<br>Angle: ${tag.angle.toFixed(2)}°` : ''}
                            </div>
                        </div>
                    `;
                    tagsList.appendChild(tagItem);
                });
                tagCount.textContent = data.length;
            } else {
                tagsList.innerHTML = '<div class="no-tags">No tags detected</div>';
                tagCount.textContent = '0';
            }
            lastUpdate.textContent = 'Last update: ' + formatTime(new Date());
        })
        .catch(error => {
            console.error('Error fetching ArUco data:', error);
            document.getElementById('tags-list').innerHTML = '<div class="no-tags">Error loading data</div>';
        });
}

// Initialize on page load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() {
        console.log('DOM Content Loaded');
        fetchArucoData();
    });
} else {
    console.log('DOM already loaded');
    fetchArucoData();
}

// Fetch data every 100ms (10 times per second)
setInterval(fetchArucoData, 100);
