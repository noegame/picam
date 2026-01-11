#include "opencv_wrapper.h"
#include <stdio.h>
#include <assert.h>

int main() {
    printf("Testing OpenCV wrapper...\n");
    
    // Test 1: Invalid image path
    printf("Test 1: Loading invalid image... ");
    ImageHandle* invalid = load_image("nonexistent.jpg");
    assert(invalid == NULL);
    printf("PASSED\n");
    
    // Add more tests:
    // - Load valid image
    // - Convert to grayscale
    // - Check dimensions
    // - Save image
    // - Memory cleanup
    
    printf("All tests passed!\n");
    return 0;
}
