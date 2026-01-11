#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv) {
    // Check if image path is provided
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <image_path> [output_path]" << std::endl;
        return -1;
    }

    // Read the input image
    std::string input_path = argv[1];
    cv::Mat image = cv::imread(input_path, cv::IMREAD_COLOR);

    // Check if image was loaded successfully
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image: " << input_path << std::endl;
        return -1;
    }

    std::cout << "Image loaded successfully: " << image.cols << "x" << image.rows << " pixels" << std::endl;

    // Convert to grayscale
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

    std::cout << "Image converted to grayscale" << std::endl;

    // Determine output path
    std::string output_path;
    if (argc >= 3) {
        output_path = argv[2];
    } else {
        // Default: add "_gray" before the extension
        size_t dot_pos = input_path.find_last_of('.');
        if (dot_pos != std::string::npos) {
            output_path = input_path.substr(0, dot_pos) + "_gray" + input_path.substr(dot_pos);
        } else {
            output_path = input_path + "_gray.jpg";
        }
    }

    // Save the grayscale image
    if (cv::imwrite(output_path, gray_image)) {
        std::cout << "Grayscale image saved to: " << output_path << std::endl;
    } else {
        std::cerr << "Error: Could not save the image to: " << output_path << std::endl;
        return -1;
    }

    // Optional: Display the images
    // cv::imshow("Original Image", image);
    // cv::imshow("Grayscale Image", gray_image);
    // 
    // std::cout << "Press any key to close the windows..." << std::endl;
    // cv::waitKey(0);
    // cv::destroyAllWindows();

    return 0;
}
