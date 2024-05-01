# Project Title: Image Restoration using Fourier Transform and Relaxation Labeling

## Introduction
This project aims to leverage advanced computational techniques, namely Fourier Transform and Relaxation Labeling, to restore and enhance historical photographs. By addressing issues like damaged sections and low resolution, our tools breathe new life into cherished images, allowing for a clearer glimpse into the past.

## Project Overview
Our solution innovatively combines Fourier series and relaxation labeling to not only repair damaged parts of photographs but also enhance image resolution. This dual approach ensures that even heavily impaired images can be restored to a high degree of clarity and detail.

### Key Features:
- **Fourier Transform for Patch Prediction**: Identifies and reconstructs damaged sections of an image by predicting missing data through spectral analysis.
- **Relaxation Labeling for Image Upscaling**: Increases image resolution by intelligently estimating pixel values, enhancing both the detail and size of the photograph.

## Tools and Methods
### Fourier Transform - Patch Prediction
1. **Segmentation**: Divide the image into corrupted and uncorrupted areas.
2. **Spectral Analysis**: Calculate the magnitude spectrum for uncorrupted segments.
3. **Interpolation**: Predict the magnitude spectrum for the corrupted areas by interpolating the spectra of the intact parts.
4. **Reconstruction**: Apply the Inverse Fourier Transform to reconstruct the damaged sections.

### Relaxation Labeling - Image Upscaling
1. **Pixel Multiplication**: Increase the number of pixels in the image, adding new pixels with initial confidence values based on their proximity to established pixels.
2. **Optimization**: Utilize relaxation labeling to accurately determine the final values of the new pixels, ensuring a natural integration into the image.

## Getting Started
To begin using this project, clone the repository and follow the setup instructions below:
