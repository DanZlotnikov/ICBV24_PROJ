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


1. **Clone the Repository**<br>
Clone the project repository to your local machine using:
```
git clone https://github.com/DanZlotnikov/ICBV24_PROJ.git
```
2. **Set up a Python Virtual Environment**<br>
After cloning, navigate into the project directory:
```
cd project
```
Create a Python virtual environment using Python 3.11 as it is required for this project:
```Copy code
python3.11 -m venv .venv
```
**important! python 3.11 is required!**

Activate the virtual environment:
On Windows:
```Copy code
.venv\Scripts\activate
```
On macOS and Linux:

```Copy code
source .venv/bin/activate
```
Note: The virtual environment directory (.venv) is not tracked in the repository as it is included in .gitignore.

3. **Install Required Packages**
With the virtual environment activated, install all required dependencies:
```Copy code
pip install -r requirements.txt
```
4. **run the GUI**
run the python file src/GUI/app.py
```angular2html
python app.py
```
5. **load images to the app**<br>
use the 'choose image' to choose the image you would like to process, then upload the image using the 'upload image' button.
6. **choose enhance image** after choosing the enhancement technique. make sure to fill scale factor and bins with integer values.
7. choose image completion to complete image missing parts. in order to remove a part from the image, draw a square one the picture you uploded.
<br> **notice**, the shape has to be drawn from top left to button right.
<br>**notice** to reload the same picture, refresh the page.
8. **choose image completion technique** and press the complete image button.
<br><br>**watch this short demo how to complete missing patches**(notice how I draw the rectangle from top left to buttom right!)
![demo](vidGif.gif)
