# Brain Tumor Segmentation using MRI Data

This repository contains code and notebooks to explore, detect, and segment brain tumors using MRI data. The segmentation is focused on distinguishing key tumor subregions: edema, enhancing tumor, and necrosis. The main techniques employed are thresholding and K-Means clustering.

## Repository Structure

- **notebooks/**
  - **1_Data_Exploration.ipynb**: Initial exploration of the MRI data, visualizing different acquisition planes and modalities to better understand the dataset.
  - **2_Brain_Tumor_Detection.ipynb**: Detects brain tumors by fusing MRI modalities and applying thresholding to isolate the tumor region.
  - **3_Brain_Tumor_Segmentation.ipynb**: Segments the detected tumor into edema, enhancing tumor, and necrosis using K-Means clustering. Evaluation metrics such as the Dice coefficient, confusion matrix, precision, and recall are used to assess the performance.

- **requirements.txt**: A list of Python dependencies required to run the code.

- **detection.py**: Contains functions for detecting the tumor using thresholding techniques and post-processing the resulting mask.

- **visualize.py**: Provides utility functions to visualize MRI volumes, segmentation masks, and overlays between predicted and ground truth segmentations.

- **metrics.py**: Implements functions to compute evaluation metrics such as the Dice coefficient and confusion matrix.

## Dataset

The **MSD Task_01 BrainTumor dataset** consists of 750 multiparametric magnetic resonance images (mp-MRI) from patients diagnosed with either glioblastoma or lower-grade glioma. The sequences used were native T1-weighted (T1), post-Gadolinium (Gd) contrast T1-weighted (T1-Gd), native T2-weighted (T2), and T2 Fluid-Attenuated Inversion Recovery (FLAIR). The corresponding target ROIs were the three tumor sub-regions, namely edema, enhancing, and non-enhancing tumor.

This dataset was selected due to the challenge of locating these complex and heterogeneously-located targets. The data was acquired from 19 different institutions and contained a subset of the data used in the 2016 and 2017 Brain Tumor Segmentation (BraTS) challenges.

To download the dataset, you can access the Medical Segmentation Decathlon (MSD) [website](http://medicaldecathlon.com/index.html), where they offer two options for downloading: either directly from AWS or by accessing the data hosted on Google Drive. 

In this project, we'll proceed with the Google Drive option since we are working in the Google Colab environment.

## Key Features

1. **Data Exploration**: Visualize MRI volumes across different modalities (FLAIR, T1ce, T2) to gain insights into the tumor characteristics.
2. **Brain Tumor Detection**: Detects tumor regions by applying thresholding on fused MRI modalities.
3. **Brain Tumor Segmentation**: Segments tumors into key regions using K-Means clustering, followed by a quantitative evaluation using performance metrics.
4. **Visualization**: Interactive functions to explore and visualize both the 2D MRI slices and the 3D volumes with the predicted segmentation.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your_username/brain_tumor_segmentation.git
   cd brain_tumor_segmentation
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebooks or scripts in your preferred environment.

## Usage

- Start by exploring the data in `1_Data_Exploration.ipynb`.
- Proceed to detect the tumor in `2_Brain_Tumor_Detection.ipynb`.
- Finally, segment the tumor subregions using the clustering method in `3_Brain_Tumor_Segmentation.ipynb`.

## Evaluation Metrics

- **Dice Coefficient**: Measures the overlap between the predicted and ground truth segmentation.
- **Confusion Matrix**: Provides a detailed breakdown of class-wise predictions.
- **Precision and Recall**: Evaluates the accuracy of segmentation for each tumor subregion.

## Contributing

Feel free to open issues or pull requests if you find any bugs or want to suggest improvements.