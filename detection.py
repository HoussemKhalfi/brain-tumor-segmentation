import numpy as np
from skimage.filters import threshold_multiotsu
#from skimage.segmentation import expand_labels
from skimage import morphology

def detect_tumor_thresholding(img, threshold_value):
    """
    Detect tumor using multi-Otsu thresholding technique and create a binary mask.
    
    Parameters:
    img (numpy array): The 3D image (e.g., from MRI modalities) to process.
    threshold_value (float): The threshold value for generating the binary mask.
    
    Returns:
    numpy array: The binary mask where tumor regions are set to 1 and background to 0.
    """
    # Apply multi-Otsu thresholding to generate the thresholds
    thresholds = threshold_multiotsu(img)
    
    # Use the thresholds to segment the image into regions
    regions = np.digitize(img, bins=thresholds)
    
    # Generate a segmented image
    segmented_img = regions * img
    
    # Create a binary mask from the segmented image based on the threshold value
    binary_mask =  (segmented_img > threshold_value).astype(np.float64) #np.where(segmented_img > threshold_value, 1, 0)
    
    return binary_mask


def postprocess_tumor_mask(binary_mask, small_objects_min_size1=80, small_objects_min_size2=100, dilation_ball_radius=2, expand_distance=1.8):
    """
    Post-process the binary mask to refine the tumor segmentation using morphological operations.
    
    Parameters:
    binary_mask (numpy array): The binary mask generated from the detect_tumor_thresholding function.
    small_objects_min_size1 (int, optional): Minimum size for removing small objects in the first stage. Default is 80.
    small_objects_min_size2 (int, optional): Minimum size for removing small objects in subsequent stages. Default is 100.
    dilation_ball_radius (int, optional): Radius of the spherical structuring element for dilation. Default is 2.
    expand_distance (float, optional): Distance for expanding the labels. Default is 1.8.
    
    Returns:
    numpy array: The final processed binary mask after morphological operations.
    """
    # Step 1: Binary closing to fill small holes in the binary mask
    img_closed = morphology.binary_closing(binary_mask)
    
    # Step 2: Remove small objects with a minimum size threshold
    img_no_small_object1 = morphology.remove_small_objects(img_closed, min_size=small_objects_min_size1)
    
    # Step 3: Apply binary erosion to refine the mask
    eroded_img = morphology.binary_erosion(img_no_small_object1)
    
    # Step 4: Remove small objects again after erosion
    img_no_small_object2 = morphology.remove_small_objects(eroded_img, min_size=small_objects_min_size2)
    
    # Step 5: Apply binary erosion again to further refine the mask
    eroded_img1 = morphology.binary_erosion(img_no_small_object2)
    
    # Step 6: Remove small objects again after the second erosion
    img_no_small_object3 = morphology.remove_small_objects(eroded_img1, min_size=small_objects_min_size2)
    
    # Step 7: Create a spherical structuring element for binary dilation
    footprint = morphology.ball(dilation_ball_radius)
    
    # Step 8: Apply binary dilation to expand the regions
    dilated_img = morphology.binary_dilation(img_no_small_object3, footprint)
    
    # Step 9: Apply binary closing again to smooth the final mask
    tumor_mask_res = morphology.binary_closing(dilated_img)
    
    # Optional : Step 10: Expand the labels for the final refinement
    #expanded_mask = expand_labels(tumor_mask_res, distance=expand_distance)
    
    return tumor_mask_res
