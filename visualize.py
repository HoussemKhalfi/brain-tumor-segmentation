import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate
import matplotlib.patches as mpatches
from skimage import color

# Define a function to visualize 3D MRI data in different planes
def explore_4dimage(layer, MRI_Seq, input_data, plane='axial'):
    """
    Visualizes a single layer of the 3D MRI volume in the specified plane (axial, sagittal, or coronal).

    Args:
        layer (int): The index of the slice to visualize.
        MRI_Seq (int): The MRI modality to visualize (e.g., FLAIR, T1ce, T2).
        input_data (numpy array): 4D MRI volume data (x, y, z, modalities).
        plane (str): The plane of visualization ('axial', 'sagittal', 'coronal').
    """
    plt.figure(figsize=(10, 5))

    if plane == 'axial':
        # Visualize axial plane (default)
        plt.imshow(input_data[:, :, layer, MRI_Seq], cmap='gray', origin="lower")
        plt.title(f'Axial Plane - Layer {layer}', fontsize=16)

    elif plane == 'sagittal':
        # Visualize sagittal plane
        plt.imshow(rotate(input_data[layer,:,:, MRI_Seq], 90, resize=True), cmap="gray", origin="lower")
        plt.title(f'Sagittal Plane - Layer {layer}', fontsize=16)

    elif plane == 'coronal':
        # Visualize coronal plane
        plt.imshow(rotate(input_data[:, layer, :,MRI_Seq], 90, resize=True), cmap='gray', origin="lower")
        plt.title(f'Coronal Plane - Layer {layer}', fontsize=16)

    else:
        raise ValueError("Invalid plane argument. Choose 'axial', 'sagittal', or 'coronal'.")

    plt.axis('off')
    plt.show()


def explore_4dimage_axial(layer, MRI_Seq, input_data):
    plt.figure(figsize=(10, 5))
    plt.imshow(input_data[:, :, layer,MRI_Seq], cmap='gray', origin="lower")
    #plt.title('Explore Layers of Brain MRI', fontsize=18)
    plt.axis('off')
    plt.show()


def explore_4dimage_sagittal(layer, MRI_Seq, input_data):
    plt.figure(figsize=(6, 6))
    plt.imshow(rotate(input_data[layer,:,:, MRI_Seq], 90, resize=True), cmap="gray", origin="lower")
    #plt.title('Explore Layers of Brain MRI', fontsize=18)
    plt.axis('off')
    plt.show()

def explore_4dimage_coronal(layer, MRI_Seq, input_data):
    plt.figure(figsize=(6, 6))
    plt.imshow(rotate(input_data[:, layer, :,MRI_Seq], 90, resize=True), cmap='gray', origin="lower")
    #plt.title('Explore Layers of Brain MRI', fontsize=18)
    plt.axis('off')
    plt.show()

# Define a function to visualize 3D MRI data in different planes
def explore_3dimage(layer, input_data, plane='axial'):
    """
    Visualizes a single layer of the 3D MRI volume in the specified plane (axial, sagittal, or coronal).

    Args:
        layer (int): The index of the slice to visualize.
        input_data (numpy array): 3D MRI volume data (x, y, z).
        plane (str): The plane of visualization ('axial', 'sagittal', 'coronal').
    """
    plt.figure(figsize=(10, 5))

    if plane == 'axial':
        # Visualize axial plane (default)
        plt.imshow(input_data[:, :, layer], cmap='gray', origin="lower")
        plt.title(f'Axial Plane - Layer {layer}', fontsize=16)

    elif plane == 'sagittal':
        # Visualize sagittal plane
        plt.imshow(rotate(input_data[layer,:,:], 90, resize=True), cmap="gray", origin="lower")
        plt.title(f'Sagittal Plane - Layer {layer}', fontsize=16)

    elif plane == 'coronal':
        # Visualize coronal plane
        plt.imshow(rotate(input_data[:, layer, :], 90, resize=True), cmap='gray', origin="lower")
        plt.title(f'Coronal Plane - Layer {layer}', fontsize=16)

    else:
        raise ValueError("Invalid plane argument. Choose 'axial', 'sagittal', or 'coronal'.")

    plt.axis('off')
    plt.show()

def explore_3dimage_axial(layer, input_data):
    plt.figure(figsize=(10, 5))
    plt.imshow(input_data[:,:, layer], cmap='gray', origin='lower')
    #plt.title('Explore Layers of Brain MRI', fontsize=18)
    plt.axis('off')
    #plt.colorbar(label='Signal intensity',use_gridspec= False)
    plt.show()

def display_volume_slices(volume, start_slice=66, num_slices=10, num_rows=2, num_cols=5, cmap="gray"):
  """
  Displays a grid of slices from a 3D volume.

  Args:
    volume: The 3D volume data (NumPy array).
    start_slice: The index of the first slice to display.
    num_slices: The number of consecutive slices to display.
    cmap: The colormap to use for visualization.

  Example usage:
    display_volume_slices(image_data, start_slice=66, num_slices=10)
  """
  fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 6))  # Adjust rows/cols as needed

  for i in range(num_slices):
      ax = axes[i // num_cols, i % num_cols]  # Get the correct subplot
      ax.imshow(volume[:, :, start_slice + i], cmap=cmap, origin="lower")
      ax.axis('off')

  plt.tight_layout()  # Improves spacing between subplots
  plt.show()


#define a function to explore the histogram of 3D image of axial slices of the brain
def explore_3dimage_histogram(input_data):
    histo, bin_edges = np.histogram(input_data[50:195,20:202,10:142], bins=70)
    _ = plt.hist(histo, bins=bin_edges)
    plt.title('Histogram of the selected MRI sequence')
    plt.xlabel("Signal intensity")
    plt.ylabel("Number of voxels")

#define a function to explore histogram of 3D image of axial slices of the brain per layer
def explore_3dimage_histo_per_layer(layer,input_data):
    img1=input_data[50:195,20:202,layer]
    histo, bin_edges = np.histogram(img1, bins=15)
    _, axis = plt.subplots(ncols=2, figsize=(12, 3))
    axis[0].imshow(img1,'gray',origin='lower')
    axis[0].set_title('Brain Slice in the Axial Plane with MRI')
    axis[0].axis('off')
    axis[1].hist(histo, bins=bin_edges)
    axis[1].set_title('Histogram of the selected slice')
    axis[1].set_xlabel("Signal intensity")
    axis[1].set_ylabel("Number of voxels")
    plt.show()

# Generate RGB masks for predictions and ground truth
def create_rgb_masks(tumor_mask, label):
    return color.gray2rgb(np.where(tumor_mask == label, 255, 0))

# Function to visualize the predicted and ground truth tumor regions overlaid on the MRI scan
def explore_labeled_image(layer, MRI_Seq, input_data, predict_seg, gt_seg):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot predicted tumor regions
    ax1.imshow(input_data[:, :, layer, MRI_Seq], cmap='gray', interpolation='none', origin='lower')
    ax1.imshow(predict_seg[:, :, layer], cmap='jet', interpolation='none', alpha=0.4, origin='lower')
    red_patch = mpatches.Patch(color='red', label='Edema/Invation')
    blue_patch = mpatches.Patch(color='blue', label='Necrosis')
    green_patch = mpatches.Patch(color='green', label='Enhancing Tumor')
    ax1.legend(handles=[red_patch, green_patch, blue_patch], loc='upper right')
    ax1.set_title('Prediction', fontsize=14)
    ax1.axis('off')

    # Plot ground truth tumor regions
    ax2.imshow(input_data[:, :, layer, MRI_Seq], cmap='gray', interpolation='none', origin='lower')
    ax2.imshow(gt_seg[:, :, layer], cmap='jet', interpolation='none', alpha=0.4, origin='lower')
    ax2.legend(handles=[red_patch, green_patch, blue_patch], loc='upper right')
    ax2.set_title('Ground Truth', fontsize=14)
    ax2.axis('off')

    plt.tight_layout()
    plt.show()