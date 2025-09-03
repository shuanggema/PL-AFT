import numpy as np
import pandas as pd
from skimage import measure, color, filters, exposure, morphology, segmentation
from skimage.feature import graycomatrix, graycoprops, peak_local_max
from scipy.ndimage import distance_transform_edt
from scipy.stats import skew
import openslide
import argparse  
import os  

parser = argparse.ArgumentParser()
parser.add_argument("--svs_file", type=str, help="File path for the whole-slide image (WSI) in SVS format")
parser.add_argument("--output_folder", type=str, help="Folder to save the extracted features in CSV format")


def extract_features(svs_file, level=1, crop_size=512):
    """
    Extracts morphological and texture features from a whole-slide image (WSI) at the cell level.
    :param svs_file: str, path to the whole-slide image (WSI) file in SVS format
    :param level: int, resolution level to read the image (default: 2)
    :param crop_size: int, size of the sub-image (crop) to extract features (default: 512)
    :return: DataFrame, feature vectors for all sub-images
    """ 
    # Read the whole-slide image (WSI) using OpenSlide
    slide = openslide.OpenSlide(svs_file)
    
    # Define the size of the crops (512x512 pixels)
    min_cells_threshold = 50

    # Extract the dimensions of the image at the desired resolution level (e.g., level 0 for highest resolution)
    
    image_width, image_height = slide.level_dimensions[level]

    # Initialize an empty list to store feature vectors for all sub-images
    feature_vectors = []

    # Loop over the image and crop it into sub-images
    for x in range(0, image_width, crop_size):
        for y in range(0, image_height, crop_size):
            # Ensure that the crop doesn't exceed the image dimensions
            width = min(crop_size, image_width - x)
            height = min(crop_size, image_height - y)
            
            # Read the region at the current coordinates (x, y)
            sub_image = slide.read_region((x, y), level, (width, height)).convert("RGB")

            gray_scale_image = color.rgb2gray(sub_image)
            if gray_scale_image.std() < 0.05:
                #print(f"Skip {x}, {y} due to low contrast")
                continue
            
            # Convert the sub-image to grayscale with integer values (0-255) using scikit-image
            gray_scale_image = exposure.equalize_adapthist(gray_scale_image, clip_limit=0.01)
            gray_scale_image = filters.gaussian(gray_scale_image, sigma=0.1)
            gray_image = (gray_scale_image * 255).astype(np.uint8)  # Retain integer intensity values

            # Perform Otsu's thresholding for segmentation at the cell level
            thresh = filters.threshold_otsu(gray_scale_image)
            binary_image = gray_scale_image < thresh  # Otsu thresholding for cell segmentation
            pre_label = measure.label(binary_image)
            cleaned_image = morphology.remove_small_objects(pre_label, min_size=100)
            cleaned_binary_region = cleaned_image > 0

            distance = distance_transform_edt(cleaned_binary_region)
            # Watershed segmentation
            local_maxi = peak_local_max(distance, footprint=morphology.disk(5), labels=cleaned_binary_region)
            mask = np.zeros_like(gray_image, dtype=bool)
            mask[tuple(local_maxi.T)] = True
            markers = measure.label(mask)
            labels = segmentation.watershed(-distance, markers, mask=cleaned_binary_region)
            cleaned_seg_labels = morphology.remove_small_objects(labels, min_size=100)

            if len(np.unique(cleaned_seg_labels)) < min_cells_threshold:
                #print(f"Skip {x}, {y} due to low cell count")
                continue

            # Initialize lists to collect features for the sub-image
            features = {
                'Area': [], 'Perimeter': [], 'Eccentricity': [], 'Solidity': [], 'Aspect Ratio': [],
                'Intensity Mean': [], 'Intensity Median': [], 'Intensity SD': [], 'Intensity Max': [], 'Intensity Min': [], 'Intensity Skewness': [],
                'Texture Contrast Mean': [], 'Texture Contrast Median': [], 'Texture Contrast SD': [], 'Texture Contrast Max': [], 'Texture Contrast Min': [],
                'Texture Homogeneity Mean': [], 'Texture Homogeneity Median': [], 'Texture Homogeneity SD': [], 'Texture Homogeneity Max': [], 'Texture Homogeneity Min': [],
                'Texture Energy Mean': [], 'Texture Energy Median': [], 'Texture Energy SD': [], 'Texture Energy Max': [], 'Texture Energy Min': [],
                'Texture Correlation Mean': [], 'Texture Correlation Median': [], 'Texture Correlation SD': [], 'Texture Correlation Max': [], 'Texture Correlation Min': []
            }

            # Extract Morphological Features (Cell Level)
            regions = measure.regionprops(cleaned_seg_labels, intensity_image=gray_scale_image)
            for region in regions:
                area = region.area
                perimeter = region.perimeter
                eccentricity = region.eccentricity
                solidity = region.solidity
                aspect_ratio = region.major_axis_length / region.minor_axis_length if region.minor_axis_length > 0 else 0

                features['Area'].append(area)
                features['Perimeter'].append(perimeter)
                features['Eccentricity'].append(eccentricity)
                features['Solidity'].append(solidity)
                features['Aspect Ratio'].append(aspect_ratio)

            # Compute Texture Features (GLCM at Cell Level)
            distances = [5]  # GLCM distance
            angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # GLCM angles

            for region in measure.regionprops(cleaned_seg_labels, intensity_image=gray_image):
                cell_label = region.label
                cell_intensity_values = region.intensity_image[region.image]
                cell_mean_intensity = region.mean_intensity
                cell_median_intensity = np.median(cell_intensity_values)
                cell_min_intensity = cell_intensity_values.min()
                cell_max_intensity = cell_intensity_values.max()
                skewness_intensity = skew(cell_intensity_values)
                std_dev_intensity = cell_intensity_values.std()

                cell_mask = cleaned_seg_labels == cell_label
                cell_gray_values = gray_image * cell_mask
                min_row, min_col, max_row, max_col = region.bbox
                cell_gray_cropped = cell_gray_values[min_row:max_row, min_col:max_col]

                glcm = graycomatrix(cell_gray_cropped, distances=distances, angles=angles, symmetric=True, normed=True)
                contrast = graycoprops(glcm, 'contrast')
                energy = graycoprops(glcm, 'energy')
                homogeneity = graycoprops(glcm, 'homogeneity')
                correlation = graycoprops(glcm, 'correlation')

                contrast_mean = contrast.mean()
                energy_mean = energy.mean()
                homogeneity_mean = homogeneity.mean()
                correlation_mean = correlation.mean()

                contrast_median = np.median(contrast)
                energy_median = np.median(energy)
                homogeneity_median = np.median(homogeneity)
                correlation_median = np.median(correlation)

                contrast_std = contrast.std()
                energy_std = energy.std()
                homogeneity_std = homogeneity.std()
                correlation_std = correlation.std()

                contrast_max = contrast.max()
                energy_max = energy.max()
                homogeneity_max = homogeneity.max()
                correlation_max = correlation.max()

                contrast_min = contrast.min()
                energy_min = energy.min()
                homogeneity_min = homogeneity.min()
                correlation_min = correlation.min()

                features['Texture Contrast Mean'].append(contrast_mean)
                features['Texture Homogeneity Mean'].append(homogeneity_mean)
                features['Texture Energy Mean'].append(energy_mean)
                features['Texture Correlation Mean'].append(correlation_mean)

                features['Texture Contrast Median'].append(contrast_median)
                features['Texture Homogeneity Median'].append(homogeneity_median)
                features['Texture Energy Median'].append(energy_median)
                features['Texture Correlation Median'].append(correlation_median)

                features['Texture Contrast Max'].append(contrast_max)
                features['Texture Homogeneity Max'].append(homogeneity_max)
                features['Texture Energy Max'].append(energy_max)
                features['Texture Correlation Max'].append(correlation_max)

                features['Texture Contrast Min'].append(contrast_min)
                features['Texture Homogeneity Min'].append(homogeneity_min)
                features['Texture Energy Min'].append(energy_min)
                features['Texture Correlation Min'].append(correlation_min)

                features['Texture Contrast SD'].append(contrast_std)
                features['Texture Homogeneity SD'].append(homogeneity_std)
                features['Texture Energy SD'].append(energy_std)
                features['Texture Correlation SD'].append(correlation_std)

                features['Intensity Mean'].append(cell_mean_intensity)
                features['Intensity Median'].append(cell_median_intensity)
                features['Intensity Min'].append(cell_min_intensity)
                features['Intensity Max'].append(cell_max_intensity)
                features['Intensity SD'].append(std_dev_intensity)
                features['Intensity Skewness'].append(skewness_intensity)

            summary_features = {f'{key} Median': np.median(values) for key, values in features.items()}
            summary_features.update({f'{key} Std': np.std(values) for key, values in features.items()})
            summary_features.update({f'{key} Skew': skew(values) for key, values in features.items()})
            summary_features['Cell Number'] = len(np.unique(cleaned_seg_labels))

            feature_vectors.append(summary_features)

    features_df = pd.DataFrame(feature_vectors).mean(axis=0)
    return features_df


if __name__ == "__main__":
    args = parser.parse_args()
    svs_file = args.svs_file
    output_folder = args.output_folder
    df_feature = extract_features(svs_file,
                                  level=0, crop_size=512)
    df_feature.to_csv(os.path.join(output_folder, svs_file + ".csv"), 
                      index=False)

