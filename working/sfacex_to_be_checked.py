import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import statistics
import math
import cv2
import os
import glob
import json
from operator import attrgetter
from datetime import datetime
from skimage.feature import local_binary_pattern, hog
from numba import jit, prange

# Imports for ROC analysis
from sklearn.metrics import roc_curve, auc


# Convert image to grayscale
def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def reverse_code(code):
    # Flip all bits and mask to keep it 8-bit
    reversed_code = ~code & 0xFF
    return reversed_code


# Uniform pattern lookup table (mapping 256 patterns to 59 bins)
def uniform_lbp_mapping():
    table = np.zeros(256, dtype=np.uint8)
    index = 0
    for i in range(256):
        b = format(i, "08b")
        transitions = sum((b[j] != b[(j + 1) % 8]) for j in range(8))
        if transitions <= 2:
            table[i] = index
            index += 1
        else:
            table[i] = 58  # last bin for all non-uniform patterns
    return table


# Median-based thresholding
def calculate_ceiling_median(arr):
    median_value = statistics.median(arr)
    ceiling_median = math.ceil(median_value)
    return ceiling_median


# LBP pixel function using uniform LBP mapping
def lbp_pixel(img, x, y, mapping):
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    center = img[x, y]

    code = 0
    for i, (dx, dy) in enumerate(offsets):
        if img[x + dx, y + dy] >= center:
            code |= 1 << i
    return mapping[(code)]


# Full LBP image
def lbp_image(gray, mapping):
    h, w = gray.shape
    lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            lbp[i - 1, j - 1] = lbp_pixel(gray, i, j, mapping)
    return lbp


# # Divide into grid cells
# def divide_into_grids(img, grid_x=8, grid_y=8):
#     h, w = img.shape
#     grid_h, grid_w = h // grid_y, w // grid_x
#     grids = []
#     for i in range(grid_y):
#         for j in range(grid_x):
#             cell = img[i * grid_h : (i + 1) * grid_h, j * grid_w : (j + 1) * grid_w]
#             grids.append(cell)
#     return grids


def divide_into_grids(img, grid_x=8, grid_y=8):
    h, w = img.shape
    grid_h, grid_w = h // grid_y, w // grid_x
    grids = []
    for i in range(grid_y):
        for j in range(grid_x):
            start_h = i * grid_h
            end_h = (
                (i + 1) * grid_h if i < grid_y - 1 else h
            )  # Include remaining pixels in last row
            start_w = j * grid_w
            end_w = (
                (j + 1) * grid_w if j < grid_x - 1 else w
            )  # Include remaining pixels in last column

            cell = img[start_h:end_h, start_w:end_w]
            grids.append(cell)
    return grids


# Compute normalized histograms
def grid_histograms(grids, bins=59):
    hists = np.array(
        [
            np.histogram(cell.ravel(), bins=bins, range=(0, bins))[0].astype("float32")
            for cell in grids
        ]
    )
    hists /= np.linalg.norm(hists, axis=1, keepdims=True) + 1e-6
    return hists


def extract_lbp_features(image_path_or_cv2_img):
    starting_time = time.time()
    print("Loading image...")
    gray = preprocessing_image_optimized(image_path_or_cv2_img)
    mapping = np.load("./uniform_lbp_table.npy")  # Load precomputed mapping table

    # Generate LBP image using uniform LBP
    lbp = lbp_image(gray, mapping)

    # Divide into grids and compute histograms
    grids = divide_into_grids(lbp, grid_x=8, grid_y=8)
    histograms = grid_histograms(grids, bins=59)

    # Flatten and normalize the final feature vector
    final_vector = np.concatenate(histograms)
    final_vector /= np.linalg.norm(final_vector) + 1e-6  # L2 normalization

    print("Feature vector shape:", final_vector.shape)
    print("Execution time:", time.time() - starting_time)

    return final_vector


# def extract_lbp_features(image_path_or_cv2_img):
#     """
#     Extracts a multi-scale LBP feature vector from a given image.
#     This version uses scikit-image for efficient LBP computation at multiple scales.
#     """
#     starting_time = time.time()
#     print("Loading and preprocessing image...")

#     # 1. Preprocess the image (make sure this includes face detection and alignment!)
#     gray = preprocessing_image_optimized(image_path_or_cv2_img)
#     if gray is None:
#         print("Preprocessing failed; cannot extract features.")
#         return None

#     # Define the LBP configurations (radius, number_of_points)
#     # You can experiment with these values
#     lbp_configs = [
#         {"radius": 1, "n_points": 8},
#         {"radius": 2, "n_points": 16},
#         {"radius": 3, "n_points": 24},
#     ]

#     # Grid configuration
#     grid_x, grid_y = 8, 8

#     # List to hold the histograms from all scales
#     all_scale_histograms = []

#     print("Extracting LBP features at multiple scales...")
#     for config in lbp_configs:
#         radius = config["radius"]
#         n_points = config["n_points"]

#         # 2. Compute LBP using scikit-image
#         # The 'uniform' method automatically handles mapping to uniform patterns.
#         lbp = local_binary_pattern(gray, n_points, radius, method="uniform")

#         # The number of bins for 'uniform' LBP is n_points + 2
#         n_bins = n_points + 2

#         # 3. Divide into grids and compute histograms for the current scale
#         grids = divide_into_grids(lbp, grid_x, grid_y)

#         # Compute histograms for this scale. Note the dynamic `bins` and `range`.
#         hists = np.array(
#             [
#                 np.histogram(cell.ravel(), bins=n_bins, range=(0, n_bins))[0].astype(
#                     "float32"
#                 )
#                 for cell in grids
#             ]
#         )
#         # Normalize histograms for this scale
#         hists /= np.linalg.norm(hists, axis=1, keepdims=True) + 1e-6

#         # Flatten and append to our list of all histograms
#         all_scale_histograms.append(hists.flatten())

#     # 4. Concatenate histograms from all scales into one final feature vector
#     final_vector = np.concatenate(all_scale_histograms)

#     # 5. Apply a final L2 normalization to the entire vector
#     final_vector /= np.linalg.norm(final_vector) + 1e-6

#     print("Multi-scale feature vector shape:", final_vector.shape)
#     print("Execution time:", time.time() - starting_time)

#     return final_vector


@jit(nopython=True, parallel=True)
def custom_local_binary_pattern_numba(gray_image, n_points, radius, method="uniform"):
    """
    Numba-optimized version of your custom LBP implementation.
    Maintains exactly the same logic and behavior as the original.
    """
    gray_image = np.asarray(gray_image, dtype=np.float32)
    height, width = gray_image.shape
    lbp_image = np.zeros((height, width), dtype=np.uint8)

    for y in prange(height):  # prange for parallel execution
        for x in prange(width):
            center_pixel_value = gray_image[y, x]
            binary_pattern = np.zeros(n_points, dtype=np.int32)

            # Step 1: Find neighbors and generate binary pattern
            for i in range(n_points):
                angle = 2 * np.pi * i / n_points
                neighbor_y = y - radius * np.sin(angle)
                neighbor_x = x + radius * np.cos(angle)

                # Step 2: Bilinear Interpolation (same logic)
                y1 = int(np.floor(neighbor_y))
                x1 = int(np.floor(neighbor_x))
                y2 = int(np.ceil(neighbor_y))
                x2 = int(np.ceil(neighbor_x))

                wy = neighbor_y - y1
                wx = neighbor_x - x1

                # Boundary handling (same logic)
                y1 = max(0, min(y1, height - 1))
                y2 = max(0, min(y2, height - 1))
                x1 = max(0, min(x1, width - 1))
                x2 = max(0, min(x2, width - 1))

                q11 = gray_image[y1, x1]
                q12 = gray_image[y1, x2]
                q21 = gray_image[y2, x1]
                q22 = gray_image[y2, x2]

                r1 = (1 - wx) * q11 + wx * q12
                r2 = (1 - wx) * q21 + wx * q22
                interpolated_value = (1 - wy) * r1 + wy * r2

                # Step 3: Compare (same logic)
                if interpolated_value >= center_pixel_value:
                    binary_pattern[i] = 1
                else:
                    binary_pattern[i] = 0

            # Step 4: Convert to LBP code (same logic)
            if method == "uniform":
                # Count transitions (same logic as original)
                transitions = 0
                for j in range(len(binary_pattern) - 1):
                    if binary_pattern[j] != binary_pattern[j + 1]:
                        transitions += 1
                if binary_pattern[-1] != binary_pattern[0]:
                    transitions += 1

                if transitions <= 2:
                    lbp_code = np.sum(binary_pattern)  # Sum of 1's
                else:
                    lbp_code = n_points + 1
            else:  # "default" method
                lbp_code = 0
                for bit_index in range(len(binary_pattern)):
                    lbp_code += binary_pattern[bit_index] * (2**bit_index)

            lbp_image[y, x] = lbp_code

    return lbp_image


def custom_local_binary_pattern(gray_image, n_points, radius, method="uniform"):
    """
    Implements a custom version of the Local Binary Pattern (LBP) feature descriptor
    to replicate the behavior of scikit-image's implementation, including the
    "uniform" method and bilinear interpolation for non-integer coordinates.

    Args:
        gray_image (np.ndarray): The input grayscale image. Should be 2D.
        n_points (int): The number of circularly symmetric neighbor points.
        radius (int or float): The radius of the circle of neighbors.
        method (str): The method to use. Supports "uniform" or "default".

    Returns:
        np.ndarray: The LBP image, where each pixel contains its LBP code.
    """
    # Ensure the input image is a numpy array
    gray_image = np.asarray(gray_image, dtype=np.float32)
    height, width = gray_image.shape

    # Create an output image for the LBP codes, initialized to zeros.
    # The padding (radius) is to handle pixels at the image borders.
    lbp_image = np.zeros((height, width), dtype=np.uint8)

    # Iterate over each pixel of the image. We will process from (0,0) to (h,w)
    # and handle borders inside the loop.
    for y in range(height):
        for x in range(width):
            # Get the intensity of the center pixel
            center_pixel_value = gray_image[y, x]

            # This list will store the binary result of the comparisons
            binary_pattern = []

            # --- Step 1: Find neighbors and generate binary pattern ---
            for i in range(n_points):
                # Calculate the angle for the current neighbor point
                angle = 2 * np.pi * i / n_points

                # Calculate the floating-point coordinates of the neighbor.
                # The y-coordinate is subtracted because image y-axes are inverted (0 is at the top).
                neighbor_y = y - radius * np.sin(angle)
                neighbor_x = x + radius * np.cos(angle)

                # --- Step 2: Perform Bilinear Interpolation ---
                # Since neighbor coordinates are floats, we find the intensity
                # by interpolating between the four nearest integer pixels.

                # Top-left integer coordinates
                y1, x1 = int(np.floor(neighbor_y)), int(np.floor(neighbor_x))
                # Bottom-right integer coordinates
                y2, x2 = int(np.ceil(neighbor_y)), int(np.ceil(neighbor_x))

                # Get interpolation weights
                wy = neighbor_y - y1
                wx = neighbor_x - x1

                # Handle boundary conditions by clamping coordinates
                y1 = max(0, min(y1, height - 1))
                y2 = max(0, min(y2, height - 1))
                x1 = max(0, min(x1, width - 1))
                x2 = max(0, min(x2, width - 1))

                # Get the intensity of the four corner pixels
                q11 = gray_image[y1, x1]  # Top-left
                q12 = gray_image[y1, x2]  # Top-right
                q21 = gray_image[y2, x1]  # Bottom-left
                q22 = gray_image[y2, x2]  # Bottom-right

                # Interpolate along the x-axis
                r1 = (1 - wx) * q11 + wx * q12
                r2 = (1 - wx) * q21 + wx * q22

                # Interpolate along the y-axis to get the final value
                interpolated_value = (1 - wy) * r1 + wy * r2

                # --- Step 3: Compare and build the binary pattern ---
                if interpolated_value >= center_pixel_value:
                    binary_pattern.append(1)
                else:
                    binary_pattern.append(0)

            # --- Step 4: Convert the binary pattern to an LBP code ---
            if method == "uniform":
                # A uniform pattern has at most two 0->1 or 1->0 transitions.
                # We check this by creating a string and checking transitions,
                # including the wrap-around from the last to the first bit.
                binary_str = "".join(map(str, binary_pattern))
                transitions = 0
                for j in range(len(binary_str) - 1):
                    if binary_str[j] != binary_str[j + 1]:
                        transitions += 1
                if binary_str[-1] != binary_str[0]:  # Check wrap-around transition
                    transitions += 1

                if transitions <= 2:
                    # For uniform patterns, the LBP code is the number of '1's
                    lbp_code = sum(binary_pattern)
                else:
                    # Non-uniform patterns are assigned a single special value
                    lbp_code = n_points + 1
            else:  # "default" method
                # Convert the binary list to a decimal number
                lbp_code = 0
                for bit_index, bit_value in enumerate(binary_pattern):
                    lbp_code += bit_value * (2**bit_index)

            lbp_image[y, x] = lbp_code

    return lbp_image


def extract_combined_features(image_path_or_cv2_img):
    """
    Extracts a combined Multi-Scale LBP and HOG feature vector.
    """
    starting_time = time.time()
    print("Loading and preprocessing image for combined features...")

    # # 1. Preprocess the image (this MUST return an aligned, normalized grayscale face)
    gray_face = preprocessing_image_optimized(image_path_or_cv2_img)
    if gray_face is None:
        print("Preprocessing failed; cannot extract features.")
        return None

    # --- Part A: Multi-Scale LBP Feature Extraction (as before) ---
    print("Extracting Multi-Scale LBP features...")
    lbp_configs = [
        {"radius": 1, "n_points": 8},
        {"radius": 2, "n_points": 16},
        {"radius": 3, "n_points": 24},
    ]
    grid_x, grid_y = 8, 8
    all_scale_histograms = []

    for config in lbp_configs:
        radius, n_points = config["radius"], config["n_points"]
        lbp = custom_local_binary_pattern_numba(
            gray_face, n_points, radius, method="uniform"
        )
        n_bins = n_points + 2
        grids = divide_into_grids(lbp, grid_x, grid_y)
        hists = np.array(
            [
                np.histogram(cell.ravel(), bins=n_bins, range=(0, n_bins))[0]
                for cell in grids
            ]
        )
        hists = hists.astype("float32")
        # L2-normalize the histograms for this LBP scale
        hists /= np.linalg.norm(hists, axis=1, keepdims=True) + 1e-6
        all_scale_histograms.append(hists.flatten())

    # Concatenate all LBP scale histograms and normalize the final LBP vector
    lbp_vector = np.concatenate(all_scale_histograms)
    lbp_vector /= np.linalg.norm(lbp_vector) + 1e-6

    # lbp_vector = extract_lbp_features(image_path_or_cv2_img=image_path_or_cv2_img)

    # --- Part B: HOG Feature Extraction ---
    print("Extracting HOG features...")
    # These are standard HOG parameters that work well for faces.
    # `feature_vector=True` returns a flattened 1D array.
    hog_vector = hog(
        gray_face,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False,
        transform_sqrt=True,
        feature_vector=True,
    )

    # It's good practice to ensure it's float32 and normalized
    hog_vector = hog_vector.astype("float32")
    hog_vector /= np.linalg.norm(hog_vector) + 1e-6

    # --- Part C: Concatenate LBP and HOG vectors ---
    print("Combining LBP and HOG feature vectors...")
    # The final super-vector
    combined_vector = np.concatenate([lbp_vector, hog_vector])

    # Final normalization of the combined vector
    combined_vector /= np.linalg.norm(combined_vector) + 1e-6

    print(f"LBP vector shape: {lbp_vector.shape}")
    print(f"HOG vector shape: {hog_vector.shape}")
    print(f"Combined feature vector shape: {combined_vector.shape}")
    print(f"Total execution time: {time.time() - starting_time:.2f} seconds")

    # if combined_vector.shape[0] != 11556:
    #     print(f"Feature vector shape mismatch: {combined_vector.shape}")
    #     return None
    print("Feature vector shape:", combined_vector.shape)
    return combined_vector


def preprocessing_image_optimized(image_input, target_size=(128, 128)):
    """Streamlined preprocessing with fewer operations"""
    if isinstance(image_input, str):  # If it's a file path
        image = cv2.imread(image_input)
        if image is None:
            raise ValueError("Image not found or path is incorrect.")
    elif isinstance(image_input, np.ndarray):  # If it's an OpenCV frame
        image = image_input
    else:
        raise TypeError(
            "Input must be a file path (str) or an OpenCV image (np.ndarray)."
        )

    # Convert to grayscale and resize
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_gray = cv2.resize(
        gray, target_size, interpolation=cv2.INTER_AREA
    )  # Fixed variable name

    # Apply CLAHE to the resized image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized_gray)  # Apply to resized image

    return enhanced  # Now returns the correctly sized image


def get_uniform_lbp_mapping(table_path="./uniform_lbp_table.npy"):
    """
    Loads the uniform LBP mapping table from a file, or generates it if not found.
    """
    if os.path.exists(table_path):
        return np.load(table_path)
    else:
        print(f"LBP mapping table not found at {table_path}. Generating a new one.")
        return uniform_lbp_mapping()


class TemplateEntry:
    def __init__(
        self,
        id,
        template_vector,
        pose_label,
        source="enrollment",
        initial_sample_count=5,
    ):
        self.id = id  # Unique ID, e.g., "pose_left"
        self.template_vector = (
            template_vector  # The MEAN LBP feature vector for this pose
        )
        self.pose_label = (
            pose_label  # Critical new field: "center", "left", "right", "up", "down"
        )
        self.source = source  # "enrollment" or "adapted_positive"
        self.match_count = 0  # To track how often this pose is the best match
        self.sample_count = initial_sample_count  # How many images were averaged to create this template
        self.creation_timestamp = time.time()
        self.last_matched_timestamp = time.time()

    def to_dict(self):
        """Convert TemplateEntry to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "template_vector": self.template_vector.tolist(),  # Convert numpy array to list
            "pose_label": self.pose_label,
            "source": self.source,
            "match_count": self.match_count,
            "sample_count": self.sample_count,
            "creation_timestamp": self.creation_timestamp,
            "last_matched_timestamp": self.last_matched_timestamp,
        }

    @classmethod
    def from_dict(cls, data):
        """Create TemplateEntry from dictionary (for JSON deserialization)"""
        template_entry = cls(
            id=data["id"],
            template_vector=np.array(
                data["template_vector"]
            ),  # Convert list back to numpy array
            pose_label=data["pose_label"],
            source=data.get("source", "enrollment"),
            initial_sample_count=data.get("sample_count", 5),
        )
        template_entry.match_count = data.get("match_count", 0)
        template_entry.sample_count = data.get("sample_count", 5)
        template_entry.creation_timestamp = data.get("creation_timestamp", time.time())
        template_entry.last_matched_timestamp = data.get(
            "last_matched_timestamp", time.time()
        )
        return template_entry


def chi_square_distance(hist1, hist2, eps=1e-10):
    """
    Computes the Chi-Square distance between two histograms (feature vectors).
    """
    hist1 = np.asarray(hist1, dtype=np.float32)
    hist2 = np.asarray(hist2, dtype=np.float32)

    # The Chi-Square distance: 0.5 * sum(((a - b)^2) / (a + b))
    distance = 0.5 * np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2 + eps))
    return distance


# --- LBP Chi-Square Authenticator Class ---
class LBPChiSquareAuthenticator:
    def __init__(
        self,
        target_size=(128, 128),
        grid_cells=(8, 8),
        num_lbp_bins=59,
        num_templates=5,
        ratio_threshold=0.7,
        adaptation_threshold_factor=1.25,
        max_adapted_templates=5,
        similarity_check_factor=0.3,
        data_file="./authenticator_data.npz",
    ):
        self.target_size = target_size
        self.grid_cells = grid_cells
        self.num_lbp_bins = num_lbp_bins
        self.initial_num_templates = num_templates
        self.max_total_templates = num_templates + max_adapted_templates
        self.new_template_distance = 0
        self.enrolled_templates_data = []
        self.threshold = None
        self.data_file = data_file  # JSON file to store persistent data

        self.ratio_threshold = (
            ratio_threshold  # A value between 0 and 1. Lower is stricter.
        )

        self.unlock_method = None
        self.last_auth_result = None

        # Parameters for Adaptation
        self.adaptation_threshold_factor = adaptation_threshold_factor
        self.similarity_check_factor = similarity_check_factor

        self.last_failed_attempt_features = None
        self.last_failed_attempt_distance = None

        # Additional metadata to persist
        self.enrollment_date = None
        self.total_authentications = 0
        self.successful_authentications = 0
        self.model_version = "1.0"

        # Ensure LBP mapping table is generated/loaded once
        get_uniform_lbp_mapping()

        # Load existing data if available
        self.load_data()

    def save_data(self):
        """Save all important data in .npz format"""
        try:
            data_to_save = {
                "model_version": self.model_version,
                "pin": "123456",
                "enrollment_date": self.enrollment_date,
                "ratio_threshold": self.ratio_threshold,
                "total_authentications": self.total_authentications,
                "successful_authentications": self.successful_authentications,
                "threshold": float(self.threshold),
                "new_template_distance": float(self.new_template_distance),
                "target_size": self.target_size,
                "grid_cells": self.grid_cells,
                "num_lbp_bins": self.num_lbp_bins,
                "initial_num_templates": self.initial_num_templates,
                "max_total_templates": self.max_total_templates,
                "adaptation_threshold_factor": self.adaptation_threshold_factor,
                "similarity_check_factor": self.similarity_check_factor,
                "enrollment_date": self.enrollment_date,
                "last_saved": datetime.now().isoformat(),
                "feature_vector_length": (
                    len(self.enrolled_templates_data[0].template_vector)
                    if self.enrolled_templates_data
                    else None
                ),
                "template_ids": [t.id for t in self.enrolled_templates_data],
                "pose_labels": [t.pose_label for t in self.enrolled_templates_data],
                "sources": [t.source for t in self.enrolled_templates_data],
                "match_counts": [t.match_count for t in self.enrolled_templates_data],
                "sample_counts": [t.sample_count for t in self.enrolled_templates_data],
                "creation_timestamps": [
                    t.creation_timestamp for t in self.enrolled_templates_data
                ],
                "last_matched_timestamps": [
                    t.last_matched_timestamp for t in self.enrolled_templates_data
                ],
            }

            template_vectors = np.array(
                [t.template_vector for t in self.enrolled_templates_data],
                dtype=np.float32,
            )

            np.savez(
                self.data_file,
                metadata=json.dumps(data_to_save),
                template_vectors=template_vectors,
            )

            print(f"Data saved successfully to {self.data_file}")
            return True
        except Exception as e:
            print(f"Error saving data: {e}")
            return False

    def load_data(self):
        """Load data from .npz file"""
        try:
            if not os.path.exists(self.data_file):
                print(f"No data file found at {self.data_file}")
                return False

            npzfile = np.load(self.data_file, allow_pickle=True)
            metadata = json.loads(npzfile["metadata"].item())
            template_vectors = npzfile["template_vectors"]

            self.model_version = metadata.get("model_version", "1.0")
            self.pin = metadata.get("pin", "123456")  #
            self.enrollment_date = metadata.get("enrollment_date")
            self.total_authentications = metadata.get("total_authentications", 0)
            self.successful_authentications = metadata.get(
                "successful_authentications", 0
            )
            self.threshold = metadata.get("threshold")
            self.ratio_threshold = metadata.get("ratio_threshold", 0.7)
            self.new_template_distance = metadata.get("new_template_distance")
            self.target_size = tuple(metadata.get("target_size", (128, 128)))
            self.grid_cells = tuple(metadata.get("grid_cells", (8, 8)))
            self.num_lbp_bins = metadata.get("num_lbp_bins", 59)
            self.initial_num_templates = metadata.get("initial_num_templates", 5)
            self.max_total_templates = metadata.get("max_total_templates", 10)
            self.adaptation_threshold_factor = metadata.get(
                "adaptation_threshold_factor", 1.25
            )
            self.similarity_check_factor = metadata.get("similarity_check_factor", 0.3)

            self.enrolled_templates_data = []
            ids = metadata["template_ids"]
            poses = metadata["pose_labels"]
            sources = metadata["sources"]
            match_counts = metadata["match_counts"]
            sample_counts = metadata["sample_counts"]
            creation_times = metadata["creation_timestamps"]
            last_matched = metadata["last_matched_timestamps"]

            for i in range(len(ids)):
                template = TemplateEntry(
                    id=ids[i],
                    template_vector=template_vectors[i],
                    pose_label=poses[i],
                    source=sources[i],
                    initial_sample_count=sample_counts[i],
                )
                template.match_count = match_counts[i]
                template.creation_timestamp = creation_times[i]
                template.last_matched_timestamp = last_matched[i]
                self.enrolled_templates_data.append(template)

            print(f"Data loaded successfully from {self.data_file}")
            return True

        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def backup_data(self, backup_suffix=None):
        """Create a backup of the current data file"""
        try:
            if not os.path.exists(self.data_file):
                print("No data file to backup")
                return False

            if backup_suffix is None:
                backup_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")

            backup_file = f"{self.data_file}.backup_{backup_suffix}"

            with open(self.data_file, "r") as src:
                with open(backup_file, "w") as dst:
                    dst.write(src.read())

            print(f"Backup created: {backup_file}")
            return True

        except Exception as e:
            print(f"Error creating backup: {str(e)}")
            return False

    def _extract_features(self, image_path_or_cv2_img):
        """Helper to call the main feature extraction pipeline."""
        return extract_combined_features(image_path_or_cv2_img)

    def enroll_person_template(self, list_of_person_image_paths):
        """
        Enrolls a person by creating multiple TemplateEntry objects.
        """
        if len(list_of_person_image_paths) < self.initial_num_templates:
            print(
                f"Error: Enrollment requires at least {self.initial_num_templates} images, but received {len(list_of_person_image_paths)}."
            )
            return False

        self.enrolled_templates_data = []  # Clear any previous enrollment
        print(f"Enrolling person using {self.initial_num_templates} templates...")

        successful_extractions = 0
        for i in range(self.initial_num_templates):
            img_path = list_of_person_image_paths[i]
            print(
                f"  Processing enrollment image {i+1}/{self.initial_num_templates}: {os.path.basename(img_path)}"
            )
            features = self._extract_features(img_path)
            if features is not None and features.size > 0:
                template_entry = TemplateEntry(
                    id=i, template_vector=features, pose_label=f"template_{i}"
                )
                self.enrolled_templates_data.append(template_entry)
                successful_extractions += 1
            else:
                print(
                    f"  Warning: Could not extract features for enrollment image: {img_path}. Skipping this template."
                )

        if successful_extractions < self.initial_num_templates:
            print(
                f"Enrollment failed: Only {successful_extractions}/{self.initial_num_templates} templates could be successfully created."
            )
            self.enrolled_templates_data = []
            return False

        self.enrollment_date = datetime.now().isoformat()
        print(
            f"Enrollment successful. {len(self.enrolled_templates_data)} reference LBP templates created."
        )

        # Save data after successful enrollment
        self.save_data()
        return True

    def enroll_person_with_poses(self, pose_image_dict):
        """
        Enrolls a person using a structured dictionary of images for specific poses.
        """
        self.enrolled_templates_data = []  # Clear any previous enrollment
        print("Starting enrollment with specific poses...")

        expected_poses = ["right", "left", "center", "down", "up"]
        print("Expected poses: ", expected_poses)
        self.enrolled_templates_data = []
        template_counter = 0

        if list(pose_image_dict.keys()) != expected_poses:
            print(
                f"Error: Input dictionary must contain keys for all expected poses: {expected_poses}"
            )
            return False

        # for pose, image_paths in pose_image_dict.items():
        #     if not image_paths or len(image_paths) == 0:
        #         print(
        #             f"Error: No images provided for pose '{pose}'. Enrollment failed."
        #         )
        #         self.enrolled_templates_data = []
        #         return False

        #     print(f"  Processing pose: '{pose}' with {len(image_paths)} images...")
        #     pose_features_list = []
        #     for img_path in image_paths:
        #         features = self._extract_features(img_path)
        #         if features is not None and features.size > 0:
        #             pose_features_list.append(features)
        #         else:
        #             print(
        #                 f"    Warning: Could not extract features for {img_path}. Skipping."
        #             )

        #     if not pose_features_list:
        #         print(
        #             f"Error: Failed to extract any features for pose '{pose}'. Enrollment failed."
        #         )
        #         self.enrolled_templates_data = []
        #         return False

        #     # Calculate the mean feature vector
        #     mean_features = np.mean(np.array(pose_features_list), axis=0)

        #     # Create a single TemplateEntry for this pose using the mean features
        #     template_entry = TemplateEntry(
        #         id=f"pose_{pose}",
        #         template_vector=mean_features,
        #         pose_label=pose,
        #         initial_sample_count=len(pose_features_list),
        #     )
        #     self.enrolled_templates_data.append(template_entry)
        for pose, image_paths in pose_image_dict.items():
            print(f"  Processing pose: '{pose}'...")
            for img_path in image_paths:
                features = self._extract_features(img_path)
                if features is not None and features.size > 0:
                    # Create a UNIQUE ID for each individual template
                    template_id = f"{pose}_{template_counter}"

                    # Create a SEPARATE template for EACH image
                    template_entry = TemplateEntry(
                        id=template_id,
                        template_vector=features,
                        pose_label=pose,
                        source="enrollment",
                        initial_sample_count=1,  # Sample count is 1 for an individual image
                    )
                    self.enrolled_templates_data.append(template_entry)
                    template_counter += 1
                else:
                    print(
                        f"    Warning: Could not extract features for {img_path}. Skipping."
                    )

        self.enrollment_date = datetime.now().isoformat()
        print(
            f"Enrollment successful. {len(self.enrolled_templates_data)} pose-based templates created."
        )

        # Save data after successful enrollment
        self.save_data()
        return True

    def get_distance(self, feature):
        templates_to_check_ordered = sorted(
            self.enrolled_templates_data,
            key=attrgetter("match_count"),
            reverse=True,
        )

        overall_min_distance = float("inf")

        for template_entry in templates_to_check_ordered:
            distance = chi_square_distance(feature, template_entry.template_vector)

            if distance < overall_min_distance:
                overall_min_distance = distance

        return overall_min_distance

    def classify_template(self, image_path_or_cv2_img):
        self.last_failed_attempt_features = None
        self.last_failed_attempt_distance = None
        self.total_authentications += 1

        if not self.enrolled_templates_data or self.threshold is None:
            print(
                "Error: System not ready. Please enroll person and calculate threshold first."
            )
            return "error_not_ready", -1.0

        features_to_test = self._extract_features(image_path_or_cv2_img)
        if features_to_test is None or features_to_test.size == 0:
            print("Error: Could not extract features from the provided image.")
            return "error_extracting_features", -1.0

        # Sort templates by match_count (prioritize frequently matched ones)
        templates_to_check_ordered = sorted(
            self.enrolled_templates_data,
            key=attrgetter("match_count"),
            reverse=True,
        )

        overall_min_distance = float("inf")

        for template_entry in templates_to_check_ordered:
            distance = chi_square_distance(
                features_to_test, template_entry.template_vector
            )

            if distance < overall_min_distance:
                overall_min_distance = distance

            if distance < self.threshold:
                # Find the original template to update its mutable attributes
                original_template = next(
                    (
                        t
                        for t in self.enrolled_templates_data
                        if t.id == template_entry.id
                    ),
                    None,
                )
                if original_template:
                    original_template.match_count += 1
                    original_template.last_matched_timestamp = time.time()

                self.successful_authentications += 1
                # Save data after successful authentication
                self.save_data()
                return "enrolled_person", distance

        # If no template matched (authentication failed)
        self.last_failed_attempt_features = features_to_test
        self.last_failed_attempt_distance = overall_min_distance

        # Save data after authentication attempt
        self.save_data()

        # Determine if it's a "near miss"
        near_miss_upper_bound = self.threshold * self.adaptation_threshold_factor
        if self.threshold < overall_min_distance < near_miss_upper_bound:
            return "not_enrolled_person_near_miss", overall_min_distance
        else:
            return "not_enrolled_person_far_miss", overall_min_distance

    # def classify_template(self, image_path_or_cv2_img):
    #     """
    #     Classifies an image using a robust distance ratio test to prevent ambiguity.
    #     """
    #     self.last_failed_attempt_features = None
    #     self.last_failed_attempt_distance = None
    #     self.total_authentications += 1

    #     if (
    #         not self.enrolled_templates_data
    #         or self.threshold is None
    #         or len(self.enrolled_templates_data) < 2
    #     ):
    #         print(
    #             "Error: System not ready. Please enroll and calculate threshold first."
    #         )
    #         print("You need at least two templates for the ratio test.")
    #         return "error_not_ready", -1.0

    #     features_to_test = self._extract_features(image_path_or_cv2_img)
    #     if features_to_test is None or features_to_test.size == 0:
    #         print("Error: Could not extract features from the provided image.")
    #         return "error_extracting_features", -1.0

    #     # --- NEW CLASSIFICATION LOGIC ---

    #     # 1. Calculate the distance from the input to every enrolled template.
    #     # We'll store both the distance and the template's ID for later.
    #     distances_with_ids = []
    #     for template in self.enrolled_templates_data:
    #         dist = chi_square_distance(features_to_test, template.template_vector)
    #         distances_with_ids.append({"distance": dist, "id": template.id})

    #     # 2. Sort the distances to find the two best matches.
    #     sorted_distances = sorted(distances_with_ids, key=lambda x: x["distance"])

    #     best_match = sorted_distances[0]
    #     second_best_match = sorted_distances[1]

    #     dist1 = best_match["distance"]
    #     dist2 = second_best_match["distance"]

    #     # 3. Apply the dual-condition check
    #     # Condition A: The best match must be below our absolute distance threshold.
    #     is_close_enough = dist1 < self.threshold

    #     # Condition B: The best match must be significantly better than the second-best match.
    #     # Add a small epsilon to avoid division by zero if dist2 is 0.
    #     ratio = dist1 / (dist2 + 1e-9)
    #     is_unambiguous = ratio < self.ratio_threshold

    #     print(
    #         f"Best match distance (d1): {dist1:.4f}, Second best (d2): {dist2:.4f}, Ratio (d1/d2): {ratio:.4f}"
    #     )

    #     # 4. Final Decision
    #     if is_close_enough and is_unambiguous:
    #         print("Authentication successful: Match is both close and unambiguous.")
    #         # Find the original template to update its mutable attributes
    #         original_template = next(
    #             (t for t in self.enrolled_templates_data if t.id == best_match["id"]),
    #             None,
    #         )
    #         if original_template:
    #             original_template.match_count += 1
    #             original_template.last_matched_timestamp = time.time()

    #         self.successful_authentications += 1
    #         self.save_data()
    #         return "enrolled_person", dist1
    #     else:
    #         # Authentication failed. Determine why.
    #         reason = []
    #         if not is_close_enough:
    #             reason.append("match was not close enough")
    #         if not is_unambiguous:
    #             reason.append("match was ambiguous (too similar to second best)")
    #         print(f"Authentication failed: {', '.join(reason)}.")

    #         self.last_failed_attempt_features = features_to_test
    #         self.last_failed_attempt_distance = dist1
    #         self.save_data()

    #         # You can return more specific failure reasons if needed
    #         return "not_enrolled_person_ambiguous", dist1

    # def calculate_roc_and_find_threshold_template(
    #     self, positive_sample_paths, negative_sample_paths, plot_roc=True
    # ):
    #     """
    #     Performs ROC analysis using the minimum distance to any of the enrolled templates.
    #     """
    #     if not self.enrolled_templates_data:
    #         print("Error: Person not enrolled. Cannot perform ROC analysis.")
    #         return None, None, None
    #     positive_paths_person_A = []
    #     neagative_paths_person_A = []
    #     positive_path = "./positive/"
    #     negative_path = "./negative/"
    #     if positive_sample_paths is None or negative_sample_paths is None:

    #         if os.path.exists(positive_path):
    #             positive_paths_person_A = [
    #                 os.path.join(positive_path, img)
    #                 for img in os.listdir(positive_path)
    #                 if img.endswith(".jpg") or img.endswith(".png")
    #             ]

    #         if os.path.exists(negative_path):
    #             neagative_paths_person_A = [
    #                 os.path.join(negative_path, img)
    #                 for img in os.listdir(negative_path)
    #                 if img.endswith(".jpg") or img.endswith(".png")
    #             ]

    #         positive_sample_paths = positive_paths_person_A
    #         negative_sample_paths = neagative_paths_person_A

    #     y_true = []
    #     y_scores_dist = []

    #     print("ROC: Processing positive samples...")
    #     for img_path in positive_sample_paths:
    #         features = self._extract_features(img_path)
    #         if features is not None and features.size > 0:
    #             min_dist_to_enrolled_template = float("inf")
    #             for enrolled_template in self.enrolled_templates_data:
    #                 dist = chi_square_distance(
    #                     features, enrolled_template.template_vector
    #                 )
    #                 if dist < min_dist_to_enrolled_template:
    #                     min_dist_to_enrolled_template = dist

    #             if min_dist_to_enrolled_template != float("inf"):
    #                 y_true.append(1)
    #                 y_scores_dist.append(min_dist_to_enrolled_template)
    #         else:
    #             print(
    #                 f"Skipping positive ROC sample {os.path.basename(img_path)}: feature extraction issue."
    #             )

    #     print("ROC: Processing negative samples...")
    #     for img_path in negative_sample_paths:
    #         features = self._extract_features(img_path)
    #         if features is not None and features.size > 0:
    #             min_dist_to_enrolled_template = float("inf")
    #             for enrolled_template in self.enrolled_templates_data:
    #                 dist = chi_square_distance(
    #                     features, enrolled_template.template_vector
    #                 )
    #                 if dist < min_dist_to_enrolled_template:
    #                     min_dist_to_enrolled_template = dist

    #             if min_dist_to_enrolled_template != float("inf"):
    #                 y_true.append(0)
    #                 y_scores_dist.append(min_dist_to_enrolled_template)
    #         else:
    #             print(
    #                 f"Skipping negative ROC sample {os.path.basename(img_path)}: feature extraction issue."
    #             )

    #     if len(y_true) < 2 or len(np.unique(y_true)) < 2:
    #         print(
    #             "Error: Not enough data or not enough classes for ROC analysis after processing samples."
    #         )
    #         return None, None, None

    #     y_true_np = np.array(y_true)
    #     y_scores_dist_np = np.array(y_scores_dist)
    #     probas_for_roc = -y_scores_dist_np

    #     fpr, tpr, roc_thresholds_probas = roc_curve(
    #         y_true_np, probas_for_roc, pos_label=1
    #     )
    #     roc_auc_value = auc(fpr, tpr)

    #     optimal_idx = np.argmin(np.sqrt(fpr**2 + (1 - tpr) ** 2))
    #     optimal_distance_threshold = -roc_thresholds_probas[optimal_idx]

    #     print(
    #         f"Optimal distance threshold (closest to (0,1) on ROC): {optimal_distance_threshold:.4f}"
    #     )
    #     print(
    #         f"  Corresponding TPR: {tpr[optimal_idx]:.3f}, FPR: {fpr[optimal_idx]:.3f}"
    #     )
    #     print(f"  AUC: {roc_auc_value:.3f}")

    #     self.threshold = optimal_distance_threshold
    #     self.new_template_distance = optimal_distance_threshold + 2

    #     # Save data after threshold calculation
    #     self.save_data()

    #     return (
    #         optimal_distance_threshold,
    #         roc_auc_value,
    #         (fpr, tpr, -roc_thresholds_probas),
    #     )

    def calculate_roc_and_find_threshold_template(
        self, positive_sample_paths, negative_sample_paths, plot_roc=True
    ):
        """
        Performs ROC analysis using the minimum distance to any of the enrolled templates.
        Selects a secure threshold by constraining False Acceptance Rate (FAR) ≤ 1%.
        """
        if not self.enrolled_templates_data:
            print("Error: Person not enrolled. Cannot perform ROC analysis.")
            return None, None, None

        # Load from default folders if none provided
        if positive_sample_paths is None or negative_sample_paths is None:
            positive_path = "./positive/"
            negative_path = "./negative/"

            positive_sample_paths = (
                [
                    os.path.join(positive_path, img)
                    for img in os.listdir(positive_path)
                    if img.endswith(".jpg") or img.endswith(".png")
                ]
                if os.path.exists(positive_path)
                else []
            )

            negative_sample_paths = (
                [
                    os.path.join(negative_path, img)
                    for img in os.listdir(negative_path)
                    if img.endswith(".jpg") or img.endswith(".png")
                ]
                if os.path.exists(negative_path)
                else []
            )

        y_true = []
        y_scores_dist = []

        positive_distances = []
        negative_distances = []

        # Process positives
        print("ROC: Processing positive samples...")
        for img_path in positive_sample_paths:
            features = self._extract_features(img_path)
            if features is not None and features.size > 0:
                # dists = [
                #     chi_square_distance(features, template.template_vector)
                #     for template in self.enrolled_templates_data
                # ]
                distance = self.get_distance(features)
                y_true.append(1)
                y_scores_dist.append(distance)
                positive_distances.append(distance)
                print("Distance for postive images: ", y_scores_dist[-1])
            else:
                print(f"Skipping positive: {img_path}")

        # Process negatives
        print("ROC: Processing negative samples...")
        for img_path in negative_sample_paths:
            features = self._extract_features(img_path)
            if features is not None and features.size > 0:
                # dists = [
                #     chi_square_distance(features, template.template_vector)
                #     for template in self.enrolled_templates_data
                # ]
                distance = self.get_distance(features)
                y_true.append(0)
                y_scores_dist.append(distance)
                negative_distances.append(distance)
                print("Distance for negative images:", y_scores_dist[-1])
            else:
                print(f"Skipping negative: {img_path}")

        if len(set(y_true)) < 2:
            print("Not enough class diversity for ROC.")
            return None, None, None

        y_true_np = np.array(y_true)
        y_scores_dist_np = np.array(y_scores_dist)
        roc_scores = -y_scores_dist_np  # Negate for correct ROC orientation

        fpr, tpr, thresholds = roc_curve(y_true_np, roc_scores)
        roc_auc_value = auc(fpr, tpr)

        # Secure threshold: maximize TPR under FAR ≤ 1%
        desired_far = 0.001
        allowed_idxs = np.where(fpr <= desired_far)[0]

        if len(allowed_idxs) > 0:
            best_idx = allowed_idxs[-1]
            optimal_distance_threshold = -thresholds[best_idx]
            print(
                f"Threshold chosen for FAR ≤ {desired_far*100:.1f}%: {optimal_distance_threshold:.4f}"
            )
            print(f"  TPR: {tpr[best_idx]:.3f}, FPR: {fpr[best_idx]:.3f}")
        else:
            optimal_distance_threshold = -thresholds[0]  # most conservative
            print(
                "WARNING: No threshold found below desired FAR. Using strictest threshold."
            )

        # Optional clamp for additional safety
        optimal_distance_threshold = min(optimal_distance_threshold, 10.0)

        self.threshold = optimal_distance_threshold
        self.new_template_distance = optimal_distance_threshold + 1
        self.save_data()

        # Plot and save histograms for positive and negative distances
        import matplotlib.pyplot as plt

        os.makedirs("./roc_output", exist_ok=True)

        plt.figure()
        plt.hist(
            positive_distances,
            bins=30,
            alpha=0.7,
            label="Positive Distances",
            color="g",
        )
        plt.axvline(
            optimal_distance_threshold, color="k", linestyle="--", label="Threshold"
        )
        plt.xlabel("Chi-Square Distance")
        plt.ylabel("Frequency")
        plt.title("Histogram of Positive Distances")
        plt.legend()
        plt.tight_layout()
        plt.savefig("./roc_output/positive_distances_hist.png")
        plt.close()

        plt.figure()
        plt.hist(
            negative_distances,
            bins=30,
            alpha=0.7,
            label="Negative Distances",
            color="r",
        )
        plt.axvline(
            optimal_distance_threshold, color="k", linestyle="--", label="Threshold"
        )
        plt.xlabel("Chi-Square Distance")
        plt.ylabel("Frequency")
        plt.title("Histogram of Negative Distances")
        plt.legend()
        plt.tight_layout()
        plt.savefig("./roc_output/negative_distances_hist.png")
        plt.close()

        # Save ROC curve
        if plot_roc:
            import matplotlib.pyplot as plt

            output_path = "./roc_output/roc_curve.png"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc_value:.3f}")
            plt.axvline(
                desired_far,
                color="gray",
                linestyle="--",
                label=f"FAR ≤ {desired_far*100:.1f}%",
            )
            plt.axvline(
                fpr[best_idx] if len(allowed_idxs) > 0 else fpr[0],
                color="r",
                linestyle="--",
                label="Selected Threshold",
            )
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve (Chi-Square Distance)")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            print(f"ROC plot saved to: {output_path}")

        return self.threshold, roc_auc_value, (fpr, tpr, -thresholds)

    def adapt_model_with_feedback(self, user_confirmed_identity):
        """
        Adapts the model by updating the most relevant pose template using a moving average.
        """
        if not user_confirmed_identity or self.last_failed_attempt_features is None:
            self.last_failed_attempt_features = None
            self.last_failed_attempt_distance = None
            return False

        print("Adapting model with user feedback on a pose-based template...")
        confirmed_features = self.last_failed_attempt_features

        # Find the closest existing pose template
        closest_template = None
        min_dist_to_existing = float("inf")
        for template in self.enrolled_templates_data:
            dist = chi_square_distance(confirmed_features, template.template_vector)
            if dist < min_dist_to_existing:
                min_dist_to_existing = dist
                closest_template = template

        if closest_template is None:
            print("  Could not find a closest template to adapt. Aborting.")
            return False

        print(
            f"  Closest existing pose was '{closest_template.pose_label}'. Updating this template."
        )

        # Update the template using a weighted moving average
        current_mean = closest_template.template_vector
        n = closest_template.sample_count

        new_mean = (current_mean * n + confirmed_features) / (n + 1)

        # Update the template in place
        closest_template.template_vector = new_mean
        closest_template.sample_count += 1
        closest_template.last_matched_timestamp = time.time()

        print(
            f"  Template '{closest_template.pose_label}' updated. It is now based on {closest_template.sample_count} samples."
        )

        # Clean up and save
        self.last_failed_attempt_features = None
        self.last_failed_attempt_distance = None
        self.save_data()
        return True

    def adapt_model_updated(self, user_confirmed_identity):
        """Improved version with better ID generation"""
        if not user_confirmed_identity or self.last_failed_attempt_features is None:
            self.last_failed_attempt_features = None
            self.last_failed_attempt_distance = None
            return False

        print("Adapting model with user feedback on a pose-based template...")
        confirmed_features = self.last_failed_attempt_features
        if confirmed_features is None:
            print(f"Warning: Could not extract features for last failed attempt.")
            return False

        # Find closest template
        closest_template = None
        min_dist_to_existing = float("inf")
        for template in self.enrolled_templates_data:
            dist = chi_square_distance(confirmed_features, template.template_vector)
            if dist < min_dist_to_existing:
                min_dist_to_existing = dist
                closest_template = template

        if closest_template is None:
            print("Could not find a closest template to adapt.")
            return False

        print(
            f"Closest existing pose: '{closest_template.pose_label}' (distance: {min_dist_to_existing:.4f})"
        )

        if (
            min_dist_to_existing > self.new_template_distance
            and self.max_total_templates >= len(self.enrolled_templates_data)
        ):
            print(
                f"Adding new template because distance {min_dist_to_existing:.4f} is greater than threshold {self.new_template_distance:.4f}."
            )
            # Generate unique ID
            existing_ids = {t.id for t in self.enrolled_templates_data}
            counter = 1
            while f"extra_{counter}" in existing_ids:
                counter += 1

            new_id = f"extra_{counter}"
            template_entry = TemplateEntry(
                id=new_id,
                template_vector=confirmed_features,
                pose_label="extra",
                source="adapted_positive",
                initial_sample_count=1,
            )
            self.enrolled_templates_data.append(template_entry)
            print(
                f"Added new template '{new_id}'. Total templates: {len(self.enrolled_templates_data)}"
            )
        else:
            # Update existing template
            current_mean = closest_template.template_vector
            n = closest_template.sample_count
            new_mean = (current_mean * n + confirmed_features) / (n + 1)

            closest_template.template_vector = new_mean
            closest_template.sample_count += 1
            closest_template.last_matched_timestamp = time.time()
            print(
                f"Updated template '{closest_template.pose_label}' (now based on {closest_template.sample_count} samples)"
            )

        # Save after adaptation
        self.save_data()
        return True

    def adapt_model_mannual(self, image_path):
        """Improved version with better ID generation"""
        confirmed_features = self._extract_features(image_path)
        if confirmed_features is None:
            print(f"Warning: Could not extract features for {image_path}.")
            return False

        # Find closest template
        closest_template = None
        min_dist_to_existing = float("inf")
        for template in self.enrolled_templates_data:
            dist = chi_square_distance(confirmed_features, template.template_vector)
            if dist < min_dist_to_existing:
                min_dist_to_existing = dist
                closest_template = template

        if closest_template is None:
            print("Could not find a closest template to adapt.")
            return False

        print(
            f"Closest existing pose: '{closest_template.pose_label}' (distance: {min_dist_to_existing:.4f})"
        )

        if (
            min_dist_to_existing > self.new_template_distance
            and self.max_total_templates >= len(self.enrolled_templates_data)
        ):
            print(
                f"Adding new template because distance {min_dist_to_existing:.4f} is greater than threshold {self.new_template_distance:.4f}."
            )
            # Generate unique ID
            existing_ids = {t.id for t in self.enrolled_templates_data}
            counter = 1
            while f"extra_{counter}" in existing_ids:
                counter += 1

            new_id = f"extra_{counter}"
            template_entry = TemplateEntry(
                id=new_id,
                template_vector=confirmed_features,
                pose_label="extra",
                source="adapted_positive",
                initial_sample_count=1,
            )
            self.enrolled_templates_data.append(template_entry)
            print(
                f"Added new template '{new_id}'. Total templates: {len(self.enrolled_templates_data)}"
            )
        else:
            # Update existing template
            current_mean = closest_template.template_vector
            n = closest_template.sample_count
            new_mean = (current_mean * n + confirmed_features) / (n + 1)

            closest_template.template_vector = new_mean
            closest_template.sample_count += 1
            closest_template.last_matched_timestamp = time.time()
            print(
                f"Updated template '{closest_template.pose_label}' (now based on {closest_template.sample_count} samples)"
            )

        # Save after adaptation
        self.save_data()
        return True
