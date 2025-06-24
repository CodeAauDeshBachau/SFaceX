import numpy as np
import cv2
import os
import time

# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# import seaborn as sns
from numba import jit, prange
import joblib
import warnings

warnings.filterwarnings("ignore")


# Import optimized LBP functions from previous code
@jit(nopython=True)
def uniform_lbp_mapping():
    """Generate uniform LBP mapping table using numba for speed"""
    table = np.zeros(256, dtype=np.uint8)
    index = 0

    for i in range(256):
        binary = np.zeros(8, dtype=np.uint8)
        temp = i
        for j in range(8):
            binary[j] = temp & 1
            temp >>= 1

        transitions = 0
        for j in range(8):
            if binary[j] != binary[(j + 1) % 8]:
                transitions += 1

        if transitions <= 2:
            table[i] = index
            index += 1
        else:
            table[i] = 58

    return table


@jit(nopython=True)
def lbp_image_optimized(gray, mapping):
    """Optimized LBP computation using numba"""
    h, w = gray.shape
    lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)

    offsets = np.array(
        [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)],
        dtype=np.int32,
    )

    for i in prange(1, h - 1):
        for j in range(1, w - 1):
            center = gray[i, j]
            code = 0

            for k in range(8):
                dx, dy = offsets[k]
                if gray[i + dx, j + dy] >= center:
                    code |= 1 << k

            lbp[i - 1, j - 1] = mapping[code]

    return lbp


@jit(nopython=True)
def compute_grid_histograms_optimized(lbp_img, grid_x=8, grid_y=8, bins=59):
    """Optimized grid division and histogram computation"""
    h, w = lbp_img.shape
    grid_h, grid_w = h // grid_y, w // grid_x

    histograms = np.zeros((grid_x * grid_y, bins), dtype=np.float32)

    grid_idx = 0
    for i in range(grid_y):
        for j in range(grid_x):
            start_h, end_h = i * grid_h, (i + 1) * grid_h
            start_w, end_w = j * grid_w, (j + 1) * grid_w

            for y in range(start_h, end_h):
                for x in range(start_w, end_w):
                    pixel_val = lbp_img[y, x]
                    if pixel_val < bins:
                        histograms[grid_idx, pixel_val] += 1

            norm = 0.0
            for k in range(bins):
                norm += histograms[grid_idx, k] ** 2
            norm = np.sqrt(norm) + 1e-6

            for k in range(bins):
                histograms[grid_idx, k] /= norm

            grid_idx += 1

    return histograms


class LBPFeatureExtractor:
    """Optimized LBP feature extractor for emotion recognition"""

    def __init__(self, grid_x=8, grid_y=8, target_size=(64, 64)):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.target_size = target_size
        self.mapping = uniform_lbp_mapping()
        self.n_features = grid_x * grid_y * 59
        print(f"Feature extractor initialized: {self.n_features} features per image")

    def preprocess_image(self, image_path):
        """Enhanced preprocessing pipeline"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return None

            # Resize with better interpolation
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Face region enhancement
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

            # Illumination normalization using gamma correction
            gamma = 0.7
            inv_gamma = 1.0 / gamma
            table = np.array(
                [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
            ).astype("uint8")
            gray = cv2.LUT(gray, table)

            # Gentle noise reduction while preserving edges
            gray = cv2.bilateralFilter(gray, 5, 50, 50)

            return gray

        except Exception as e:
            print(f"Error preprocessing {image_path}: {e}")
            return None

    def preprocessing_image_optimized(self, image_path):
        """Streamlined preprocessing with fewer operations"""
        # Load and resize image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image not found or path is incorrect.")

        # Resize first to reduce computation
        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Single-step preprocessing: CLAHE only (most effective for LBP)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        return enhanced

    def extract_single_feature(self, image_path):
        """Extract LBP features from single image"""
        # gray = self.preprocess_image(image_path)
        gray = self.preprocessing_image_optimized(image_path)
        if gray is None:
            return np.zeros(self.n_features, dtype=np.float32)

        # Compute LBP
        lbp = lbp_image_optimized(gray, self.mapping)

        # Compute grid histograms
        histograms = compute_grid_histograms_optimized(
            lbp, self.grid_x, self.grid_y, 59
        )

        # Flatten and normalize
        feature_vector = histograms.flatten()
        feature_vector = feature_vector / (np.linalg.norm(feature_vector) + 1e-6)

        return feature_vector

    def extract_batch_features(self, image_paths, batch_size=100):
        """Extract features from multiple images with progress tracking"""
        n_images = len(image_paths)
        features = []
        valid_indices = []  # Track which images were successfully processed

        print(f"Extracting features from {n_images} images...")
        start_time = time.time()

        for i in range(0, n_images, batch_size):
            batch_end = min(i + batch_size, n_images)

            for j in range(i, batch_end):
                feature = self.extract_single_feature(image_paths[j])
                # IMPROVEMENT: Only keep valid features
                if not np.all(feature == 0):  # Check if feature extraction succeeded
                    features.append(feature)
                    valid_indices.append(j)

            # Progress update
            progress = batch_end / n_images * 100
            elapsed = time.time() - start_time
            rate = batch_end / elapsed
            remaining = (n_images - batch_end) / rate if rate > 0 else 0

            print(
                f"Progress: {progress:.1f}% ({batch_end}/{n_images}) - "
                f"{rate:.1f} img/sec - ETA: {remaining:.1f}s"
            )

        total_time = time.time() - start_time
        print(f"Feature extraction completed in {total_time:.2f} seconds")
        print(f"Successfully processed {len(features)} out of {n_images} images")

        return np.array(features, dtype=np.float32), valid_indices


feature_extractor = LBPFeatureExtractor(grid_x=8, grid_y=8, target_size=(64, 64))
feature = feature_extractor.extract_single_feature(
    "/content/data/captured_face_20250608-213444.jpg"
)

print("\nLoading model for a test prediction...")
loaded_model = joblib.load("face_mask_svm_model.joblib")
test_prediction = loaded_model.predict([feature])
print(
    f"Prediction for a sample test image: {'Without Mask' if test_prediction[0] else 'With Mask'}"
)
