import socket
import subprocess
import platform
import datetime
import random
import hashlib
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import threading
import csv
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os


class OptimizedSystemInfo:
    def __init__(self):
        self._cache = {}
        self._cache_timeout = 30  # seconds
        self._cache_lock = threading.Lock()

    def _is_cache_valid(self, key):
        """Check if cached value is still valid"""
        if key not in self._cache:
            return False
        return (
            datetime.datetime.now() - self._cache[key]["timestamp"]
        ).seconds < self._cache_timeout

    def _get_cached_or_compute(self, key, compute_func):
        """Get cached value or compute new one"""
        with self._cache_lock:
            if self._is_cache_valid(key):
                return self._cache[key]["value"]

            value = compute_func()
            self._cache[key] = {"value": value, "timestamp": datetime.datetime.now()}
            return value

    @staticmethod
    def get_hour_of_day():
        """Get current hour - no caching needed for time"""
        return datetime.datetime.now().hour

    @staticmethod
    def get_day_of_week():
        """Get current day - no caching needed for time"""
        mapping = {
            "Monday": 0,
            "Tuesday": 1,
            "Wednesday": 2,
            "Thursday": 3,
            "Friday": 4,
            "Saturday": 5,
            "Sunday": 6,
        }
        return mapping[datetime.datetime.now().strftime("%A")]

    @staticmethod
    def get_orientation():
        """Simulate device orientation"""
        return random.choice(["flat", "vertical"])

    def get_internet_status(self):
        """Check internet connectivity with caching"""

        def _check_internet():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    s.settimeout(0.5)  # Reduced timeout for faster execution
                    s.connect(("8.8.8.8", 53))
                    return True
            except:
                return False

        return self._get_cached_or_compute("internet_status", _check_internet)

    # @staticmethod
    # def recent_mask_usage():
    #     """Simulate recent mask usage"""
    #     return random.choice([True, False])

    @staticmethod
    def hash_string(s):
        """Hash string and return first 8 characters"""
        if not s or s == "Unable to retrieve":
            return "unknown"
        return hashlib.sha256(str(s).encode()).hexdigest()[:8]

    def _get_ip_address(self):
        """Get local IP address"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.settimeout(0.5)
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except:
            return "Unable to retrieve"

    def _get_wifi_ssid(self):
        """Get WiFi SSID (Windows only)"""
        if platform.system() != "Windows":
            return "Not Windows"

        try:
            result = subprocess.run(
                ["netsh", "wlan", "show", "interfaces"],
                capture_output=True,
                text=True,
                timeout=3,  # Add timeout for subprocess
                creationflags=subprocess.CREATE_NO_WINDOW,  # Hide console window
            )

            for line in result.stdout.split("\n"):
                if "SSID" in line and "BSSID" not in line:
                    ssid = line.split(":", 1)[1].strip()
                    return ssid if ssid else "Unable to retrieve"

            return "Unable to retrieve"
        except (subprocess.TimeoutExpired, Exception):
            return "Unable to retrieve"

    def _get_brightness_wmi(self):
        """Alternative brightness method using WMI (Windows only)"""
        if platform.system() != "Windows":
            return "Not Windows"

        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-Command",
                    "Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightness | Select-Object -ExpandProperty CurrentBrightness",
                ],
                capture_output=True,
                text=True,
                timeout=2,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )

            if result.returncode == 0 and result.stdout.strip():
                return int(result.stdout.strip())
            else:
                return "Unable to retrieve"

        except Exception:
            return "Unable to retrieve"

    def _get_brightness(self):
        """Get screen brightness with fallback methods"""
        # Try WMI method first (more reliable on Windows)
        if platform.system() == "Windows":
            wmi_result = self._get_brightness_wmi()
            if wmi_result != "Unable to retrieve":
                return wmi_result

        # Fallback to screen-brightness-control
        try:
            import screen_brightness_control as sbc

            # Suppress Win32 COM warnings
            if platform.system() == "Windows":
                import warnings

                warnings.filterwarnings("ignore", category=UserWarning)

            brightness_list = sbc.get_brightness()
            result = brightness_list[0] if brightness_list else "Unable to retrieve"

            # Force garbage collection to help with COM cleanup
            import gc

            gc.collect()

            return result

        except ImportError:
            return "screen-brightness-control not installed"
        except Exception as e:
            return "Unable to retrieve"

    def get_system_info(self):
        """Get system information using parallel execution for I/O operations"""

        # Use ThreadPoolExecutor for I/O bound operations
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all I/O operations concurrently
            ip_future = executor.submit(
                lambda: self._get_cached_or_compute("ip_address", self._get_ip_address)
            )
            wifi_future = executor.submit(
                lambda: self._get_cached_or_compute("wifi_ssid", self._get_wifi_ssid)
            )
            brightness_future = executor.submit(
                lambda: self._get_cached_or_compute("brightness", self._get_brightness)
            )

            # Collect results
            return {
                "ip_address": ip_future.result(),
                "wifi_ssid": wifi_future.result(),
                "brightness": brightness_future.result(),
            }

    def extract_features(self):
        """Extract all features efficiently"""
        # Get system info once
        system_info = self.get_system_info()

        # Use parallel execution for independent operations
        with ThreadPoolExecutor(max_workers=2) as executor:
            internet_future = executor.submit(self.get_internet_status)
            # mask_future = executor.submit(self.recent_mask_usage)

            return {
                "hour": self.get_hour_of_day(),
                "day_of_week": self.get_day_of_week(),
                "brightness": system_info["brightness"],
                "orientation": self.get_orientation(),
                "internet_connected": internet_future.result(),
                "wifi_hash": self.hash_string(system_info["wifi_ssid"]),
                "ip_hash": self.hash_string(system_info["ip_address"]),
                # "recent_mask_use": mask_future.result(),
            }


class BehaviorModel:
    def __init__(self, log_file="./data/behavioral_logs.csv", retrain_threshold=20):
        self.log_file = log_file
        self.model = CategoricalNB()
        self.label_encoders = {}
        self.retrain_threshold = retrain_threshold
        self.is_trained = False
        self.feature_extractor = OptimizedSystemInfo()

    def load_data(self):
        if not os.path.exists(self.log_file):
            return pd.DataFrame()
        return pd.read_csv(self.log_file)

    def fit(self):
        df = self.load_data()
        if df.empty or len(df) < 10:  # Need at least 10 samples for training
            print("Not enough data to train model. Need at least 10 samples.")
            return False

        X = df.drop(columns=["unlock_type_used"])
        y = df["unlock_type_used"]

        # Reset label encoders
        self.label_encoders = {}

        # Encode features
        for col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le

        # Encode target
        self.label_encoders["target"] = LabelEncoder()
        y = self.label_encoders["target"].fit_transform(y)

        self.model.fit(X, y)
        self.is_trained = True
        print(f"Model trained with {len(df)} samples.")
        return True

    def predict(self, features=None):
        if features is None:
            features = self.feature_extractor.extract_features()

        if not self.is_trained or not self.label_encoders:
            print("Model not trained yet. Using default prediction.")
            return "FACE"  # Default fallback

        df = pd.DataFrame([features])

        # Handle unseen categorical values
        for col in df.columns:
            le = self.label_encoders.get(col)
            if le:
                try:
                    df[col] = le.transform(df[col].astype(str))
                except ValueError as e:
                    # Handle unseen values by using the most frequent class
                    print(f"Unseen value in {col}: {df[col].iloc[0]}. Using fallback.")
                    df[col] = 0  # Use first class as fallback

        try:
            y_pred = self.model.predict(df)
            return self.label_encoders["target"].inverse_transform(y_pred)[0]
        except Exception as e:
            print(f"Prediction error: {e}. Using fallback.")
            return "FACE"  # Default fallback

    def log_event(self, features=None, unlock_type_used=None):
        if features is None:
            features = self.feature_extractor.extract_features()

        if unlock_type_used is None:
            # For demonstration, randomly choose
            unlock_type_used = random.choice(["PIN", "FACE"])

        features_copy = features.copy()
        features_copy["unlock_type_used"] = unlock_type_used

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        # Check if file exists and has header
        file_exists = os.path.exists(self.log_file)

        with open(self.log_file, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=features_copy.keys())
            if not file_exists or f.tell() == 0:
                writer.writeheader()
            writer.writerow(features_copy)

        print(f"Logged event: {unlock_type_used}")

        # Auto retrain when enough data points are collected
        df = self.load_data()
        if len(df) >= 10 and len(df) % self.retrain_threshold == 0:
            print("Retraining model...")
            self.fit()

    def get_model_stats(self):
        df = self.load_data()
        if df.empty:
            return "No data available"

        stats = {
            "total_samples": len(df),
            "unlock_methods": df["unlock_type_used"].value_counts().to_dict(),
            "is_trained": self.is_trained,
        }
        return stats

    def get_log_count(self):
        """Get the number of logged events"""
        df = self.load_data()
        return len(df)

    def simulate_user_behavior(self, num_events=30):
        """Simulate user behavior for testing"""
        unlock_methods = ["PIN", "FACE"]

        print("Simulating user behavior...")
        for i in range(num_events):
            features = self.feature_extractor.extract_features()

            # Simulate context-aware behavior
            if features["recent_mask_use"]:
                # More likely to use PIN when wearing mask
                unlock_method = random.choices(unlock_methods, weights=[0.7, 0.3])[0]
            elif features["brightness"] == "Unable to retrieve" or (
                isinstance(features["brightness"], int) and features["brightness"] < 30
            ):
                # More likely to use PIN in low light
                unlock_method = random.choices(unlock_methods, weights=[0.7, 0.3])[0]
            else:
                # Normal distribution
                unlock_method = random.choice(unlock_methods)

            self.log_event(features, unlock_method)

            if i % 10 == 0:
                print(f"Logged {i+1} events...")

        print(f"Simulation complete. {num_events} events logged.")


class BehavioralAuthenticationSystem:
    """Main system that combines feature extraction and behavior modeling"""

    def __init__(self, log_file="./data/behavioral_logs.csv"):
        self.behavior_model = BehaviorModel(log_file)
        self.feature_extractor = OptimizedSystemInfo()

    def get_recommendation(self):
        """Get unlock method recommendation based on current context"""
        features = self.feature_extractor.extract_features()
        prediction = self.behavior_model.predict(features)

        return {
            "current_features": features,
            "recommended_method": prediction,
            "confidence": (
                "Model-based" if self.behavior_model.is_trained else "Default"
            ),
        }

    def record_unlock_attempt(self, method_used):
        """Record an unlock attempt for learning"""
        features = self.feature_extractor.extract_features()
        self.behavior_model.log_event(features, method_used)

    def train_model(self):
        """Train the behavior model"""
        return self.behavior_model.fit()

    def get_data_count(self):
        """Get the number of logged events"""
        return self.behavior_model.get_log_count()

    def get_system_stats(self):
        """Get system statistics"""
        return {
            "model_stats": self.behavior_model.get_model_stats(),
            "current_features": self.feature_extractor.extract_features(),
        }


# === Example Usage ===
# def main():
#     """Main function demonstrating the merged system"""
#     print("=== Behavioral Authentication System ===\n")

#     # Initialize the system
#     auth_system = BehavioralAuthenticationSystem()

#     # Generate initial training data
#     print("1. Generating initial training data...")
#     auth_system.behavior_model.simulate_user_behavior(25)

#     # Train the model
#     print("\n2. Training the behavior model...")
#     auth_system.train_model()

#     # Show current system info
#     print("\n3. Current System Information:")
#     system_info = auth_system.feature_extractor.get_system_info()
#     for key, value in system_info.items():
#         print(f"   {key}: {value}")

#     # Make recommendations
#     print("\n4. Getting unlock recommendations...")
#     for i in range(5):
#         recommendation = auth_system.get_recommendation()
#         print(f"\nRecommendation {i+1}:")
#         print(f"   Recommended method: {recommendation['recommended_method']}")
#         print(f"   Confidence: {recommendation['confidence']}")
#         print(
#             f"   Context: Hour={recommendation['current_features']['hour']}, "
#             f"Day={recommendation['current_features']['day_of_week']}, "
#             f"Mask={recommendation['current_features']['recent_mask_use']}"
#         )

#         # Simulate user choice and record it
#         simulated_choice = random.choice(["PIN", "FACE"])
#         auth_system.record_unlock_attempt(simulated_choice)
#         print(f"   User chose: {simulated_choice}")

#     # Show final statistics
#     print("\n5. System Statistics:")
#     stats = auth_system.get_system_stats()
#     print(f"   Total samples: {stats['model_stats']['total_samples']}")
#     print(f"   Unlock methods distribution: {stats['model_stats']['unlock_methods']}")
#     print(f"   Model trained: {stats['model_stats']['is_trained']}")


# if __name__ == "__main__":
#     main()
# this is the final code till now date: 25-6-17
