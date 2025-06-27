import socket
import subprocess
import platform
import datetime
import random
import hashlib
from concurrent.futures import ThreadPoolExecutor
import threading
import csv
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
import joblib


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
        if platform.system() == "Windows":
            wmi_result = self._get_brightness_wmi()
            if wmi_result != "Unable to retrieve":
                return wmi_result
        try:
            import screen_brightness_control as sbc

            if platform.system() == "Windows":
                import warnings

                warnings.filterwarnings("ignore", category=UserWarning)
            brightness_list = sbc.get_brightness()
            result = brightness_list[0] if brightness_list else "Unable to retrieve"
            import gc

            gc.collect()
            return result
        except ImportError:
            return "screen-brightness-control not installed"
        except Exception as e:
            return "Unable to retrieve"

    def get_system_info(self):
        """Get system information using parallel execution for I/O operations"""
        with ThreadPoolExecutor(max_workers=3) as executor:
            ip_future = executor.submit(
                lambda: self._get_cached_or_compute("ip_address", self._get_ip_address)
            )
            wifi_future = executor.submit(
                lambda: self._get_cached_or_compute("wifi_ssid", self._get_wifi_ssid)
            )
            brightness_future = executor.submit(
                lambda: self._get_cached_or_compute("brightness", self._get_brightness)
            )
            return {
                "ip_address": ip_future.result(),
                "wifi_ssid": wifi_future.result(),
                "brightness": brightness_future.result(),
            }

    def extract_features(self):
        """Extract all features efficiently"""
        system_info = self.get_system_info()
        with ThreadPoolExecutor(max_workers=2) as executor:
            internet_future = executor.submit(self.get_internet_status)
            return {
                "hour": self.get_hour_of_day(),
                "day_of_week": self.get_day_of_week(),
                "brightness": system_info["brightness"],
                "orientation": self.get_orientation(),
                "internet_connected": internet_future.result(),
                "wifi_hash": self.hash_string(system_info["wifi_ssid"]),
                "ip_hash": self.hash_string(system_info["ip_address"]),
            }


class BehaviorModel:
    def __init__(
        self,
        log_file="./data/behavioral_logs.csv",
        model_path="./data/behavior_model.joblib",
        retrain_threshold=20,
    ):
        self.log_file = log_file
        self.model_path = model_path
        self.encoders_path = model_path.replace(".joblib", "_encoders.joblib")
        self.model = CategoricalNB()
        self.label_encoders = {}
        self.retrain_threshold = retrain_threshold
        self.is_trained = False
        self.feature_extractor = OptimizedSystemInfo()

        # Load the model if it exists
        self.load_model()

    def load_model(self):
        """Loads a trained model and encoders from disk."""
        if os.path.exists(self.model_path) and os.path.exists(self.encoders_path):
            try:
                self.model = joblib.load(self.model_path)
                self.label_encoders = joblib.load(self.encoders_path)
                self.is_trained = True
                print("Successfully loaded pre-trained model.")
                return True
            except Exception as e:
                print(f"Error loading model: {e}. A new model will be created.")
                self.is_trained = False
                return False
        else:
            print(
                "No pre-trained model found. A new model will be created after training."
            )
            self.is_trained = False
            return False

    def save_model(self):
        """Saves the trained model and encoders to disk."""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.label_encoders, self.encoders_path)
            print(f"Model saved successfully to {self.model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_data(self):
        if not os.path.exists(self.log_file):
            return pd.DataFrame()
        return pd.read_csv(self.log_file)

    def fit(self):
        df = self.load_data()
        if df.empty or len(df) < 10:
            print("Not enough data to train model. Need at least 10 samples.")
            return False

        X = df.drop(columns=["unlock_type_used"])
        y = df["unlock_type_used"]

        self.label_encoders = {}
        for col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le

        self.label_encoders["target"] = LabelEncoder()
        y = self.label_encoders["target"].fit_transform(y)

        self.model.fit(X, y)
        self.is_trained = True
        print(f"Model trained with {len(df)} samples.")

        # Save the newly trained model
        self.save_model()
        return True

    def predict(self, features=None):
        if features is None:
            features = self.feature_extractor.extract_features()

        if not self.is_trained:
            print("Model not trained yet. Using default prediction.")
            return "FACE"  # Default fallback

        if not self.label_encoders:
            print("Label encoders not initialized. Using default prediction.")
            return "FACE"

        df = pd.DataFrame([features])

        # Ensure all columns from training are present
        for col, le in self.label_encoders.items():
            if col != "target" and col not in df.columns:
                # If a feature is missing during prediction, we can't proceed.
                # A more robust solution might impute a value, but for now we fallback.
                print(f"Missing feature '{col}' during prediction. Using fallback.")
                return "FACE"

        for col in df.columns:
            le = self.label_encoders.get(col)
            if le:
                # Handle unseen values by assigning a known category (e.g., the first one)
                df[col] = (
                    df[col]
                    .astype(str)
                    .apply(lambda x: x if x in le.classes_ else le.classes_[0])
                )
                df[col] = le.transform(df[col])

        try:
            y_pred = self.model.predict(df)
            return self.label_encoders["target"].inverse_transform(y_pred)[0]
        except Exception as e:
            print(f"Prediction error: {e}. Using fallback.")
            return "FACE"

    def log_event(self, features=None, unlock_type_used=None):
        if features is None:
            features = self.feature_extractor.extract_features()

        if unlock_type_used is None:
            unlock_type_used = random.choice(["PIN", "FACE"])

        features_copy = features.copy()
        features_copy["unlock_type_used"] = unlock_type_used

        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        file_exists = os.path.exists(self.log_file)

        with open(self.log_file, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=features_copy.keys())
            if not file_exists or f.tell() == 0:
                writer.writeheader()
            writer.writerow(features_copy)

        print(f"Logged event: {unlock_type_used}")
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
        df = self.load_data()
        return len(df)

    def simulate_user_behavior(self, num_events=30):
        """Simulate user behavior for testing"""
        unlock_methods = ["PIN", "FACE"]
        print("Simulating user behavior...")
        for i in range(num_events):
            features = self.feature_extractor.extract_features()

            # Simple simulation logic
            if isinstance(features["brightness"], int) and features["brightness"] < 30:
                unlock_method = "PIN"
            else:
                unlock_method = "FACE"

            self.log_event(features, unlock_method)
            if (i + 1) % 10 == 0:
                print(f"Logged {i+1} events...")
        print(f"Simulation complete. {num_events} events logged.")


class BehavioralAuthenticationSystem:
    def __init__(
        self,
        log_file="./data/behavioral_logs.csv",
        model_path="./data/behavior_model.joblib",
    ):
        self.behavior_model = BehaviorModel(log_file=log_file, model_path=model_path)
        self.feature_extractor = OptimizedSystemInfo()

    def get_recommendation(self):
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
        features = self.feature_extractor.extract_features()
        self.behavior_model.log_event(features, method_used)

    def train_model(self):
        return self.behavior_model.fit()

    def get_data_count(self):
        return self.behavior_model.get_log_count()

    def get_system_stats(self):
        return {
            "model_stats": self.behavior_model.get_model_stats(),
            "current_features": self.feature_extractor.extract_features(),
        }
