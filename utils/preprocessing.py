"""
utils/preprocessing.py
Feature engineering, label generation, normalization, and train/val/test splitting.
"""
import logging
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

logger = logging.getLogger(__name__)


KELVIN_OFFSET = 273.15


class HeatwavePreprocessor:
    """Transforms raw ERA5 DataFrame into ML-ready X, y arrays."""

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.threshold_c = self.cfg["data"]["heatwave_threshold_celsius"]
        self.features = self.cfg["data"]["features"]
        self.label_col = self.cfg["data"]["label_col"]
        self.split_cfg = self.cfg["split"]
        self.scaler = StandardScaler()

    # ------------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame):
        """
        Apply full preprocessing pipeline.
        Returns (X_train, X_val, X_test, y_train, y_val, y_test, feature_names)
        """
        df = self._engineer_features(df)
        df = self._generate_labels(df)
        df = self._drop_na(df)

        feature_names = self._get_feature_names(df)
        X = df[feature_names].values
        y = df[self.label_col].values

        logger.info("Label distribution: heatwave=%.2f%%", 100 * y.mean())
        logger.info("Total samples: %d | Features: %d", len(X), len(feature_names))

        # Split
        stratify = y if self.split_cfg.get("stratify", True) else None
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y,
            test_size=self.split_cfg["test"],
            random_state=self.split_cfg["random_state"],
            stratify=stratify,
        )
        val_ratio = self.split_cfg["val"] / (1 - self.split_cfg["test"])
        stratify_train = y_train_full if self.split_cfg.get("stratify", True) else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=val_ratio,
            random_state=self.split_cfg["random_state"],
            stratify=stratify_train,
        )

        # Scale (fit only on train)
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        logger.info("Split — Train: %d | Val: %d | Test: %d", len(X_train), len(X_val), len(X_test))
        return X_train, X_val, X_test, y_train, y_val, y_test, feature_names

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply preprocessing to new data using already-fitted scaler."""
        df = self._engineer_features(df)
        feature_names = self._get_feature_names(df)
        X = df[feature_names].fillna(0).values
        return self.scaler.transform(X)

    def save_scaler(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.scaler, path)
        logger.info("Scaler saved to %s", path)

    def load_scaler(self, path: str):
        self.scaler = joblib.load(path)
        logger.info("Scaler loaded from %s", path)

    # ------------------------------------------------------------------
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Convert temperatures from Kelvin to Celsius where applicable
        temp_vars = ["t2m", "d2m"]
        for col in temp_vars:
            if col in df.columns:
                # Only convert if values look like Kelvin (> 200)
                if df[col].median() > 200:
                    df[f"{col}_c"] = df[col] - KELVIN_OFFSET
                else:
                    df[f"{col}_c"] = df[col]

        # Heat index approximation (Steadman simplified)
        if "t2m_c" in df.columns and "d2m_c" in df.columns:
            T = df["t2m_c"]
            Td = df["d2m_c"]
            # Relative humidity approximation
            RH = 100 - 5 * (T - Td)
            RH = RH.clip(0, 100)
            df["heat_index"] = T + 0.33 * (RH / 100 * 6.105 * np.exp(17.27 * T / (237.7 + T))) - 4.0

        # Wind speed magnitude
        if "u10" in df.columns and "v10" in df.columns:
            df["wind_speed"] = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2)

        return df

    def _generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "t2m_c" in df.columns:
            df[self.label_col] = (df["t2m_c"] >= self.threshold_c).astype(int)
        elif "t2m" in df.columns:
            df[self.label_col] = (df["t2m"] - KELVIN_OFFSET >= self.threshold_c).astype(int)
        else:
            raise ValueError("Temperature variable 't2m' not found in DataFrame")
        return df

    def _drop_na(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.dropna(subset=[self.label_col])
        dropped = before - len(df)
        if dropped:
            logger.debug("Dropped %d rows with NaN labels", dropped)
        return df

    def _get_feature_names(self, df: pd.DataFrame) -> list:
        engineered = ["t2m_c", "d2m_c", "heat_index", "wind_speed"]
        base = ["sp"]
        candidates = engineered + base
        return [c for c in candidates if c in df.columns]


# ------------------------------------------------------------------
def preprocess(df: pd.DataFrame, config_path: str = "config/config.yaml"):
    """Convenience function used by the pipeline."""
    preprocessor = HeatwavePreprocessor(config_path)
    return preprocessor.fit_transform(df)
