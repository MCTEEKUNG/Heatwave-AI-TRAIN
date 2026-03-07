"""
utils/data_loader.py
Loads ERA5 NetCDF surface files and returns a flat pandas DataFrame.
"""
import os
import glob
import yaml
import logging
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ERA5DataLoader:
    """Loads ERA5 surface NetCDF files and converts them to a DataFrame."""

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.raw_dir = self.cfg["data"]["raw_dir"]
        self.years = self.cfg["data"]["years"]
        self.surface_prefix = self.cfg["data"]["surface_prefix"]
        self.features = self.cfg["data"]["features"]
        self.threshold_c = self.cfg["data"]["heatwave_threshold_celsius"]

    # ------------------------------------------------------------------
    def load(self) -> pd.DataFrame:
        """Load all surface files matching configured years, return DataFrame."""
        all_frames = []

        for year in tqdm(self.years, desc="Loading ERA5 years"):
            fname = os.path.join(self.raw_dir, f"{self.surface_prefix}{year}.nc")
            if not os.path.isfile(fname):
                logger.warning("File not found, skipping: %s", fname)
                continue

            try:
                ds = xr.open_dataset(fname, engine="netcdf4")
                df = self._dataset_to_df(ds, year)
                all_frames.append(df)
                ds.close()
            except Exception as e:
                logger.error("Error reading %s: %s", fname, e)

        if not all_frames:
            raise RuntimeError("No ERA5 data files could be loaded. Check raw_dir in config.")

        full_df = pd.concat(all_frames, ignore_index=True)
        logger.info("Loaded %d total samples across %d years", len(full_df), len(self.years))
        return full_df

    # ------------------------------------------------------------------
    def _dataset_to_df(self, ds: xr.Dataset, year: int) -> pd.DataFrame:
        """
        Convert an xarray Dataset to a flat DataFrame row per (time, lat, lon).
        Includes time, latitude, and longitude coordinates.
        """
        # Select only the variables we want + the coordinates
        available_vars = list(ds.data_vars)
        load_vars = [v for v in self.features if v in available_vars]
        
        # Convert to dataframe
        # This will result in a MultiIndex (time, latitude/lat, longitude/lon)
        df = ds[load_vars].to_dataframe().reset_index()
        
        # Standardize coordinate names
        rename_map = {
            "valid_time": "time",
            "lat": "latitude", "lon": "longitude",
            "y": "latitude", "x": "longitude"
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # Drop unnecessary ERA5 metadata columns if they exist
        drop_cols = ["number", "expver"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

        # Add year column for tracking
        df["year"] = year

        # Ensure all configured features exist (even if NaN)
        for var in self.features:
            if var not in df.columns:
                df[var] = np.nan

        return df


# ------------------------------------------------------------------
def load_data(config_path: str = "config/config.yaml") -> pd.DataFrame:
    """Convenience function used by the pipeline."""
    loader = ERA5DataLoader(config_path)
    return loader.load()


if __name__ == "__main__":
    df = load_data()
    print(f"Shape: {df.shape}")
    print(df.head())
