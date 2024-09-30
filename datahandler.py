import xarray as xr
import zarr
import dask
import dask.array as da
import numpy as np
import json
import random
import datetime
import pandas as pd
import pyproj
from pathlib import Path
from typing import Union
from dask.array.core import map_blocks
import regionmask
import time
import sys
import os
from global_land_mask import globe
from abc import ABC, abstractmethod
from config import InitializationConfig, CLIMATIC_INDICES, ECOLOGICAL_INDICES

np.set_printoptions(threshold=sys.maxsize)
from utils import printt

NORTH_POLE_THRESHOLD = 66.5
SOUTH_POLE_THRESHOLD = -66.5
MAX_NAN_PERCENTAGE = 0.7
CLIMATIC_FILEPATH = "/Net/Groups/BGI/scratch/mweynants/DeepExtremes/v3/PEICube.zarr"
ECOLOGICAL_FILEPATH = (
    lambda index: f"/Net/Groups/BGI/work_1/scratch/fluxcom/upscaling_inputs/MODIS_VI_perRegion061/{index}/Groups_{index}gapfilled_QCdyn.zarr"
)
VARIABLE_NAME = lambda index: f"{index}gapfilled_QCdyn"


@staticmethod
def create_handler(config, n_samples):
    if config.index in ECOLOGICAL_INDICES:
        return EcologicalDatasetHandler(config=config, n_samples=n_samples)
    elif config.index in CLIMATIC_INDICES:
        return ClimaticDatasetHandler(config=config, n_samples=n_samples)
    else:
        raise ValueError("Invalid index")


class DatasetHandler(ABC):
    def __init__(self, config: InitializationConfig, n_samples: Union[int, None]):
        """
        Initialize DatasetHandler.

        Parameters:
        n_samples (Union[int, None]): Number of samples to select.
        time_resolution (int, optional): temporal resolution of the msc, to reduce computationnal workload. Defaults to 5.
        """
        self.config = config
        self.n_samples = n_samples

        self.max_data = None
        self.min_data = None
        self.data = None
        self.variable = None

    def preprocess_data(
        self,
        scale=True,
        reduce_temporal_resolution=True,
        return_time_serie=False,
        remove_nan=True,
    ):
        """
        Preprocess data based on the index.
        """
        self._dataset_specific_loading()
        self.filter_dataset()
        # Stack the dimensions
        self.data = self.data.stack(location=("longitude", "latitude"))

        # Select only a subset of the data if n_samples is specified
        if self.n_samples:
            self.randomly_select_n_samples()
        else:
            self.data = self.data[self.variable_name]
            printt(
                f"Computation on the entire dataset. {self.data.sizes['location']} samples"
            )

        self.compute_variable()

        if reduce_temporal_resolution:
            self._reduce_temporal_resolution()

        if remove_nan and ~self.n_samples:
            self._remove_nans()

        if scale:
            self._scale_variable()

        self.variable = self.variable.transpose("location", "dayofyear", ...)

        if return_time_serie:
            self.data = self.data.transpose("location", "time", ...)
            return self.variable, self.data
        else:
            return self.variable

    @abstractmethod
    def _dataset_specific_loading(self, *args, **kwargs):
        pass

    @abstractmethod
    def filter_dataset(self, *args, **kwargs):
        pass

    def _spatial_filtering(self, data):
        # Filter data from the polar regions
        data = self.data.where(
            np.abs(self.data.latitude) <= NORTH_POLE_THRESHOLD, drop=True
        )
        data = self.data.where(
            np.abs(self.data.latitude) >= SOUTH_POLE_THRESHOLD, drop=True
        )

        # Filter dataset to select Europe
        # Select European data
        in_europe = self._is_in_europe(data.longitude, data.latitude)
        data = data.where(in_europe, drop=True)
        printt("Data filtred to Europe.")

        data = self._is_land(data)
        printt("Data filtred to land.")
        return data

    def _is_land(self, data):
        # Create a land mask
        land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
        mask = land.mask(data.longitude, data.latitude).astype(bool)

        # Mask the ocean
        mask = ~mask

        # Apply the mask to filter out ocean locations
        data = data.where(mask, drop=True)
        return data

    def randomly_select_n_samples(self, factor=5):
        """
        Randomly select a subset of n_samples of data.
        """
        # Generate a large number of random coordinates
        n_candidates = self.n_samples * factor
        lons = np.random.choice(self.data.longitude, size=n_candidates, replace=True)
        lats = np.random.choice(self.data.latitude, size=n_candidates, replace=True)

        selected_locations = list(zip(lons, lats))
        self.data = self.data.chunk({"time": len(self.data.time), "location": 1})

        # Select the values at the specified coordinates
        selected_data = self.data[self.variable_name].sel(location=selected_locations)
        # Remove NaNs
        condition = ~selected_data.isnull().any(dim="time").compute()  #
        selected_data = selected_data.where(condition, drop=True)

        # Select randomly n_samples samples in selected_data
        self.data = selected_data.isel(
            location=np.random.choice(
                selected_data.location.size,
                size=min(self.n_samples, selected_data.location.size),
                replace=False,
            )
        )
        if self.data.sizes["location"] != self.n_samples:
            raise (
                "Number of samples != n_samples. The number of samples without NaNs is likely too low, increase the factor of n_candidates."
            )
        printt(f"Randomly selected {self.data.sizes['location']} samples for training.")

    def _is_in_europe(self, lon, lat):
        """
        Check if the given longitude and latitude are within the bounds of Europe.
        """
        # Define Europe boundaries (these are approximate)
        lon_min, lon_max = -31.266, 39.869  # Longitude boundaries
        lat_min, lat_max = 27.636, 81.008  # Latitude boundaries

        # Check if the point is within the defined boundaries
        in_europe = (
            (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
        )
        return in_europe

    def compute_variable(self):
        """
        Compute the Mean Seasonal Cycle (MSC) and optionally the Variance Seasonal Cycle (VSC)
        of n samples and scale it between 0 and 1.

        Time resolution reduces the resolution of the MSC to decrease computation workload.
        Number of values = 366 / time_resolution.
        """
        msc = self._compute_msc()
        printt("MSC computed.")

        if self.config.compute_variance:
            vsc = self._compute_vsc()
            self.variable = self._combine_msc_vsc(msc, vsc)
            printt("Variance is computed.")
        else:
            self.variable = msc
        self._rechunk_data()

    def _compute_msc(self):
        return self.data.groupby("time.dayofyear").mean("time", skipna=True)

    def _compute_vsc(self):
        return (
            self.data.groupby("time.dayofyear")
            .var("time", skipna=True)
            .isel(dayofyear=slice(1, 365))
        )

    def _combine_msc_vsc(self, msc, vsc):
        msc_vsc = xr.concat([msc, vsc], dim="dayofyear")
        total_days = len(msc_vsc.dayofyear)
        return msc_vsc.assign_coords(dayofyear=("dayofyear", range(total_days)))

    def _rechunk_data(self):
        self.variable = self.variable.chunk(
            {"dayofyear": len(self.variable.dayofyear), "location": 1}
        )

    def _remove_nans(self):
        condition = ~self.variable.isnull().any(dim="dayofyear").compute()
        self.variable = self.variable.where(condition, drop=True)
        printt("NaNs removed.")

    def _reduce_temporal_resolution(self):
        self.variable = self.variable.isel(
            dayofyear=slice(
                2, len(self.variable.dayofyear) - 1, self.config.time_resolution
            )
        )

    def _get_min_max_data(self):
        min_max_data_path = self.config.saving_path / "min_max_data.zarr"
        if min_max_data_path.exists():
            self._load_min_max_data(min_max_data_path)
        else:
            self._compute_and_save_min_max_data(min_max_data_path)

    def _compute_and_save_min_max_data(self, min_max_data_path):
        assert (
            self.max_data and self.min_data
        ) is None, "the min and max of the data are already defined."
        assert self.config.path_load_experiment is None, "A model is already loaded."

        self.max_data = self.variable.max(dim=["location"])
        self.min_data = self.variable.min(dim=["location"])
        # Save min_data and max_data
        if not min_max_data_path.exists():
            min_max_data = xr.Dataset(
                {
                    "max_data": self.max_data,
                    "min_data": self.min_data,
                },
                coords={"dayofyear": self.variable.coords["dayofyear"].values},
            )
            min_max_data.to_zarr(min_max_data_path)
            printt("Min and max data saved.")
        else:
            raise FileNotFoundError(f"{min_max_data_path} already exist.")

    def _load_min_max_data(self, min_max_data_path):
        """
        Load min-max data from the file.
        """
        min_max_data = xr.open_zarr(min_max_data_path)
        self.min_data = min_max_data.min_data
        self.max_data = min_max_data.max_data

    def _scale_variable(self):
        self._get_min_max_data()
        self.variable = (self.min_data - self.variable) / (
            self.max_data - self.min_data
        ) + 1e-8
        self.variable = self.variable.chunk(
            {"dayofyear": len(self.variable.dayofyear), "location": 1}
        )
        printt("Data are scaled between 0 and 1.")

    def _deseasonalize(self, subset_data, subset_msc):
        # Align subset_msc with subset_data
        aligned_msc = subset_msc.sel(dayofyear=subset_data["time.dayofyear"])
        # Subtract the seasonal cycle
        deseasonalized = subset_data - aligned_msc
        deseasonalized = deseasonalized.isel(
            time=slice(2, len(deseasonalized.time) - 1, 5)
        )
        return deseasonalized


class ClimaticDatasetHandler(DatasetHandler):
    def _dataset_specific_loading(self):
        """
        Preprocess data based on the index.
        """
        if self.config.index in ["pei_30", "pei_90", "pei_180"]:
            self.load_data(CLIMATIC_FILEPATH)
        else:
            raise ValueError(
                "Index unavailable. Index available:\n -Climatic: 'pei_30', 'pei_90', 'pei_180'."
            )
        return self.data

    def load_data(self, filepath):
        """
        Load data from the specified filepath.

        Parameters:
        filepath (str): Path to the data file.
        """
        if not filepath:
            filepath = CLIMATIC_FILEPATH(self.config.index)
        # name of the variable in the xarray. self.variable_name
        self.variable_name = self.config.index
        self.data = xr.open_zarr(filepath)[[self.variable_name]]
        self._transform_longitude()
        printt("Data loaded from {}".format(filepath))

    def _transform_longitude(self):
        # Transform the longitude coordinates
        self.data = self.data.roll(
            longitude=180 * 4, roll_coords=True
        )  # Shifts the data of longitude of 180*4 elements, elements that roll past the end are re-introduced

        # Transform the longitude coordinates to -180 and 180
        self.data = self.data.assign_coords(
            longitude=self._coordstolongitude(self.data.longitude)
        )

    def _coordstolongitude(self, x):
        """Transform the longitude coordinates from between 0 and 360 to between -180 and 180."""
        return ((x + 180) % 360) - 180

    def filter_dataset(self):
        """
        Apply climatic transformations using xarray.apply_ufunc.
        """
        assert (
            self.config.index in CLIMATIC_INDICES
        ), f"Index unavailable. Index available: {CLIMATIC_INDICES}."

        assert self.data is not None, "Data not loaded."

        # Assert dimensions are as expected after loading and transformation
        assert all(
            dim in self.data.sizes for dim in ("time", "latitude", "longitude")
        ), "Dimension missing"
        # Ensure longitude values are within the expected range
        assert (
            (self.data.longitude >= -180) & (self.data.longitude <= 180)
        ).all(), "Longitude values should be in the range -180 to 180"

        # Remove the years before 1970 due to quality
        self.data = self.data.sel(
            time=slice(datetime.date(1970, 1, 1), datetime.date(2022, 12, 31))
        )

        self.data = self._spatial_filtering(self.data)

        printt(f"Climatic data loaded with dimensions: {self.data.sizes}")


class EcologicalDatasetHandler(DatasetHandler):

    def _dataset_specific_loading(self):
        """
        Preprocess data based on the index.
        """
        if self.config.index in ECOLOGICAL_INDICES:
            filepath = ECOLOGICAL_FILEPATH(self.config.index)
            self.load_data(filepath)
            self.reduce_resolution()
        else:
            raise NotImplementedError(
                f"Index {self.config.index} unavailable. Ecological Index available: {ECOLOGICAL_INDICES}."
            )
        return self.data

    def load_data(self, filepath=None):
        """
        Load data from the specified filepath.

        Parameters:
        filepath (str): Path to the data file.
        """
        if not filepath:
            filepath = ECOLOGICAL_FILEPATH(self.config.index)

        self.variable_name = VARIABLE_NAME(self.config.index)
        self.data = xr.open_zarr(filepath, consolidated=False)[[self.variable_name]]
        printt("Data loaded from {}".format(filepath))
        self._stackdims()

    def _stackdims(self):
        self.data = self.data.stack(
            {
                "latitude": ["latchunk", "latstep_modis"],
                "longitude": ["lonchunk", "lonstep_modis"],
            }
        )
        self.data = self.data.reset_index(["latitude", "longitude"])
        self.data["latitude"] = self.data.latchunk + self.data.latstep_modis
        self.data["longitude"] = self.data.lonchunk + self.data.lonstep_modis

        self.data = self.data.set_index(latitude="latitude", longitude="longitude")
        self.data = self.data.drop(
            ["latchunk", "latstep_modis", "lonchunk", "lonstep_modis"]
        )

    def reduce_resolution(self):
        res_lat, res_lon = len(self.data.latitude), len(self.data.longitude)
        self.data = self.data.coarsen(latitude=5, longitude=5, boundary="trim").mean()
        printt(
            f"Reduce the resolution from ({res_lat}, {res_lon}) to ({len(self.data.latitude)}, {len(self.data.longitude)})."
        )

    def filter_dataset(self):
        """
        Standarize the ecological xarray. Remove irrelevant area, and reshape for the PCA.
        """
        assert (
            self.config.index in ECOLOGICAL_INDICES
        ), f"Index {self.config.index} unavailable. Index available: {ECOLOGICAL_INDICES}."

        assert self.data is not None, "Data not loaded."

        # Assert dimensions are as expected after loading and transformation
        assert all(
            dim in self.data.sizes for dim in ("time", "latitude", "longitude")
        ), "Dimension missing"

        # Temporal filtering.
        self.data = self.data.sel(
            time=slice(datetime.date(2000, 1, 1), datetime.date(2022, 12, 31))
        )

        self.data = self._spatial_filtering(self.data)

        printt(f"Ecological data loaded with dimensions: {self.data.sizes}")
