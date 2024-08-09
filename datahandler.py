import xarray as xr
import zarr
import dask
import dask.array as da
import numpy as np
import json
import random
import datetime
import pandas as pd
from pathlib import Path
from typing import Union
from dask.array.core import map_blocks
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
MAX_NAN_PERCENTAGE = 0.3
CLIMATIC_FILEPATH = "/Net/Groups/BGI/scratch/mweynants/DeepExtremes/v3/PEICube.zarr"
ECOLOGICAL_FILEPATH = (
    lambda index: f"/Net/Groups/BGI/work_1/scratch/fluxcom/upscaling_inputs/MODIS_VI_perRegion061/{index}/Groups_{index}gapfilled_QCdyn.zarr"
)
VARIABLE_NAME = lambda index: f"{index}gapfilled_QCdyn"


class DatasetHandler(ABC):
    def __init__(
        self,
        config: InitializationConfig,
        n_samples: Union[int, None],
    ):
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

    def preprocess_data(self):
        """
        Preprocess data based on the index.
        """
        self._dataset_specific_loading()

        # Select only a subset of the data if n_samples is specified
        if self.n_samples:
            self.randomly_select_data()
        else:
            printt(
                f"Computation on the entire dataset. {self.data.sizes['latitude'] * self.data.sizes['longitude']} samples"
            )
            self.standardize_dataset()

        self.compute_and_scale_the_msc()
        return self.data

    @abstractmethod
    def _dataset_specific_loading(self, *args, **kwargs):
        pass

    @abstractmethod
    def standardize_dataset(self, *args, **kwargs):
        pass

    def randomly_select_data(self):
        """
        Randomly select a subset of n_samples of data.
        """
        selected_locations = self._select_valid_locations()
        self._update_data_with_selected_locations(selected_locations)
        printt(
            f"Randomly selected {self.data.sizes['location']} samples for training in Europe."
        )

    def _select_valid_locations(self):
        selected_locations = []
        while len(selected_locations) < self.n_samples:
            lon, lat = self._get_random_coordinates()
            if self._is_valid_location(lon, lat):
                selected_locations.append((lon, lat))
        return selected_locations

    def _get_random_coordinates(self):
        lon_index = random.randint(0, self.data.longitude.sizes["longitude"] - 1)
        lat_index = random.randint(0, self.data.latitude.sizes["latitude"] - 1)
        return (
            self.data.longitude[lon_index].item(),
            self.data.latitude[lat_index].item(),
        )

    def _is_valid_location(self, lon, lat):
        return (
            globe.is_land(lat, lon)
            and abs(lat) <= NORTH_POLE_THRESHOLD
            and self._is_in_europe(lon, lat)
            and self._has_acceptable_nan_percentage(lon, lat)
        )

    def _has_acceptable_nan_percentage(self, lon, lat):
        data_location = self.data[self.variable_name].sel(longitude=lon, latitude=lat)
        nan_count = np.isnan(data_location).sum().values
        nan_percentage = nan_count / data_location.size
        return nan_percentage < MAX_NAN_PERCENTAGE

    def _update_data_with_selected_locations(self, selected_locations):

        lons, lats = zip(*selected_locations)
        # Ensure that each lons and lats are unique
        lons, lats = list(set(lons)), list(set(lats))

        # Select data using the MultiIndex
        self.data = (
            self.data.sel(longitude=lons, latitude=lats)
            .stack(location=("longitude", "latitude"))
            .sel(location=selected_locations)
            .transpose("location", "time", ...)
        )

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

    def compute_and_scale_the_msc(self):
        """
        Compute the Mean Seasonal Cycle (MSC) of n samples and scale it between 0 and 1.
        Time resolution reduces the resolution of the MSC to decrease computation workload.
        Number of values = 366 / time_resolution.
        """
        self._compute_msc()

        if self.config.compute_variance:
            self._compute_and_combine_vsc()

        self._reduce_temporal_resolution()
        # self.data = self.data.drop_vars(["location", "longitude", "latitude"])

        # Rechunck the data per time serie
        self.data = self.data.chunk(
            {"dayofyear": len(self.data.dayofyear), "location": 1}
        )
        # Use map_blocks to compute the condition in parallel
        # condition = self.data.map_blocks(
        #     lambda x: ~x.isnull().any(dim="dayofyear"),
        # )
        condition = ~self.data.isnull().any(dim="dayofyear").compute()
        # Assuming 'self.data' is your Xarray DataArray
        # condition = da.map_blocks(
        #     lambda x: ~x.isnull().any(dim="dayofyear"), self.data, dtype=bool
        # ).compute()

        self.data = self.data.where(condition, drop=True)
        printt("NaN removed.")

        self._get_min_max_data()
        self._scale_data()
        self.data = self.data.chunk(
            {"dayofyear": len(self.data.dayofyear), "location": 1}
        )
        printt("Data are scaled between 0 and 1.")

    def _compute_msc(self):
        self.data["msc"] = (
            self.data[self.variable_name]
            .groupby("time.dayofyear")
            .mean("time", skipna=True)
        )

    def _compute_and_combine_vsc(self):
        self.data["vsc"] = (
            self.data[self.variable_name]
            .groupby("time.dayofyear")
            .var("time", skipna=True)
            .drop_vars(["location", "longitude", "latitude"])
            .dropna(dim="time", how="any")
        )
        msc_vsc = xr.concat([self.data["msc"], self.data["vsc"]], dim="dayofyear")
        total_days = len(msc_vsc.dayofyear)
        msc_vsc = msc_vsc.assign_coords(dayofyear=("dayofyear", range(total_days)))
        self.data = msc_vsc
        printt("Variance is computed")

    def _reduce_temporal_resolution(self):
        self.data = self.data["msc"].isel(
            dayofyear=slice(1, len(self.data.dayofyear), self.config.time_resolution)
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

        # self.data = self.data.chunk({"dayofyear": 1, "location": 12000})
        self.max_data = self.data.max(dim=["location"])
        self.min_data = self.data.min(dim=["location"])
        # Save min_data and max_data
        if not min_max_data_path.exists():
            min_max_data = xr.Dataset(
                {
                    "max_data": self.max_data,
                    "min_data": self.min_data,
                },
                coords={"dayofyear": self.data.coords["dayofyear"].values},
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

    def _scale_data(self):
        self.data = (self.min_data.broadcast_like(self.data) - self.data) / (
            self.max_data.broadcast_like(self.data)
            - self.min_data.broadcast_like(self.data)
        ) + 1e-8


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

    def load_data(self, filepath):
        """
        Load data from the specified filepath.

        Parameters:
        filepath (str): Path to the data file.
        """
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

    def standardize_dataset(self):
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
        ).all(), "Longitude values should be in the range 0 to 360"

        # Remove the years before 1970 due to quality
        self.data = self.data.sel(
            time=slice(datetime.date(1970, 1, 1), datetime.date(2022, 12, 31))
        )

        # Filter data from the polar regions
        self.data = self.data.where(
            np.abs(self.data.latitude) <= NORTH_POLE_THRESHOLD, drop=True
        )
        self.data = self.data.where(
            np.abs(self.data.latitude) >= SOUTH_POLE_THRESHOLD, drop=True
        )

        # Filter dataset to select Europe
        # Select European data
        in_europe = self._is_in_europe(self.data.longitude, self.data.latitude)
        self.data = self.data.where(in_europe, drop=True)
        printt("Data filtred to Europe.")

        # Stack the dimensions
        self.data = self.data.stack(location=("longitude", "latitude")).transpose(
            "location", "time", ...
        )

        printt(f"Climatic data loaded with dimensions: {self.data.sizes}")


class EcologicalDatasetHandler(DatasetHandler):

    def _dataset_specific_loading(self):
        """
        Preprocess data based on the index.
        """
        if self.config.index in ECOLOGICAL_INDICES:
            filepath = ECOLOGICAL_FILEPATH(self.config.index)
            self.load_data(filepath)
            self.stackdims()
            self.reduce_resolution()
        else:
            raise NotImplementedError(
                f"Index {self.config.index} unavailable. Ecological Index available: {ECOLOGICAL_INDICES}."
            )

    def load_data(self, filepath):
        """
        Load data from the specified filepath.

        Parameters:
        filepath (str): Path to the data file.
        """
        self.variable_name = VARIABLE_NAME(self.config.index)
        self.data = xr.open_zarr(filepath, consolidated=False)[[self.variable_name]]
        printt("Data loaded from {}".format(filepath))

    def stackdims(self):
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

    def standardize_dataset(self):
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

        # Filter data to the drought indices range.
        self.data = self.data.sel(
            time=slice(datetime.date(2000, 1, 1), datetime.date(2022, 12, 31))
        )

        # Filter data from the polar regions
        self.data = self.data.where(
            np.abs(self.data.latitude) <= NORTH_POLE_THRESHOLD, drop=True
        )
        self.data = self.data.where(
            np.abs(self.data.latitude) >= SOUTH_POLE_THRESHOLD, drop=True
        )

        # Select European data
        in_europe = self._is_in_europe(self.data.longitude, self.data.latitude)
        self.data = self.data.where(in_europe, drop=True)
        printt("Data filtred to Europe.")

        # TODO extend to every region

        # Stack the dimensions
        self.data = self.data.stack(location=("longitude", "latitude")).transpose(
            "location", "time", ...
        )

        printt(f"Ecological data loaded with dimensions: {self.data.sizes}")
