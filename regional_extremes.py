import xarray as xr
from argparse import Namespace
import numpy as np
import json
import random
import datetime
from sklearn.decomposition import PCA
import pickle as pk
from pathlib import Path
from typing import Union
import time
import sys
import os

from utils import printt, int_or_none

np.set_printoptions(threshold=sys.maxsize)

from global_land_mask import globe

import argparse  # import ArgumentParser

CLIMATIC_FILEPATH = "/Net/Groups/BGI/scratch/mweynants/DeepExtremes/v3/PEICube.zarr"
CURRENT_DIRECTORY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))
PARENT_DIRECTORY_PATH = os.path.abspath(os.path.join(CURRENT_DIRECTORY_PATH, os.pardir))
NORTH_POLE_THRESHOLD = 66.5
SOUTH_POLE_THRESHOLD = -66.5


# Argparser for all configuration needs
def parser_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--id",
        type=str,
        default=None,
        help="id is time of the job launch and job_id",
    )
    parser.add_argument(
        "--index",
        type=str,
        default="pei_180",
        help=" The climatic or ecological index to be processed (default: pei_180). "
        "Index available:\n -Climatic: 'pei_30', 'pei_90', 'pei_180'. \n Ecological: 'None.",
    )
    parser.add_argument(
        "--time_resolution",
        type=int,
        default=5,
        help="time_resolution (int, optional): temporal resolution of the msc, to reduce computationnal workload (default: 5). ",
    )

    parser.add_argument(
        "--n_components",
        type=int,
        default=3,
        help="Path to the raw MMEarth dataset folder (default: None). "
        "If not given the environment variable MMEARTH_DIR will be used",
    )

    parser.add_argument(
        "--n_samples",
        type=int_or_none,
        default=10,
        help="Select randomly n_samples**2. Use 'None' for no limit.",
    )

    parser.add_argument(
        "--n_bins",
        type=int,
        default=25,
        help="Path to the raw MMEarth dataset folder (default: None). "
        "If not given the environment variable MMEARTH_DIR will be used",
    )

    parser.add_argument(
        "--saving_path",
        type=Path,
        default="./experiments/",
        help="Path to the raw MMEarth dataset folder (default: None). "
        "If not given the environment variable MMEARTH_DIR will be used",
    )

    parser.add_argument(
        "--path_load_model",
        type=Path,
        default=None,
        help="Path of the trained model folder.",
    )
    return parser


class SharedConfig:
    def __init__(self, args: Namespace):
        """
        Initialize SharedConfig with the provided arguments.

        Args:
            args (argparse.Namespace): Parsed arguments from argparse.ArgumentParser().parse_args()
        """
        self.index = args.index
        self.time_resolution = args.time_resolution
        self.path_load_model = args.path_load_model
        self.pca = None
        self.min_data = None
        self.max_data = None

        if self.path_load_model is None:
            printt(
                "Initialisation of a new model, no path provided for an existing model."
            )
            self._initialize_new_model(args)
        else:
            printt(f"Loading of the model path: {self.path_load_model}")
            self._load_existing_model()

    def _initialize_new_model(self, args: Namespace):
        """
        Initialize settings for a new model when no model is loaded.

        Args:
            args (argparse.Namespace): Parsed arguments from argparse.ArgumentParser().parse_args()
        """
        self._set_saving_path(args)
        self._save_args(args)

    def _set_saving_path(self, args: Namespace):
        """
        Set the saving path for the new model.

        Args:
            args (argparse.Namespace): Parsed arguments from argparse.ArgumentParser().parse_args()
        """
        if args.id:
            date_of_today = args.id
        else:
            date_of_today = datetime.datetime.today().strftime("%Y-%m-%d_%H:%M:%S")

        self.saving_path = (
            Path(PARENT_DIRECTORY_PATH)
            / Path(args.saving_path)
            / date_of_today
            / self.index
        )
        print(f"The saving path is: {self.saving_path}")
        self.saving_path.mkdir(parents=True, exist_ok=True)
        args.saving_path = str(self.saving_path)

    def _save_args(self, args: Namespace):
        """
        Save the arguments to a JSON file for future reference.

        Args:
            args (argparse.Namespace): Parsed arguments from argparse.ArgumentParser().parse_args()
        """
        assert self.path_load_model is None
        args_path = self.saving_path / "args.json"
        if not args_path.exists():
            with open(args_path, "w") as f:
                json.dump(args.__dict__, f, indent=4)
        printt("args saved. ")

    def _load_existing_model(self):
        """
        Load an existing model's PCA matrix and min-max data from files.
        """
        self.path_load_model = Path(self.path_load_model) / self.index
        self.saving_path = self.path_load_model
        self._load_min_max_data()
        self._load_pca_matrix()

    def _load_min_max_data(self):
        """
        Load min-max data from the file.
        """
        min_max_data_path = self.path_load_model / "min_max_data.json"
        with open(min_max_data_path, "r") as f:
            min_max_data = json.load(f)
            self.min_data = min_max_data["min_data"]
            self.max_data = min_max_data["max_data"]

    def _load_pca_matrix(self):
        """
        Load PCA matrix from the file.
        """
        pca_path = self.path_load_model / "pca_matrix.pkl"
        with open(pca_path, "rb") as f:
            self.pca = pk.load(f)


class RegionalExtremes(SharedConfig):
    def __init__(
        self,
        config: SharedConfig,
        n_components: int,
        n_bins: int,
    ):
        """
        Compute the regional extremes by defining boxes of similar region using a PCA computed on the mean seasonal cycle of the samples.

        Args:
            config (SharedConfig): Shared attributes across the classes.
            n_components (int): number of components of the PCA
            n_bins (int): Number of bins per component to define the boxes. Number of boxes = n_bins**n_components
        """
        self.config = config
        self.n_components = n_components
        self.n_bins = n_bins
        self.pca = self.config.pca

    def compute_pca_and_transform(
        self,
        scaled_data,
    ):
        """compute the principal component analysis (PCA) on the mean seasonal cycle (MSC) of n samples and scale it between 0 and 1. Each time step of the msc is considered as an independent component. nb of time_step used for the PCA computation = 366 / time_resolution.

        Args:
            n_components (int, optional): number of components to compute the PCA. Defaults to 3.
            time_resolution (int, optional): temporal resolution of the msc, to reduce computationnal workload. Defaults to 5.
        """
        assert self.config.path_load_model is None, "A model is already loaded."
        assert self.pca is None, "The PCA is already fitted."
        assert scaled_data.shape[1] == round(366 / self.config.time_resolution)
        assert (self.n_components > 0) & (
            self.n_components <= 366
        ), "n_components have to be in the range of days of a years"

        # Compute the PCA
        self.pca = PCA(n_components=self.n_components)

        # Fit the PCA. Each colomns give us the projection through 1 component.
        pca_components = self.pca.fit_transform(scaled_data)

        printt(
            f"PCA performed. sum explained variance: {sum(self.pca.explained_variance_ratio_)}"
        )

        # Save the PCA model
        pca_path = self.config.saving_path / "pca_matrix.pkl"
        with open(pca_path, "wb") as f:
            pk.dump(self.pca, f)
        printt("PCA saved.")

        return pca_components

    def apply_pca(self, scaled_data: np.ndarray) -> np.ndarray:
        """
        Compute the mean seasonal cycle (MSC) of the samples and scale it between 0 and 1.
        Then apply the PCA already fit on the new data. Each time step of the MSC is considered
        as an independent component. The number of time steps used for the PCA computation
        is 366 / time_resolution.

        Args:
            scaled_data (np.ndarray): Data to be transformed using PCA.

        Returns:
            np.ndarray: Transformed data after applying PCA.
        """
        self._validate_scaled_data(scaled_data)
        transformed_data = self.pca.transform(scaled_data)
        return transformed_data

    def _validate_scaled_data(self, scaled_data: np.ndarray) -> None:
        """Validates the scaled data to ensure it matches the expected shape."""
        expected_shape = round(366 / self.config.time_resolution)
        if scaled_data.shape[1] != expected_shape:
            raise ValueError(
                f"scaled_data should have {expected_shape} columns, but has {scaled_data.shape[1]} columns."
            )

    def define_limits_bins(self, projected_data: np.ndarray) -> list[np.ndarray]:
        """
        Define the bounds of each bin on the projected data for each component.
        Ideally applied on the largest possible amount of data to capture
        the distribution in the projected space (especially minimum and maximum).
        Fit the PCA with a subset of the data, then project the full dataset,
        then define the bins on the full dataset projected.
        n_bins is per component, so number of boxes = n_bins**n_components

        Args:
            projected_data (np.ndarray): Data projected after PCA.

        Returns:
            list of np.ndarray: List where each array contains the bin limits for each component.
        """
        self._validate_inputs(projected_data)
        limits_bins = self._calculate_limits_bins(projected_data)
        self._save_limits_bins(limits_bins)
        return limits_bins

    def _validate_inputs(self, projected_data: np.ndarray) -> None:
        """Validates the inputs for define_limits_bins."""
        if not hasattr(self.pca, "explained_variance_"):
            raise ValueError("PCA model has not been trained yet.")

        if projected_data.shape[1] != self.n_components:
            raise ValueError(
                "projected_data should have the same number of columns as n_components"
            )

        if self.n_bins <= 0:
            raise ValueError("n_bins should be greater than 0")

    def _calculate_limits_bins(self, projected_data: np.ndarray) -> list[np.ndarray]:
        """Calculates the limits bins for each component."""
        return [
            np.linspace(
                np.min(projected_data[:, component]),
                np.max(projected_data[:, component]),
                self.n_bins + 1,
            )[
                1:-1
            ]  # Remove first and last limits to avoid attributing new bins to extreme values
            for component in range(self.n_components)
        ]

    def _save_limits_bins(self, limits_bins: list[np.ndarray]) -> None:
        """Saves the limits bins to a file."""
        limits_bins_path = self.config.saving_path / "limits_bins.npy"
        if os.path.exists(limits_bins_path):
            raise FileExistsError(
                f"The file {limits_bins_path} already exists. Rewriting is not allowed."
            )
        np.save(limits_bins_path, limits_bins)

    # Function to find the box for multiple points
    def find_bins(self, projected_data, limits_bins):
        assert projected_data.shape[1] == len(limits_bins)
        assert (
            len(limits_bins) == self.n_components
        ), "the lenght of limits_bins list is not equal to the number of components"
        assert (
            limits_bins[0].shape[0] == self.n_bins - 1
        ), "the limits do not fit the number of bins"

        box_indices = np.zeros(
            (projected_data.shape[0], projected_data.shape[1]), dtype=int
        )
        for i, limits_bin in enumerate(limits_bins):
            box_indices[:, i] = np.digitize(projected_data[:, i], limits_bin)

        return box_indices

    def apply_threshold():
        raise NotImplementedError()


class DatasetHandler(SharedConfig):
    def __init__(
        self,
        config: SharedConfig,
        n_samples: Union[int, None],
    ):
        """
        Initialize DatasetHandler.

        Parameters:
        index (str): The climatic or ecological index to be processed.
        n_samples (Union[int, None]): Number of samples to select.
        load_data (bool): Flag to determine if data should be loaded during initialization.
        time_resolution (int, optional): temporal resolution of the msc, to reduce computationnal workload. Defaults to 5.
        """
        self.config = config
        self.n_samples = n_samples

        self.max_data = self.config.max_data
        self.min_data = self.config.min_data

        self.data = None

    def preprocess_data(self):
        """
        Preprocess data based on the index.
        """
        if self.config.index in ["pei_30", "pei_90", "pei_180"]:
            filepath = CLIMATIC_FILEPATH
            self.load_data(filepath)
        else:
            raise NotImplementedError(
                "Index unavailable. Index available:\n -Climatic: 'pei_30', 'pei_90', 'pei_180'. \n Ecological: 'None."
            )

        # Select only a subset of the data if n_samples is specified
        if self.n_samples:
            self.randomly_select_data()
        else:
            printt(
                f"Computation on the entire dataset. {self.data.sizes['latitude'] * self.data.sizes['longitude']} samples"
            )

        self.apply_climatic_transformations()
        self.compute_and_scale_the_msc()
        return self.data

    def load_data(self, filepath):
        """
        Load data from the specified filepath.

        Parameters:
        filepath (str): Path to the data file.
        """
        self.data = xr.open_zarr(filepath)[[self.config.index]]
        printt("Data loaded from {}".format(filepath))

    # Transform the longitude coordinates to -180 and 180
    def coordstolongitude(self, x):
        return ((x + 180) % 360) - 180

    def compute_and_save_min_max_data(self, data):
        assert (
            self.max_data and self.min_data
        ) is None, "the min and max of the data are already defined."
        assert self.config.path_load_model is None, "A model is already loaded."

        self.max_data = self.data[self.config.index].max().values
        self.min_data = self.data[self.config.index].min().values

        # Save min_data and max_data
        min_max_data_path = self.config.saving_path / "min_max_data.json"
        with open(min_max_data_path, "w") as f:
            json.dump(
                {
                    "min_data": self.min_data.tolist(),
                    "max_data": self.max_data.tolist(),
                },
                f,
                indent=4,
            )
        print("Min and max data saved.")

    def randomly_select_data(self):
        """
        Randomly select a subset of the data based on n_samples.
        """
        # select n_samples instead of n_samples**2 but computationnally expensive!
        # self.data = self.data.stack(lonlat=("longitude", "latitude")).transpose(
        #     "lonlat", "time", ...
        # )
        # lonlat_indices = random.choices(self.data.lonlat.values, k=self.n_samples)
        # self.data = self.data.sel(lonlat=lonlat_indices)
        lon_indices = []
        lat_indices = []

        while len(lon_indices) < self.n_samples:
            lon_index = random.randint(0, self.data.longitude.sizes["longitude"] - 1)
            lat_index = random.randint(0, self.data.latitude.sizes["latitude"] - 1)

            lon = self.coordstolongitude(self.data.longitude[lon_index].item())

            lat = self.data.latitude[lat_index].item()
            # if location is on a land and not in the polar regions.
            if globe.is_land(lat, lon) and np.abs(lat) <= NORTH_POLE_THRESHOLD:
                lon_indices.append(lon_index)
                lat_indices.append(lat_index)

        self.data = self.data.isel(longitude=lon_indices, latitude=lat_indices)
        printt(
            f"Randomly selected {self.data.sizes['latitude'] * self.data.sizes['longitude']} samples for training."
        )

    def apply_climatic_transformations(self):
        """
        Apply transformations to the climatic data.
        """
        assert self.config.index in [
            "pei_30",
            "pei_90",
            "pei_180",
        ], "Index unavailable. Index available: 'pei_30', 'pei_90', 'pei_180'."

        assert self.data is not None, "Data not loaded."

        # Assert dimensions are as expected after loading and transformation
        assert all(
            dim in self.data.sizes for dim in ("time", "latitude", "longitude")
        ), "Dimension missing"
        # Ensure longitude values are within the expected range
        assert (
            (self.data.longitude >= 0) & (self.data.longitude <= 360)
        ).all(), "Longitude values should be in the range 0 to 360"

        # Remove the year 1950 because the data are inconsistent
        self.data = self.data.sel(
            time=slice(datetime.date(1951, 1, 1), datetime.date(2022, 12, 31))
        )

        # Remove data from the polar regions
        self.data = self.data.where(
            np.abs(self.data.latitude) <= NORTH_POLE_THRESHOLD, drop=True
        )
        self.data = self.data.where(
            np.abs(self.data.latitude) >= SOUTH_POLE_THRESHOLD, drop=True
        )

        # Transform the longitude coordinates
        # Shifts the data of longitude of 180*4 elements, elements that roll past the end are re-introduced
        self.data = self.data.roll(longitude=180 * 4, roll_coords=True)
        # Transform the longitude coordinates to -180 and 180
        self.data = self.data.assign_coords(
            longitude=self.coordstolongitude(self.data.longitude)
        )

        # Remove latitude above polar circles.
        self.data = self.data.stack(lonlat=("longitude", "latitude")).transpose(
            "lonlat", "time", ...
        )

        printt(f"Climatic data loaded with dimensions: {self.data.sizes}")

    def compute_and_scale_the_msc(self):
        """
        compute the MSC of n samples and scale it between 0 and 1. Each values of the msc is considered
        as an independent component. time_resolution reduce the resolution of the msc to reduce the computation workload during the computation. nb values = 366 / time_resolution.
        """
        assert (self.n_samples > 0) & (self.n_samples <= self.data.sizes["lonlat"])

        # Compute the MSC
        self.data["msc"] = (
            self.data[self.config.index]
            .groupby("time.dayofyear")
            .mean("time")
            .drop_vars(["lonlat", "longitude", "latitude"])
        )

        # reduce the temporal resolution
        self.data = self.data["msc"].isel(
            dayofyear=slice(1, 366, self.config.time_resolution)
        )

        if (self.min_data and self.max_data) is None:
            self.compute_and_save_min_max_data(self.data)

        # Scale the data between 0 and 1
        self.data = (self.min_data - self.data) / (self.max_data - self.min_data)


# class EcologicalRegionalExtremes:
#     def __init__(
#         self,
#         vegetation_index,
#         # isclimatic: bool,
#     ):
#         self.vegetation_index = vegetation_index
#         self.filepath = f"/Net/Groups/BGI/work_1/scratch/fluxcom/upscaling_inputs/MODIS_VI_perRegion061/{vegetation_index}/Groups_dyn_{vegetation_index}_MSC_snowfrac.zarr"
#         self.filename = f"Groups_dyn_{vegetation_index}_MSC_snowfrac"
#
#     def apply_transformations(self):
#         # Load the MSC of MODIS
#         ds = xr.open_zarr(self.filepath, consolidated=False)
#         ds_msc = ds[self.filename].stack(
#             {"lat": ["latchunk", "latstep_modis"], "lon": ["lonchunk", "lonstep_modis"]}
#         )
#
#         # Select k locations randomly to train the PCA:
#         lat_indices = random.choices(ds_msc.lat.values, k=3000)
#         lon_indices = random.choices(ds_msc.lon.values, k=3000)
#
#         # Select the MSC of those locations:
#         return


def main_train_pca(args):
    config = SharedConfig(args)
    dataset_processor = DatasetHandler(
        config=config,
        n_samples=args.n_samples,
    )
    data_subset = dataset_processor.preprocess_data()

    extremes_processor = RegionalExtremes(
        config=config,
        n_components=args.n_components,
        n_bins=args.n_bins,
    )
    projected_data = extremes_processor.compute_pca_and_transform(
        scaled_data=data_subset
    )


def main_define_limits(args):
    args.path_load_model = "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2014047_2024-07-18_16:02:30"

    config = SharedConfig(args)
    dataset_processor = DatasetHandler(
        config=config,
        n_samples=args.n_samples,  # all the dataset
    )
    data = dataset_processor.preprocess_data()

    extremes_processor = RegionalExtremes(
        config=config,
        n_components=args.n_components,
        n_bins=args.n_bins,
    )
    projected_data = extremes_processor.apply_pca(scaled_data=data)
    extremes_processor.define_limits_bins(projected_data=projected_data)


if __name__ == "__main__":
    args = parser_arguments().parse_args()

    # To train the PCA:
    # main_train_pca(args)

    # To define the limits:
    main_define_limits(args)
