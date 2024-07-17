import xarray as xr
import numpy as np
import json
import random
import datetime
from sklearn.decomposition import PCA
import pickle as pk
from pathlib import Path
from typing import Union
import time
import os

import argparse  # import ArgumentParser

CLIMATIC_FILEPATH = "/Net/Groups/BGI/scratch/mweynants/DeepExtremes/v3/PEICube.zarr"
CURRENT_DIRECTORY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))

# Argparser for all configuration needs
parser = argparse.ArgumentParser()

parser.add_argument(
    "--index",
    type=str,
    default="pei_180",
    help=" The climatic or ecological index to be processed (default: pei_180). "
    "Index available:\n -Climatic: 'pei_30', 'pei_90', 'pei_180'. \n Ecological: 'None.",
)

parser.add_argument(
    "--step_msc",
    type=int,
    default=5,
    help="step_msc (int, optional): temporal resolution of the msc, to reduce computationnal workload (default: 5). ",
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
    type=int,
    default=10,
    help="Path to the raw MMEarth dataset folder (default: None). "
    "If not given the environment variable MMEARTH_DIR will be used",
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
    "--full_dataset_bins_limits",
    type=bool,
    default=False,
    help="Path to the raw MMEarth dataset folder (default: None). "
    "If not given the environment variable MMEARTH_DIR will be used",
)

parser.add_argument(
    "--load_data",
    type=bool,
    default=True,
    help="Path to the raw MMEarth dataset folder (default: None). "
    "If not given the environment variable MMEARTH_DIR will be used",
)
# return parser.parse_args(args)


class RegionalExtremes:
    def __init__(
        self,
        index: str,
        step_msc,
        n_components,
        n_bins,
        saving_path: Union[Path, None],
    ):
        """_summary_

        Args:
            index (str): _description_
            step_msc (_type_): _description_
            n_samples (int, optional): number of samples to fit the PCA. Defaults to 10.
            n_components (int, optional): number of components to compute the PCA. Defaults to 3.
            n_components (_type_): _description_
            n_bins (_type_): _description_
            saving_path (_type_): _description_
        """
        self.index = index
        self.step_msc = step_msc
        self.n_components = n_components
        self.n_bins = n_bins
        self.saving_path = Path(CURRENT_DIRECTORY_PATH) / Path(saving_path)

        self.pca = None

    def compute_pca_and_transform(
        self,
        scaled_data,
    ):
        """compute the principal component analysis (PCA) on the mean seasonal cycle (MSC) of n samples and scale it between 0 and 1. Each time step of the msc is considered as an independent component. nb of time_step used for the PCA computation = 366 / step_msc.

        Args:
            n_components (int, optional): number of components to compute the PCA. Defaults to 3.
            step_msc (int, optional): temporal resolution of the msc, to reduce computationnal workload. Defaults to 5.
        """
        assert scaled_data.shape[1] < 366
        assert (self.n_components > 0) & (
            self.n_components <= 366
        ), "n_components have to be in the range of days of a years"

        # Compute the PCA
        self.pca = PCA(n_components=self.n_components)
        # Fit the PCA. Each colomns give us the projection through 1 component.
        pca_components = self.pca.fit_transform(scaled_data)
        print(
            f"PCA performed. sum explained variance: {sum(self.pca.explained_variance_ratio_)}"
        )
        return pca_components

    def define_limits_bins(self, projected_data):
        """
            Define the bounds of each bin on the projected data for each component.


            Ideally applied on the largest possible  amount of data to capture
            the distribution in the projected space (especially minimum and maximum).
            So ideally, fit the PCA with a subset of the data, then project the full dataset,
            then define the bins on the full dataset projected.
            n_bins is per components, so number of boxes = n_bins**n_components
        Args:
        projected_data (np.ndarray): Data projected after PCA.

        Returns:
        list of np.ndarray: List where each array contains the bin limits for each component.
        """
        assert (
            projected_data.shape[1] == self.n_components
        ), "projected_data should have the same number of columns as n_components"
        assert self.n_bins > 0, "n_bins should be greater than 0"

        # Define bounds for each component  (n+1 bounds to divide each component in n bins)
        limits_bins = [
            np.linspace(
                np.min(projected_data[:, component]),
                np.max(projected_data[:, component]),
                self.n_bins + 1,
            )[
                1:-1
            ]  # Remove first and last limits to avoid attributing new bins to extreme values
            for component in range(self.n_components)
        ]
        return limits_bins

    def apply_pca(self, scaled_data):
        """Compute the mean seasonal cycle (MSC) of n samples and scale it between 0 and 1. Then apply the PCA already fit on the new data.  Each time step of the msc is considered as an independent component. Nb of time_step used for the PCA computation = 366 / step_msc.

        Args:
            n_samples (int, optional): number of samples to fit the PCA. Defaults to 10. to 5.
        """
        assert scaled_data.shape[1] == round(self.step_msc / 366) + 1
        assert scaled_data.shape[0] == self.n_samples_training
        X = self.pca.transform(scaled_data)
        return X

    # Function to find the box for multiple points
    def find_bins(self, projected_data, limits_bins):
        assert projected_data.shape[1] == len(limits_bins)

        box_indices = np.zeros(
            (projected_data.shape[0], projected_data.shape[1]), dtype=int
        )
        for i, limits_bin in enumerate(limits_bins):
            box_indices[:, i] = np.digitize(projected_data[:, i], limits_bin)

        return box_indices

    def apply_threshold():
        raise NotImplementedError()

    def save_experiment(self, args, limits_bins: list, min_data: int, max_data: int):
        assert self.saving_path is not None, "the saving path is missing"

        # Create the saving path if it does not exist
        date_of_today = datetime.datetime.today().strftime("%Y-%m-%-d_%H:%M:%S")
        self.saving_path = self.saving_path / f"{self.index}_{date_of_today}"

        self.saving_path.mkdir(parents=True, exist_ok=True)

        # Save args to a file for future reference
        args_path = self.saving_path / "args.json"
        with open(args_path, "w") as f:
            json.dump(args.__dict__, f, indent=4)

        # Save min_data and max_data
        min_max_data_path = self.saving_path / "min_max_data.json"
        with open(min_max_data_path, "w") as f:
            json.dump({"min_data": min_data, "max_data": max_data}, f, indent=4)

        # Save the PCA model
        pca_path = self.saving_path / "pca_matrix.pkl"
        with open(pca_path, "wb") as f:
            pk.dump(self.pca, f)

        # Save the limits of the bins
        assert (
            len(limits_bins) == self.n_components
        ), "the lenght of limits_bins list is not equal to the number of components"
        assert (
            limits_bins[0].shape[0] == self.n_bins - 1
        ), "the limits do not fit the number of bins"

        limits_bins_path = self.saving_path / "limits_bins.npy"
        np.save(limits_bins_path, limits_bins)
        return


class DatasetHandler:
    def __init__(
        self, index: str, n_samples: Union[int, None], step_msc: int, load_data: bool
    ):
        """
        Initialize DatasetHandler.

        Parameters:
        index (str): The climatic or ecological index to be processed.
        n_samples (Union[int, None]): Number of samples to select.
        load_data (bool): Flag to determine if data should be loaded during initialization.
        step_msc (int, optional): temporal resolution of the msc, to reduce computationnal workload. Defaults to 5.
        """

        self.index = index
        self.n_samples = n_samples
        self.step_msc = step_msc
        self.filepath = CLIMATIC_FILEPATH
        self.max_data = None
        self.min_data = None
        self.data = None
        if load_data:
            self.preprocess_data()

    def preprocess_data(self):
        """
        Preprocess data based on the index.
        """
        if self.index in ["pei_30", "pei_90", "pei_180"]:
            self.load_data()
            self.apply_climatic_transformations()
        else:
            raise NotImplementedError(
                "Index unavailable. Index available:\n -Climatic: 'pei_30', 'pei_90', 'pei_180'. \n Ecological: 'None."
            )

        # Select only a subset of the data if n_samples is specified
        if n_samples:
            self.randomly_select_data()
        else:
            print(f"computation on the entire dataset. {self.data.shape[0]} samples")

        self.compute_and_scale_the_msc()

    def load_data(self):
        """
        Load data from the specified filepath.

        Parameters:
        filepath (str): Path to the data file.
        """
        self.data = xr.open_zarr(self.filepath)[[self.index]]
        print("Data loaded from {}".format(self.filepath))

    def apply_climatic_transformations(self):
        """
        Apply transformations to the climatic data.
        """
        assert self.index in [
            "pei_30",
            "pei_90",
            "pei_180",
        ], "Index unavailable. Index available: 'pei_30', 'pei_90', 'pei_180'."

        assert self.data is not None, "Data not loaded."

        # Assert dimensions are as expected after loading and transformation
        assert all(
            dim in self.data.sizes for dim in ("time", "latitude", "longitude")
        ), "Dimension missing"

        assert (
            (self.data.longitude >= 0) & (self.data.longitude <= 360)
        ).all(), "Longitude values should be in the range 0 to 360"

        # Remove the year 1950 because the data are inconsistent
        self.data = self.data.sel(
            time=slice(datetime.date(1951, 1, 1), datetime.date(2022, 12, 31))
        )

        # Transform the longitude coordinates to -180 and 180
        def coordstolongitude(x):
            return ((x + 180) % 360) - 180

        self.data = self.data.roll(longitude=180 * 4, roll_coords=True)
        self.data = self.data.assign_coords(
            longitude=coordstolongitude(self.data.longitude)
        )

        self.data = self.data.stack(lonlat=("longitude", "latitude")).transpose(
            "lonlat", "time", ...
        )
        print(f"Climatic data loaded with dimensions: {self.data.sizes}")

    def randomly_select_data(self):
        """
        Randomly select a subset of the data based on n_samples.
        """
        lonlat_indices = random.choices(self.data.lonlat.values, k=self.n_samples)
        self.data = self.data.sel(lonlat=lonlat_indices)
        print(f"Randomly selected {self.n_samples} samples for training.")

    def compute_and_scale_the_msc(self):
        """
        compute the MSC of n samples and scale it between 0 and 1. Each time step of the msc is considered as an independent component. nb of time_step used for the PCA computation = 366 / step_msc.
        """
        assert (self.n_samples > 0) & (self.n_samples <= self.data.sizes["lonlat"])

        # Compute the MSC
        self.data["msc"] = (
            self.data[self.index]
            .groupby("time.dayofyear")
            .mean("time")
            .drop_vars(["lonlat", "longitude", "latitude"])
        )

        # Scale the data between 0 and 1
        # TODO Check if the data are in the file. especially after training pca
        if (self.max_data and self.min_data) is None:
            self.max_data = self.data[self.index].max().values
            self.min_data = self.data[self.index].min().values

        # reduce the temporal resolution
        self.data = self.data["msc"].isel(dayofyear=slice(1, 366, self.step_msc))

        # Scale the data between 0 and 1
        self.data = (self.min_data - self.data) / (self.max_data - self.min_data)
        return self.data, (self.min_data, self.max_data)


class EcologicalRegionalExtremes:
    def __init__(
        self,
        vegetation_index,
        # isclimatic: bool,
    ):
        self.vegetation_index = vegetation_index
        self.filepath = f"/Net/Groups/BGI/work_1/scratch/fluxcom/upscaling_inputs/MODIS_VI_perRegion061/{vegetation_index}/Groups_dyn_{vegetation_index}_MSC_snowfrac.zarr"
        self.filename = f"Groups_dyn_{vegetation_index}_MSC_snowfrac"

    def apply_transformations(self):
        # Load the MSC of MODIS
        ds = xr.open_zarr(self.filepath, consolidated=False)
        ds_msc = ds[self.filename].stack(
            {"lat": ["latchunk", "latstep_modis"], "lon": ["lonchunk", "lonstep_modis"]}
        )

        # Select k locations randomly to train the PCA:
        lat_indices = random.choices(ds_msc.lat.values, k=3000)
        lon_indices = random.choices(ds_msc.lon.values, k=3000)

        # Select the MSC of those locations:
        return


if __name__ == "__main__":
    args = argparse.parser.parse_args()
    print(args)
    import os

    os.exit()
    t1 = time.time()
    data_subset = DatasetHandler(
        index=args.index,
        n_samples=args.n_samples,
        step_msc=args.step_msc,
        load_data=args.load_data,
    )
    t2 = time.time()
    print(t2 - t1)
    climatic_processor = RegionalExtremes(
        index=args.index,
        step_msc=args.step_msc,
        n_components=args.n_components,
        n_bins=args.n_bins,
        saving_path=args.saving_path,
    )
    t3 = time.time()
    print(t3 - t2)
    projected_data = climatic_processor.compute_pca_and_transform(
        scaled_data=data_subset
    )

    # data_subset = DatasetHandler(
    #         index=self.index, n_samples=None, load_data=load_data
    #     )
    projected_data = self.apply_pca(scaled_data=data_subset)

    limits_bins = self.define_limits_bins(projected_data=projected_data)
    climatic_processor.load_data()
    t2 = time.time()
    print(t2 - t1)
    climatic_processor.apply_transformations()
    t3 = time.time()
    print(t3 - t2)
    climatic_processor.perform_pca_on_the_msc(n_samples=10, n_components=3)
    t4 = time.time()
    print(t4 - t3)
    climatic_processor.apply_pca(n_samples=20)
    t5 = time.time()

    # climatic_processor.compute_box_plot()
