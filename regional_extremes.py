import xarray as xr
import numpy as np
import random
import datetime
from sklearn.decomposition import PCA

from pathlib import Path
from typing import Union
import time

CLIMATIC_FILEPATH = "/Net/Groups/BGI/scratch/mweynants/DeepExtremes/v3/PEICube.zarr"


class RegionalExtremes:
    def __init__(
        self,
        filepath: Union[Path, str],
        index: str,
        step_msc,
        n_components=3,
        n_samples_training=100,
    ):
        """Base Class to compute regional extremes.
        Args:
            filepath (Union[Path, str]): filepath to load the data
            index (str): vegetation index or climatic index. Variable used to define the extreme.
            step_msc (_type_): temporal resolution of the mean seasonal cycle. nb of points = 366 / step_msc
        """

        self.filepath = filepath
        self.index = index
        self.step_msc = step_msc
        self.data = None

        self.min_data = None
        self.max_data = None
        self.n_samples_training = n_samples_training
        self.n_components = n_components
        self.pca = None

    def load_data(self):
        # Load the PEI-* drought indices
        self.data = xr.open_zarr(self.filepath)[[self.index]]

    def apply_transformations(self):
        raise NotImplementedError("Subclasses should implement this method")

    def compute_and_scale_the_msc(self, n_samples=10, step_msc=5):
        """compute the MSC of n samples and scale it between 0 and 1. Each time step of the msc is considered as an independent component. nb of time_step used for the PCA computation = 366 / step_msc.

        Args:
            n_samples (int, optional): number of samples to fit the PCA. Defaults to 10.
            n_components (int, optional): number of components to compute the PCA. Defaults to 3.
            step_msc (int, optional): temporal resolution of the msc, to reduce computationnal workload. Defaults to 5.
        """
        assert (n_samples > 0) & (
            n_samples
            <= self.data.sizes[
                "lonlat"
            ]  # self.data.sizes["longitude"] * self.data.sizes["latitude"]
        )
        # Select randomly n_samples to train the PCA:
        # TODO for elecological, select valid values
        if n_samples is not None:
            lonlat_indices = random.choices(self.data.lonlat.values, k=n_samples)
            selected_data = self.data.sel(lonlat=lonlat_indices)
        else:
            selected_data = self.data

        # Compute the MSC
        selected_data["msc"] = (
            selected_data[self.index]
            .groupby("time.dayofyear")
            .mean("time")
            .drop_vars(["lonlat", "longitude", "latitude"])
        )

        # reduce the temporal resolution
        msc_small = selected_data["msc"].isel(dayofyear=slice(1, 366, step_msc))

        # Scale the data between 0 and 1
        if (self.max_data and self.min_data) is None:
            self.max_data = self.data[self.index].max().values
            self.min_data = self.data[self.index].min().values
        scaled_data = (self.min_data - msc_small) / (self.max_data - self.min_data)
        return scaled_data

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
        self.pca.fit(scaled_data)
        print(
            f"PCA performed. sum explained variance: {sum(self.pca.explained_variance_ratio_)}"
        )

        # Each colomns give us the projection through 1 component.
        pca_components = self.pca.transform(scaled_data)
        return pca_components

    def apply_pca(self, scaled_data, n_samples=None):
        """Compute the mean seasonal cycle (MSC) of n samples and scale it between 0 and 1. Then apply the PCA already fit on the new data.  Each time step of the msc is considered as an independent component. Nb of time_step used for the PCA computation = 366 / step_msc.

        Args:
            n_samples (int, optional): number of samples to fit the PCA. Defaults to 10.
            step_msc (int, optional): temporal resolution of the msc, to reduce computationnal workload. Defaults to 5.
        """
        X = self.pca.transform(scaled_data)
        return X

    def define_limits_bins(self, projected_data, n_bins=25):
        """
            Define the bound of each bins on the projected data. The function
            should be apply on the largest possible  amount of data to capture
            the distribution in the projected space (especially minimum and maximum).
            So ideally, fit the PCA with a subset of the data, then project the full dataset,
            then define the bins on the full dataset projected.
            n_bins is per components, so number of boxes = n_bins**n_components
        Args:
            projected_data (Xarray): Data projected after the PCA.
            n_bins (int, optional): number of bins for each components. Defaults to 25.

        Returns:
            list of Xarray: list of array, where each array contains the boxes limits for each components.
        """
        assert projected_data.shape[1] == self.n_components

        # Define n+1 bounds to divide each component in n bins,
        # Then remove the first and last limits so that the box is not constrained at the extremities.
        # This prevents defining later a new box for a sample that is more extreme in any of the components.
        limits_bins = [
            np.linspace(
                np.min(projected_data[:, component]),
                np.max(projected_data[:, component]),
                n_bins + 1,
            )[1:-1]
            for component in range(self.n_components)
        ]
        return limits_bins

    # Function to find the box for multiple points
    def find_bins(self, projected_data, limits_bins):

        assert projected_data.shape[1] == len(limits_bins)

        box_indices = np.zeros(
            (projected_data.shape[0], projected_data.shape[1]), dtype=int
        )
        for i, limits_bin in enumerate(limits_bins):
            box_indices[:, i] = np.digitize(projected_data[:, i], limits_bin)

        return box_indices


class ClimaticRegionalExtremes(RegionalExtremes):
    def __init__(self, index, step_msc, n_components, n_samples_training):
        super().__init__(self, index, step_msc, n_components, n_samples_training)
        assert index in [
            "pei_30",
            "pei_90",
            "pei_180",
        ], "index unavailable. Index available:'pei_30', 'pei_90', 'pei_180'."
        assert (self.step_msc > 0) & (
            self.step_msc <= 366
        ), "step_msc have to be in the range of days of a years."
        self.filepath = CLIMATIC_FILEPATH
        self.index = index
        self.step_msc = step_msc
        self.n_components = n_components
        # self.data = None

    def apply_transformations(self):
        assert self.data is not None
        # Assert dimensions are as expected after loading and transformation
        assert all(
            dim in self.data.dims for dim in ("time", "latitude", "longitude")
        ), "Dimension missing"
        assert (
            (self.data.longitude >= 0) & (self.data.longitude <= 360)
        ).all(), "This function transform longitude values from 0, 360 to the range -180 to 180"

        # Remove the year 1950 because the data are inconsistent
        self.data = self.data.sel(
            time=slice(datetime.date(2017, 1, 1), datetime.date(2022, 12, 31))
        )

        # Transform the longitude coordinates to -180 and 180
        def coordstolongitude(x):
            return ((x + 180) % 360) - 180

        dsc = self.data.roll(longitude=180 * 4, roll_coords=True)
        ds_pei = dsc.assign_coords(longitude=coordstolongitude(dsc.longitude))

        ds_pei = ds_pei.stack(lonlat=("longitude", "latitude")).transpose(
            "lonlat", "time", ...
        )

        self.data = ds_pei
        print(f"Climatic datas loaded with dimensions: {self.data.dims}")


class EcologicalRegionalExtremes(RegionalExtremes):
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
    # For climatic data
    t0 = time.time()
    climatic_processor = ClimaticRegionalExtremes(index="pei_180")
    t1 = time.time()
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
    print(t5 - t4)

    # climatic_processor.compute_box_plot()
