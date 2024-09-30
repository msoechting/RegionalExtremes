import os

from config import InitializationConfig
import numpy as np
import xarray as xr
import pickle as pk
from utils import printt

CLIMATIC_FILEPATH = "/Net/Groups/BGI/scratch/mweynants/DeepExtremes/v3/PEICube.zarr"
ECOLOGICAL_FILEPATH = (
    lambda index: f"/Net/Groups/BGI/work_1/scratch/fluxcom/upscaling_inputs/MODIS_VI_perRegion061/{index}/Groups_{index}gapfilled_QCdyn.zarr"
)
VARIABLE_NAME = lambda index: f"{index}gapfilled_QCdyn"


class Loader:
    def __init__(
        self,
        config: InitializationConfig,
    ):
        self.config = config

    def _load_pca_matrix(self):
        """
        Load PCA matrix from the file.
        """
        pca_path = self.config.saving_path / "pca_matrix.pkl"
        with open(pca_path, "rb") as f:
            pca = pk.load(f)
        return pca

    def _load_pca_projection(self, explained_variance=False):
        """
        Load data from the specified filepath.

        Parameters:
        filepath (str): Path to the data file.
        """
        projection_path = self.config.saving_path / "pca_projection.zarr"
        if not os.path.exists(projection_path):
            printt(f"PCA projection not found at {projection_path}")
            return None

        data = xr.open_zarr(projection_path)
        pca_projection = data.pca.stack(location=("longitude", "latitude")).transpose(
            "location", "component", ...
        )
        # Remove NaNs
        condition = ~pca_projection.isnull().any(dim="component").compute()
        pca_projection = pca_projection.where(condition, drop=True)
        printt("Projection loaded from {}".format(projection_path))
        if explained_variance:
            return pca_projection, data.explained_variance
        else:
            return pca_projection

    def _load_limits_bins(self) -> list[np.ndarray]:
        """Loads the limits bins from a file."""
        limits_bins_path = self.config.saving_path / "limits_bins.npz"
        if not os.path.exists(limits_bins_path):
            print(f"Limits bins not found at {limits_bins_path}")
            return None
        data = np.load(limits_bins_path)
        limits_bins = [data[f"arr_{i}"] for i in range(len(data.files))]
        print("Limits bins loaded.")
        return limits_bins

    def _load_bins(self):
        bins_path = self.config.saving_path / "bins.zarr"
        if not os.path.exists(bins_path):
            bins_path = self.config.saving_path / "boxes.zarr"
            if not os.path.exists(bins_path):
                printt(f"Bins not found.")
                return None
            Warning(
                'boxes.zarr is an inconsistent legacy name, change it for "bins.zarr"'
            )

        data = xr.open_zarr(bins_path)
        data = data.bins.stack(location=("longitude", "latitude")).transpose(
            "location", "component", ...
        )
        condition = ~data.isnull().any(dim="component").compute()
        data = data.where(condition, drop=True)
        printt("Bins loaded.")
        return data

    def _load_extremes(self):
        """Saves the extremes quantile to a file."""
        extremes_path = self.config.saving_path / "extremes_1.zarr"
        if not os.path.exists(extremes_path):
            raise FileNotFoundError(f"The file {extremes_path} not found.")
        extremes = xr.open_zarr(extremes_path)
        # Unstack location for longitude and latitude as dimensions
        extremes = extremes.stack(location=["longitude", "latitude"])
        printt("Extremes loaded.")
        return extremes


class Saver:
    def __init__(
        self,
        config: InitializationConfig,
    ):
        self.config = config

    def _save_pca_projection(self, pca_projection, explained_variance_ratio_) -> None:
        """Saves the limits bins to a file."""
        # Split the components into separate DataArrays
        # Create a new coordinate for the 'component' dimension
        component = np.arange(pca_projection.shape[1])

        # Create the new DataArray
        pca_projection = xr.DataArray(
            data=pca_projection.values,
            dims=["location", "component"],
            coords={
                "location": pca_projection.location,
                "component": component,
            },
            name="pca",
        )
        # Unstack location for longitude and latitude as dimensions
        pca_projection = pca_projection.set_index(
            location=["longitude", "latitude"]
        ).unstack("location")

        # Explained variance for each component
        explained_variance = xr.DataArray(
            explained_variance_ratio_,
            dims=["component"],
            coords={"component": component},
        )
        pca_projection["explained_variance"] = explained_variance

        # Saving path
        pca_projection_path = self.config.saving_path / "pca_projection.zarr"

        if os.path.exists(pca_projection_path):
            raise FileExistsError(
                f"The file {pca_projection_path} already exists. Rewriting is not allowed."
            )

        # Saving the data
        pca_projection.to_zarr(pca_projection_path)
        printt("Projection saved.")

    def _save_limits_bins(self, limits_bins: list[np.ndarray]) -> None:
        """Saves the limits bins to a file."""
        limits_bins_path = self.config.saving_path / "limits_bins.npz"
        if os.path.exists(limits_bins_path):
            raise FileExistsError(
                f"The file {limits_bins_path} already exists. Rewriting is not allowed."
            )
        np.savez(limits_bins_path, *limits_bins)
        print(f"Limits bins saved to {limits_bins_path}")

    def _save_bins(self, boxes_indices, projected_data):
        """Saves the bins to a file."""

        # Create the new DataArray
        component = np.arange(boxes_indices.shape[1])

        boxes_indices = xr.DataArray(
            data=boxes_indices,
            dims=["location", "component"],
            coords={
                "location": projected_data.location,
                "component": component,
            },
            name="bins",
        )

        # Unstack location for longitude and latitude as dimensions
        boxes_indices = boxes_indices.set_index(
            location=["longitude", "latitude"]
        ).unstack("location")

        bins_path = self.config.saving_path / "boxes.zarr"
        if os.path.exists(bins_path):
            raise FileExistsError(
                f"The file {bins_path} already exists. Rewriting is not allowed."
            )
        boxes_indices.to_zarr(bins_path)
        printt("Boxes computed and saved.")

    def _save_extremes(self, extremes):
        """Saves the extremes quantile to a file."""

        # Unstack location for longitude and latitude as dimensions
        extremes = extremes.set_index(location=["longitude", "latitude"]).unstack(
            "location"
        )

        extremes_path = self.config.saving_path / "extremes.zarr"
        if os.path.exists(extremes_path):
            raise FileExistsError(
                f"The file {extremes_path} already exists. Rewriting is not allowed."
            )
        extremes.to_zarr(extremes_path)
        printt("Extremes computed and saved.")
