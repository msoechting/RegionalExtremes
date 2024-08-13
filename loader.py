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

    def _load_pca_projection(self):
        """
        Load data from the specified filepath.

        Parameters:
        filepath (str): Path to the data file.
        """
        projection_path = self.config.saving_path / "pca_projection.zarr"
        data = xr.open_zarr(projection_path)
        pca_projection = data.pca.stack(location=("longitude", "latitude")).transpose(
            "location", "component", ...
        )
        printt("Projection loaded from {}".format(projection_path))
        return pca_projection

    def _load_limits_bins(self):
        """Saves the limits bins to a file."""
        limits_bins_path = self.config.saving_path / "limits_bins.npy"
        data = np.load(limits_bins_path)
        printt("Limits bins loaded.")
        return data

    def _load_bins(self):
        bins_path = self.config.saving_path / "bins.zarr"
        data = xr.open_zarr(boxes_path)
        data = data.bins.stack(location=("longitude", "latitude")).transpose(
            "location", "component", ...
        )
        return data


class Saver:
    def __init__(
        self,
        config: InitializationConfig,
    ):
        self.config = config

    def _save_pca_projection(self, pca_projection) -> None:
        """Saves the limits bins to a file."""
        # Split the components into separate DataArrays
        # Create a new coordinate for the 'component' dimension
        component = np.arange(self.n_components)

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
            self.pca.explained_variance_ratio_,  # Example values for explained variance
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
        limits_bins_path = self.config.saving_path / "limits_bins.npy"
        if os.path.exists(limits_bins_path):
            raise FileExistsError(
                f"The file {limits_bins_path} already exists. Rewriting is not allowed."
            )
        np.save(limits_bins_path, limits_bins)

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
