import xarray as xr
import dask.array as da
from argparse import Namespace
import numpy as np
import json
import random
import datetime
from sklearn.decomposition import PCA
import pandas as pd
import pickle as pk
from pathlib import Path
from typing import Union
import time
import sys
import os


from utils import initialize_logger, printt, raiset, int_or_none
from loader_and_saver import Loader, Saver
from datahandler import create_handler
from config import InitializationConfig, CLIMATIC_INDICES, ECOLOGICAL_INDICES

np.set_printoptions(threshold=sys.maxsize)

from global_land_mask import globe

import argparse  # import ArgumentParser


# Argparser for all configuration needs
def parser_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--id",
        type=str,
        default=None,
        help="id of the experiment is time of the job launch and job_id",
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="name_of_the_experiment",
    )

    parser.add_argument(
        "--index",
        type=str,
        default="pei_180",
        help=" The climatic or ecological index to be processed (default: pei_180). "
        "Index available:\n -Climatic: 'pei_30', 'pei_90', 'pei_180'. \n Ecological: 'None.",
    )

    parser.add_argument(
        "--compute_variance",
        type=bool,
        default=False,
        help="Compute variance of the seasonal cycle in addition of the mean seasonal cycle (default: False).",
    )

    parser.add_argument(
        "--region",
        type=str,
        default="globe",
        help="Region of the globe to apply the regional extremes."
        "Region available: 'globe', 'europe'.",
    )
    parser.add_argument(
        "--time_resolution",
        type=int,
        default=5,
        help="time_resolution (int, optional): temporal resolution of the msc, to reduce computationnal workload (default: 5). ",
    )

    parser.add_argument(
        "--n_components", type=int, default=3, help="Number of component of the PCA."
    )

    parser.add_argument(
        "--n_samples",
        type=int_or_none,
        default=100,
        help="Select randomly n_samples. Use 'None' for no limit.",
    )

    parser.add_argument(
        "--n_bins",
        type=int,
        default=25,
        help="number of bins to define the regions of similar seasonal cycle.",
    )

    parser.add_argument(
        "--saving_path",
        type=str,
        default=None,
        help="Absolute path to save the experiments 'path/to/experiment'. "
        "If None, the experiment will be save in a folder /experiment in the parent folder.",
    )

    parser.add_argument(
        "--path_load_experiment",
        type=str,
        default=None,
        help="Path of the trained model folder.",
    )
    return parser


class RegionalExtremes:  # (InitializationConfig):
    def __init__(
        self,
        config: InitializationConfig,
        n_components: int,
        n_bins: int,
    ):
        """
        Compute the regional extremes by defining boxes of similar region using a PCA computed on the mean seasonal cycle of the samples. Each values of the msc is considered
        as an independent component.

        Args:
            config (InitializationConfig): Shared attributes across the classes.
            n_components (int): number of components of the PCA
            n_bins (int): Number of bins per component to define the boxes. Number of boxes = n_bins**n_components
        """
        self.config = config
        self.n_components = n_components
        self.n_bins = n_bins
        self.loader = Loader(config)
        self.saver = Saver(config)

        if self.config.path_load_experiment:
            # Load every variable if already available, otherwise return None.
            self.pca = self.loader._load_pca_matrix()
            self.projected_data = self.loader._load_pca_projection()
            self.limits_bins = self.loader._load_limits_bins()
            self.bins = self.loader._load_bins()

        else:
            # Initialize a new PCA.
            self.pca = PCA(n_components=self.n_components)
            self.projected_data = None
            self.limits_bins = None
            self.bins = None

    def compute_pca_and_transform(
        self,
        scaled_data,
    ):
        """compute the principal component analysis (PCA) on the mean seasonal cycle (MSC) of n samples and scale it between 0 and 1. Each time step of the msc is considered as an independent component. nb of time_step used for the PCA computation = 366 / time_resolution.

        Args:
            n_components (int, optional): number of components to compute the PCA. Defaults to 3.
            time_resolution (int, optional): temporal resolution of the msc, to reduce computationnal workload. Defaults to 5.
        """
        assert not hasattr(
            self.pca, "explained_variance_"
        ), "A pca already have been fit."
        assert self.config.path_load_experiment is None, "A model is already loaded."
        # sassert scaled_data.dayofyear.shape[0] == round(
        # s    366 / self.config.time_resolution
        # s)
        assert (self.n_components > 0) & (
            self.n_components <= 366
        ), "n_components have to be in the range of days of a years"
        # Fit the PCA. Each colomns give us the projection through 1 component.
        pca_components = self.pca.fit_transform(scaled_data)

        printt(
            f"PCA performed. sum explained variance: {sum(self.pca.explained_variance_ratio_)}. {self.pca.explained_variance_ratio_})"
        )

        # Save the PCA model
        pca_path = self.config.saving_path / "pca_matrix.pkl"
        with open(pca_path, "wb") as f:
            pk.dump(self.pca, f)
        printt(f"PCA saved: {pca_path}")

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
        if self.projected_data is not None:
            raiset(
                ValueError(
                    "self.projected_data is not None, projected_data already have been computed.",
                )
            )
        self._validate_scaled_data(scaled_data)

        transformed_data = xr.apply_ufunc(
            self.pca.transform,
            scaled_data.compute(),
            input_core_dims=[["dayofyear"]],  # Apply PCA along 'dayofyear'
            output_core_dims=[["component"]],  # Resulting dimension is 'component'
        )
        printt("Data are projected in the feature space.")

        self.saver._save_pca_projection(
            transformed_data, self.pca.explained_variance_ratio_
        )
        return transformed_data

    def _validate_scaled_data(self, scaled_data: np.ndarray) -> None:
        """Validates the scaled data to ensure it matches the expected shape."""
        if self.config.compute_variance:
            expected_shape = round(366 / self.config.time_resolution) * 2 + 1
        else:
            expected_shape = round(366 / self.config.time_resolution)
        if scaled_data.shape[1] != expected_shape:
            raiset(
                ValueError(
                    f"scaled_data should have {expected_shape} columns, but has {scaled_data.shape[1]} columns."
                )
            )

    def define_limits_bins(self) -> list[np.ndarray]:
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
        self._validate_inputs(self.projected_data)

        self.limits_bins = self._calculate_limits_bins(self.projected_data)
        self.saver._save_limits_bins(self.limits_bins)
        printt("Limits are computed and saved.")
        return self.limits_bins

    def _validate_inputs(self, projected_data: np.ndarray) -> None:
        """Validates the inputs for define_limits_bins."""
        if not hasattr(self.pca, "explained_variance_"):
            raiset(ValueError("PCA model has not been trained yet."))

        if projected_data.shape[1] != self.n_components:
            raiset(
                ValueError(
                    "projected_data should have the same number of columns as n_components"
                )
            )

        if self.n_bins <= 0:
            raiset(ValueError("n_bins should be greater than 0"))

    def _calculate_limits_bins(self, projected_data: np.ndarray) -> list[np.ndarray]:
        """Calculates the limits bins for each component."""
        return [
            np.linspace(
                np.quantile(projected_data[:, component], 0.05),
                np.quantile(projected_data[:, component], 0.95),
                round(self.pca.explained_variance_ratio_[component] * self.n_bins) + 1,
            )
            for component in range(self.n_components)
        ]

    # Function to find the box for multiple points
    def find_bins(self):
        assert self.projected_data.shape[1] == len(self.limits_bins)
        assert (
            len(self.limits_bins) == self.n_components
        ), "the lenght of limits_bins list is not equal to the number of components"

        box_indices = np.zeros(
            (self.projected_data.shape[0], self.projected_data.shape[1]), dtype=int
        )
        # defines boxes
        for i, limits_bin in enumerate(self.limits_bins):
            # get the indices of the bins to which each value in input array belongs.
            box_indices[:, i] = np.digitize(self.projected_data[:, i], limits_bin)
        self.saver._save_bins(box_indices, self.projected_data)
        self.bins = box_indices
        return box_indices

    def apply_threshold():
        raiset(NotImplementedError())


def main_train_pca(args):
    config = InitializationConfig(args)
    dataset_processor = create_handler(
        config=config, n_samples=args.n_samples  # args.n_samples,  # all the dataset
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
    dataset_processor = create_handler(config=config, n_samples=None)  # all the dataset
    data = dataset_processor.preprocess_data()

    projected_data = extremes_processor.apply_pca(scaled_data=data)
    limits_bins = extremes_processor.define_limits_bins(projected_data=projected_data)
    extremes_processor.find_bins(projected_data=projected_data, limits_bins=limits_bins)


def main_define_limits(args):
    args.path_load_experiment = "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2024-09-03_10:16:51_eco_final"
    config = InitializationConfig(args)

    dataset_processor = create_handler(config=config, n_samples=None)
    data = dataset_processor.preprocess_data()

    extremes_processor = RegionalExtremes(
        config=config,
        n_components=args.n_components,
        n_bins=args.n_bins,
    )
    projected_data = extremes_processor.apply_pca(scaled_data=data)
    limits_bins = extremes_processor.define_limits_bins(projected_data=projected_data)
    extremes_processor.find_bins(projected_data=projected_data, limits_bins=limits_bins)


def main_finds_bins(args):
    args.path_load_experiment = "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2024-09-03_10:16:51_eco_final"
    config = InitializationConfig(args)

    extremes_processor = RegionalExtremes(
        config=config,
        n_components=args.n_components,
        n_bins=args.n_bins,
    )
    extremes_processor.define_limits_bins()
    extremes_processor.find_bins()


if __name__ == "__main__":
    args = parser_arguments().parse_args()
    args.name = "eco_"
    args.index = "EVI"
    args.n_samples = 1000
    args.n_components = 3
    args.n_bins = 50
    args.compute_variance = False

    # args.path_load_experiment = "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2024-08-09_12:45:09_2139535_Europe_eco_small"
    # config = InitializationConfig(args)
    # extremes_processor = RegionalExtremes(
    #     config=config,
    #     n_components=args.n_components,
    #     n_bins=args.n_bins,
    # )
    # projected_data = extremes_processor.loader._load_pca_projection()
    # limits_bins = extremes_processor.define_limits_bins(projected_data=projected_data)
    # extremes_processor.find_bins(projected_data, limits_bins)

    # To train the PCA:
    main_train_pca(args)
    # To define the limits:
    # main_finds_bins(args)
