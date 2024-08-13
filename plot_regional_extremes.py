import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy
import random

from config import InitializationConfig
from loader import Loader, Saver
from regional_extremes import parser_arguments
from utils import printt
from datahandler import EcologicalDatasetHandler, ClimaticDatasetHandler


class PlotExtremes(InitializationConfig):
    def __init__(
        self,
        config: InitializationConfig,
    ):
        """
        Initialize PlotExtremes.

        """
        self.config = config
        self.saving_path = self.config.saving_path / "plots"
        self.saving_path.mkdir(parents=True, exist_ok=True)
        self.loader = Loader(config)

    def _load_pca_projection(self):
        """
        Load data from the specified filepath.

        Parameters:
        filepath (str): Path to the data file.
        """
        projection_path = self.config.saving_path / "pca_projection.zarr"
        data = xr.open_zarr(projection_path)
        self.pca_projection = data.pca
        self.explained_variance = data.explained_variance
        printt("Projection loaded from {}".format(projection_path))
        return self.pca_projection

    def _load_boxes(self):
        boxes_path = self.config.saving_path / "boxes.zarr"
        data = xr.open_zarr(boxes_path).bins
        data = data.stack(location=("longitude", "latitude"))
        return data

    def plot_map_component(self, normalization=False):
        self._load_pca_projection()
        # Normalize the explained variance
        normalized_variance = (
            self.explained_variance.explained_variance
            / self.explained_variance.explained_variance.sum()
        )

        # Normalize the data to the range [0, 1]
        def _normalization(index, normalization):
            band = self.pca_projection.isel(component=index).values

            normalized_band = (band - np.nanmin(band)) / (
                np.nanmax(band) - np.nanmin(band)
            )
            # We normalize with the 5% and 95% quantiles due to outliers.
            # normalized_band = (band - np.quantile(band, q=0.05)) / (
            #     np.quantile(band, q=0.95) - np.quantile(band, q=0.05)
            # )
            # # Normalization of the color by feature importance
            if normalization:
                normalized_band = (
                    normalized_band * normalized_variance.sel(component=index).values
                )

            return normalized_band

        normalized_red = _normalization(
            0, normalization=normalization
        )  # Red is the first component
        normalized_green = _normalization(
            1, normalization=normalization
        )  # Green is the second component
        normalized_blue = _normalization(
            2, normalization=normalization
        )  # blue is the third component

        # Stack the components into a 3D array
        rgb_normalized = np.dstack((normalized_red, normalized_green, normalized_blue))
        # Transpose the array
        rgb_normalized = np.transpose(rgb_normalized, (1, 0, 2))

        # Set up the map projection
        projection = cartopy.crs.PlateCarree()

        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(12, 5), subplot_kw={"projection": projection})

        # adjust the plot
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        # Add coastlines and set global extent
        ax.coastlines()
        ax.add_feature(cartopy.feature.OCEAN, zorder=100, edgecolor="k")

        # Plot the RGB data
        img_extent = (
            self.pca_projection.longitude.min(),
            self.pca_projection.longitude.max(),
            self.pca_projection.latitude.min(),
            self.pca_projection.latitude.max(),
        )

        ax.set_extent(img_extent, crs=projection)

        ax.imshow(
            rgb_normalized, origin="lower", extent=img_extent, transform=projection
        )

        # Add a title
        plt.title("RGB Components on Earth Map")

        map_saving_path = self.saving_path / "map_components_rgb.png"
        plt.savefig(map_saving_path)
        printt("Plot saved")

        # Show the plot
        plt.show()

    def plot_region(self, lon=None, lat=None):
        """plot the samples of a single region"""
        boxes = self._load_boxes()

        # Select randomly a first location
        if lon is None and lat is None:
            lon = random.choice(boxes.longitude).item()
            lat = random.choice(boxes.latitude).item()
        # Get the boxe indices of the location
        indices = boxes.sel(longitude=lon, latitude=lat).values

        indices = np.array([1, 1, 1])
        # Create a boolean mask for the subset
        mask = np.all(boxes.values == indices[:, np.newaxis], axis=0)

        # Get the lat and lon values where the mask is true
        masked_lons = boxes.longitude.values[mask]
        masked_lats = boxes.latitude.values[mask]
        masked_lons_lats = list(zip(masked_lons, masked_lats))
        printt(f"Number of samples in the region {indices}: {len(masked_lons_lats)}.")

        def plot_map_single_region(mask):
            # Get the subset of data using the mask
            subset = boxes.isel(location=mask)
            # Plot the subset
            # Set up the map projection
            projection = cartopy.crs.PlateCarree()

            # Create the figure and axis
            fig, ax = plt.subplots(
                figsize=(12, 5), subplot_kw={"projection": projection}
            )

            # adjust the plot
            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

            # Add coastlines and set global extent
            ax.coastlines()
            ax.add_feature(cartopy.feature.OCEAN, zorder=100, edgecolor="k")

            # Plot the points
            ax.scatter(
                subset.longitude,
                subset.latitude,
                transform=projection,
                color="red",
                s=20,
            )

            # Add a title
            plt.title(f"Location of samples of the region {indices}.")

            # Save the figure
            map_saving_path = (
                self.saving_path / f"map_location_single_region_{indices}.png"
            )
            plt.savefig(map_saving_path)
            plt.close()
            printt("Plot saved")

        def plot_time_series_single_region(masked_lons_lats):
            if len(masked_lons_lats) > 10:
                masked_lons_lats = random.choices(masked_lons_lats, k=10)
                printt(f"Selected locations: {masked_lons_lats}")

            # TODO Careful Dataset dependent
            dataloader = EcologicalDatasetHandler(
                config=config, n_samples=None  # args.n_samples,  # all the dataset
            )
            data = (
                dataloader._dataset_specific_loading()
                .stack(location=("longitude", "latitude"))
                .to_array()
                .squeeze(axis=0)
            )
            data = data.sel(location=masked_lons_lats)
            data = data.chunk({"time": len(data.time), "location": 1})
            # Remove NaNs

            def plot_ts(data):
                # Create a figure and axis
                fig, ax = plt.subplots(figsize=(12, 8))
                # Plot each location
                for loc in range(data.location.size):
                    ax.plot(
                        data.time, data.isel(location=loc), label=f"Location {loc+1}"
                    )

                # Customize the plot
                ax.set_xlabel("Time")
                ax.set_ylabel(f"{self.config.index}")
                ax.set_title(f"Time series of locations in the region {indices}")
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

                # Adjust layout to prevent cutting off the legend
                plt.tight_layout()

                # Show the plot
                plt.show()
                saving_path = self.saving_path / f"ts_single_region_{indices}.png"
                plt.savefig(saving_path)

            def plot_msc_vsc(data):
                msc = data.groupby("time.dayofyear").mean("time", skipna=True)
                msc = msc.chunk({"dayofyear": len(msc.dayofyear), "location": 1})
                vsc = data.groupby("time.dayofyear").var("time", skipna=True)
                vsc = vsc.chunk({"dayofyear": len(vsc.dayofyear), "location": 1})
                # Create a figure and axis
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

                # Plot each location for the first variable
                for loc in range(data.location.size):
                    ax1.plot(
                        msc.dayofyear,
                        msc.isel(location=loc),  # .transpose(),
                        label=f"Location {loc+1}",
                    )
                    ax2.plot(
                        vsc.dayofyear,
                        vsc.isel(location=loc),  # .transpose(),
                        label=f"Location {loc+1}",
                    )

                # Customize the plot
                ax1.set_xlabel("Day of Year")
                ax1.set_ylabel("Mean Seasonal Cycle")
                ax1.set_title(f"MSC or Each Location of the region {indices}")
                ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

                ax2.set_xlabel("Day of Year")
                ax2.set_ylabel("Variance Seasonal Cycle")
                ax2.set_title(f"VSC for Each Location of the region {indices}")
                ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

                plt.tight_layout()
                plt.subplots_adjust(hspace=0.3)

                saving_path = self.saving_path / f"msc_vsc_single_region_{indices}.png"
                plt.savefig(saving_path)

                plt.show()

            plot_ts(data)
            plot_msc_vsc(data)

        plot_map_single_region(mask)
        plot_time_series_single_region(masked_lons_lats)

    def plot_boxes_msc(self):
        # find_boxes(pca_components, pca_bins)
        pca_projection = self._load_pca_projection()
        pca_projection = pca_projection.stack(location=("longitude", "latitude"))
        n_bins = self.config.n_bins
        box_indices = self._load_boxes()
        # Convert box indices to RGB colors
        # Normalize indices to the range [0, 1] for RGB
        colors = box_indices / (n_bins + 1)
        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Scatter plot
        sc = ax.scatter(
            pca_projection.isel(component=0).values.T,
            pca_projection.isel(component=1).values.T,
            pca_projection.isel(component=2).values.T,
            c=colors.values.T,
            s=50,
            edgecolor="k",
        )

        # Adding labels and title
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.set_zlabel("PCA Component 3")
        ax.set_title("3D PCA Projection with RGB Colors")
        ax.legend()
        plt.show()
        return


if __name__ == "__main__":
    args = parser_arguments().parse_args()

    args.path_load_experiment = "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2024-08-09_12:45:09_2139535_Europe_eco_small"
    config = InitializationConfig(args)

    plot = PlotExtremes(config=config)
    plot.plot_boxes_msc()
    # plot.plot_map_component()
