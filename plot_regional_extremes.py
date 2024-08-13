import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy
import random

from config import InitializationConfig
from regional_extremes import parser_arguments
from utils import printt


class PlotExtremes(InitializationConfig):
    def __init__(
        self,
        config: InitializationConfig,
    ):
        """
        Initialize PlotExtremes.

        """
        self.config = config
        self.pca_projection = None
        self.saving_path = self.config.saving_path / "plots"
        self.saving_path.mkdir(parents=True, exist_ok=True)
        # self.projection = load_pca_projection

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

        # Create a boolean mask for the subset
        mask = np.all(boxes.values == indices[:, np.newaxis], axis=0)

        # Get the subset of data using the mask
        subset = boxes.isel(location=mask)

        # Plot the subset
        # Set up the map projection
        projection = cartopy.crs.PlateCarree()

        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(12, 5), subplot_kw={"projection": projection})

        # adjust the plot
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        # Add coastlines and set global extent
        ax.coastlines()
        ax.add_feature(cartopy.feature.OCEAN, zorder=100, edgecolor="k")

        # Plot the points
        ax.scatter(
            subset.longitude, subset.latitude, transform=projection, color="red", s=20
        )

        # Add a title
        plt.title(f"Location of samples of the region {indices}.")

        # Save the figure
        map_saving_path = self.saving_path / f"map_location_single_region_{indices}.png"
        plt.savefig(map_saving_path)
        plt.close()
        printt("Plot saved")

        pei = ds_pei.sel(
            latitude=lat_indices[300],
            longitude=lon_indices[300],
            time=slice(datetime.date(1951, 1, 1), datetime.date(2022, 12, 31)),
        ).pei_30

        # Group by day of the year and calculate required statistics
        mean_pei = pei.groupby("time.dayofyear").var("time")
        # quantile_05_pei = pei.chunk(dict(time=-1)).groupby("time.dayofyear").quantile(q=0.05, dim="time")
        # quantile_95_pei = pei.chunk(dict(time=-1)).groupby("time.dayofyear").quantile(q=0.95, dim="time")
        # min_pei = pei.groupby("time.dayofyear").min("time")
        # max_pei = pei.groupby("time.dayofyear").max("time")

        # Plot the results
        plt.figure(figsize=(10, 6))
        mean_pei.plot(label="Variance PEI")
        # quantile_05_pei.plot(label='5th Percentile PEI')
        # quantile_95_pei.plot(label='95th Percentile PEI')
        # min_pei.plot(label='Min PEI')
        # max_pei.plot(label='Max PEI')

        # Add labels and title
        plt.xlabel("Day of the Year")
        plt.ylabel("PEI-180")
        # plt.title('PEI Statistics by Day of the Year')
        plt.legend()
        plt.savefig(
            "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/results/plots/variance/plot.png"
        )

    def plot_boxes_msc(box_indices, n_bins):
        # find_boxes(pca_components, pca_bins)
        # Convert box indices to RGB colors
        # Normalize indices to the range [0, 1] for RGB
        norm_box_indices = box_indices / (n_bins + 1)
        colors = norm_box_indices

        # Check that the colors are within the 0-1 range
        print("Minimum color value:", colors.min())
        print("Maximum color value:", colors.max())

        # Ensure all values are within the [0, 1] range
        assert (
            colors.min() >= 0 and colors.max() <= 1
        ), "RGBA values should be within the 0-1 range"

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Scatter plot
        sc = ax.scatter(
            pca_components[:, 0],
            pca_components[:, 1],
            pca_components[:, 2],
            c=colors,
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
    plot.plot_region()
    # plot.plot_map_component()
