import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy


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

    def load_pca_projection(self):
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

    def load_bins(self):
        projection_path = self.config.saving_path / "boxes.zarr"
        data = 

    def plot_map_component(self):
        # Normalize the explained variance
        normalized_variance = (
            self.explained_variance.explained_variance
            / self.explained_variance.explained_variance.sum()
        )

        # Normalize the data to the range [0, 1]
        def _normalization(index, normalization=False):
            band = self.pca_projection.isel(component=index).values

            # (band - np.min(band)) / (np.max(band) - np.min(band))
            # We normalize with the 5% and 95% quantiles due to outliers.
            normalized_band = (band - np.quantile(band, q=0.05)) / (
                np.quantile(band, q=0.95) - np.quantile(band, q=0.05)
            )
            # We normalize the color by feature importance
            if normalization:
                normalized_band = (
                    normalized_band * normalized_variance.sel(component=index).values
                )

            return normalized_band

        normalized_red = _normalization(0)  # Red is the first component
        normalized_green = _normalization(1)  # Green is the second component
        normalized_blue = _normalization(2)  # blue is the third component

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

    def plot_region(self):
        """plot the samples of a single region"""

        def _get_random_coordinates(self):
            lon_index = random.randint(0, self.data.longitude.sizes["longitude"] - 1)
            lat_index = random.randint(0, self.data.latitude.sizes["latitude"] - 1)
            return (
                self.data.longitude[lon_index].item(),
                self.data.latitude[lat_index].item(),
            )

        lon, lat = self._get_random_coordinates()

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

    args.path_load_experiment = "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2024-08-07_13:48:38_eco"
    config = InitializationConfig(args)

    plot = PlotExtremes(config=config)
    plot.load_pca_projection()
    plot.plot_map_component()
