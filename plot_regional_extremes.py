import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


from regional_extremes import SharedConfig
from regional_extremes import parser_arguments
from utils import printt


class PlotExtremes(SharedConfig):
    def __init__(
        self,
        config: SharedConfig,
    ):
        """
        Initialize PlotExtremes.

        """
        self.config = config
        self.pca_projection = None
        self.saving_path = self.config.path_load_model / "plots"
        self.saving_path.mkdir(parents=True, exist_ok=True)
        # self.projection = load_pca_projection

    def load_pca_projection(self):
        """
        Load data from the specified filepath.

        Parameters:
        filepath (str): Path to the data file.
        """
        projection_path = self.config.path_load_model / "pca_projection.zarr"
        self.pca_projection = xr.open_zarr(projection_path).pca
        printt("Projection loaded from {}".format(projection_path))

    def plot_map(self):
        # Reshape the data into a 2D array for each component
        red = self.pca_projection.isel(component=0).values
        green = self.pca_projection.isel(component=1).values
        blue = self.pca_projection.isel(component=2).values

        # Normalize the data to the range [0, 1]
        def _normalization(band):
            return (band - np.min(band)) / (np.max(band) - np.min(band))

        normalized_red = _normalization(red)
        normalized_green = _normalization(green)
        normalized_blue = _normalization(blue)

        # Stack the components into a 3D array
        rgb_normalized = np.dstack((normalized_red, normalized_green, normalized_blue))

        # Set up the map projection
        projection = ccrs.PlateCarree()

        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"projection": projection})

        # Add coastlines and set global extent
        ax.coastlines()
        ax.set_global()

        # Plot the RGB data
        img_extent = (
            self.pca_projection.longitude.min(),
            self.pca_projection.longitude.max(),
            self.pca_projection.latitude.min(),
            self.pca_projection.latitude.max(),
        )
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
        return


if __name__ == "__main__":
    args = parser_arguments().parse_args()

    args.path_load_model = "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2014047_2024-07-18_16:02:30"
    config = SharedConfig(args)

    plot = PlotExtremes(config=config)
    plot.load_pca_projection()
    plot.plot_map()
