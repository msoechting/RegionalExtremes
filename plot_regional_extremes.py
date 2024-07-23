import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


class PlotExtremes(SharedConfig):

    def load_data(self, filepath):
        """
        Load data from the specified filepath.

        Parameters:
        filepath (str): Path to the data file.
        """
        # chunk_sizes = {"time": -1, "latitude": 100, "longitude": 100}
        self.data = xr.open_zarr(filepath)[[self.config.index]]
        printt("Data loaded from {}".format(filepath))

    def plot_map(self):
        # Reshape the data into a 2D array for each component
        red = ds.isel(components=0).values.reshape(len(ds.lat), len(ds.lon))
        green = ds.isel(components=1).values.reshape(len(ds.lat), len(ds.lon))
        blue = ds.isel(components=2).values.reshape(len(ds.lat), len(ds.lon))

        # Stack the components into a 3D array
        rgb = np.dstack((red, green, blue))

        # Normalize the data to the range [0, 1]
        rgb_normalized = (rgb - rgb.min()) / (rgb.max() - rgb.min())
