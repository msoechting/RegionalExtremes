import unittest
import xarray as xr
import numpy as np
from unittest.mock import patch
import matplotlib.pyplot as plt
from io import BytesIO

from regional_extremes import ClimaticRegionalExtremes


class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        """Set up mock datasets for testing."""
        lon = np.array([0, 1])
        lat = np.array([0, 1])

        latchunk = np.array([0, 1])
        latstep_modis = np.array([0, 1])
        lontchunk = np.array([0, 1])
        lonstep_modis = np.array([0, 1])

        time = np.array([0, 1, 2])
        climatic_data = np.random.random((2, 2, 3))
        ecological_data = np.random.random((2, 2, 2, 2, 3))

        self.climatic_data = xr.DataArray(
            climatic_data,
            coords=[lon, lat, time],
            dims=["longitude", "latitude", "time"],
        )
        self.vegetation_data = xr.DataArray(
            ecological_data,
            coords=[latchunk, latstep_modis, lontchunk, lonstep_modis, time],
            dims=["latchunk", "latstep_modis", "lontchunk", "lonstep_modis", "time"],
        )

        self.climatic_filepath = "mock_climatic.zarr"
        self.vegetation_filepath = "mock_vegetation.zarr"

        # Mock the open_dataset method to return the mock datasets
        self.mock_open_dataset = patch("xarray.open_dataset").start()
        self.mock_open_dataset.side_effect = lambda filepath: {
            self.climatic_filepath: xr.Dataset({"data": self.climatic_data}),
            self.vegetation_filepath: xr.Dataset({"data": self.vegetation_data}),
        }[filepath]

    def tearDown(self):
        """Stop all patches."""
        patch.stopall()


def test_climatic_regional_extremes(self):
    processor = ClimaticRegionalExtremes(self.climatic_filepath)
    processor.load_data()
    self.assertIsNotNone(processor.data)
    self.assertEqual(processor.data.dims, {"lon": 2, "lat": 2, "time": 3})

    processor.apply_transformations()
    self.assertIsNotNone(processor.transformed_data)
    self.assertEqual(processor.transformed_data.dims, {"lonlat": 4, "time": 3})

    processor.perform_pca(n_components=2)
    self.assertIsNotNone(processor.pca_result)
    self.assertEqual(processor.pca_result.shape[1], 2)

    # Mock plt.show() to avoid displaying the plot during tests
    with patch("matplotlib.pyplot.show"):
        processor.compute_box_plot()
