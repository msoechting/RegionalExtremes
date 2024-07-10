import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import xarray as xr
import numpy as np
import datetime

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from regional_extremes import CLIMATIC_FILEPATH
CLIMATIC_FILEPATH = "/Net/Groups/BGI/scratch/mweynants/DeepExtremes/v3/PEICube.zarr"
from regional_extremes import RegionalExtremes, ClimaticRegionalExtremes


class TestRegionalExtremes(unittest.TestCase):
    def setUp(self):
        """Set up mock datasets for testing."""

        lon = np.arange(0, 360, 50.5)
        lat = np.arange(-90, 90, 50.5)

        lonchunk = np.arange(-179, 179, 50.5)
        latchunck = np.arange(-89, 89, 50.5)

        latstep_modis = np.arange(-0.975, 0.975, 0.5)
        lonstep_modis = np.arange(-0.975, 0.975, 0.5)

        time = np.array(
            [
                np.datetime64("1950-01-01"),
                np.datetime64("1951-01-01"),
                np.datetime64("1979-01-01"),
                np.datetime64("2022-12-31"),
            ]
        )
        climatic_data = np.random.random((len(lon), len(lat), len(time)))

        self.mock_climatic_dataset = xr.Dataset(
            {"data": (["longitude", "latitude", "time"], climatic_data)},
            coords={"longitude": lon, "latitude": lat, "time": time},
        )

        ecological_data = np.random.random(
            (
                len(lonchunk),
                len(lonstep_modis),
                len(latchunck),
                len(latstep_modis),
                len(time),
            )
        )

        self.vegetation_data = xr.DataArray(
            ecological_data,
            coords=[lonchunk, lonstep_modis, latchunck, latstep_modis, time],
            dims=["lonchunk", "lonstep_modis", "latchunck", "latstep_modis", "time"],
        )

        # Mock the open_dataset method to return the mock datasets
        self.mock_dataset = patch("xarray.open_dataset").start()
        self.mock_dataset.side_effect = lambda filepath: {
            self.climatic_filepath: xr.Dataset({"data": self.climatic_data}),
            self.vegetation_filepath: xr.Dataset({"data": self.vegetation_data}),
        }[filepath]

    def tearDown(self):
        """Stop all patches."""
        patch.stopall()


class TestClimaticRegionalExtremes(TestRegionalExtremes):
    def test_load_data_climatic(self):
        processor = ClimaticRegionalExtremes()
        processor.load_data()
        # Check the data is not empty
        self.assertIsNotNone(processor.data)

    def test_apply_transformations_true_data(self):
        processor = ClimaticRegionalExtremes()
        processor.load_data()

        expected_len_lonlat = (
            processor.data.dims["longitude"] * processor.data.dims["latitude"]
        )

        processor.apply_transformations()

        # Check if the longitude values are within the range -180 to 180
        self.assertTrue(
            (
                (processor.data.longitude >= -180) & (processor.data.longitude <= 180)
            ).all(),
            "Longitude values are not within the range -180 to 180",
        )
        # Check if the latitude values are within the range -90 to 90
        self.assertTrue(
            ((processor.data.latitude >= -90) & (processor.data.latitude <= 90)).all(),
            "Latitude values are not within the range -180 to 180",
        )

        # Check if the first datetime is after 1951
        self.assertGreaterEqual(
            processor.data.time.values[0], np.datetime64("1951-01-01")
        )

        # Check if the dimensions are stacked correctly
        self.assertIn("lonlat", processor.data.dims)
        self.assertIn("time", processor.data.dims)
        self.assertEqual(processor.data.dims["lonlat"], expected_len_lonlat)

    # @patch("xarray.open_zarr")
    def test_apply_transformations_mock_data(self):
        processor = ClimaticRegionalExtremes()

        # Replace the data by the mock dataset
        processor.data = self.mock_climatic_dataset

        # Apply the transformations
        processor.apply_transformations()

        # Check if the longitude transformation is correctly applied
        expected_longitudes = np.sort(
            ((self.mock_climatic_dataset.longitude + 180) % 360) - 180
        )
        np.testing.assert_array_equal(
            np.unique(processor.data.lonlat.longitude.values), expected_longitudes
        )

        # Check if the longitude values are within the range -180 to 180
        self.assertTrue(
            (
                (processor.data.longitude >= -180) & (processor.data.longitude <= 180)
            ).all(),
            "Longitude values are not within the range -180 to 180",
        )
        # Check if the latitude values are within the range -90 to 90
        self.assertTrue(
            ((processor.data.latitude >= -90) & (processor.data.latitude <= 90)).all(),
            "Latitude values are not within the range -180 to 180",
        )

        # Check if the correct time range is selected
        expected_times = np.array(
            [
                np.datetime64("1951-01-01"),
                np.datetime64("1979-01-01"),
                np.datetime64("2022-12-31"),
            ]
        )
        np.testing.assert_array_equal(processor.data.time, expected_times)

        # Check if the dimensions are stacked correctly
        self.assertIn("lonlat", processor.data.dims)
        self.assertIn("time", processor.data.dims)

        # Check if the dimensions lens are correct
        expected_len_lonlat = (
            self.mock_climatic_dataset.dims["longitude"]
            * self.mock_climatic_dataset.dims["latitude"]
        )
        self.assertEqual(processor.data.dims["lonlat"], expected_len_lonlat)
        self.assertEqual(processor.data.dims["time"], len(expected_times))


if __name__ == "__main__":
    unittest.main()

"""

    processor.perform_pca(n_components=2)
    self.assertIsNotNone(processor.pca_result)
    self.assertEqual(processor.pca_result.shape[1], 2)

    # Mock plt.show() to avoid displaying the plot during tests
    with patch("matplotlib.pyplot.show"):
        processor.compute_box_plot()
"""
