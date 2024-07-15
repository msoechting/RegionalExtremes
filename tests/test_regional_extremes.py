import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import xarray as xr
import numpy as np
import datetime
import time

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from regional_extremes import CLIMATIC_FILEPATH
CLIMATIC_FILEPATH = "/Net/Groups/BGI/scratch/mweynants/DeepExtremes/v3/PEICube.zarr"
from regional_extremes import RegionalExtremes, ClimaticRegionalExtremes


class TestRegionalExtremes(unittest.TestCase):
    def setUp(self):
        #    """Set up mock datasets for testing."""

        # self.mock_climatic_dataset = lambda index, time: mock_climatic_dataset(
        #    index, time=time
        # )

        # lonchunk = np.arange(-179, 179, 50.5)
        # latchunck = np.arange(-89, 89, 50.5)

        # latstep_modis = np.arange(-0.975, 0.975, 0.5)
        # lonstep_modis = np.arange(-0.975, 0.975, 0.5)

        # ecological_data = np.random.random(
        #     (
        #         len(lonchunk),
        #         len(lonstep_modis),
        #         len(latchunck),
        #         len(latstep_modis),
        #         len(time),
        #     )
        # )
        #
        # self.vegetation_data = xr.DataArray(
        #     ecological_data,
        #     coords=[lonchunk, lonstep_modis, latchunck, latstep_modis, time],
        #     dims=["lonchunk", "lonstep_modis", "latchunck", "latstep_modis", "time"],
        # )

        # Mock the open_dataset method to return the mock datasets
        # self.mock_dataset = patch("xarray.open_dataset").start()
        # self.mock_dataset.side_effect = lambda filepath: {
        #     self.climatic_filepath: xr.Dataset({"data": self.climatic_data}),
        #     self.vegetation_filepath: xr.Dataset({"data": self.vegetation_data}),
        # }[filepath]
        a = 1

    def tearDown(self):
        """Stop all patches."""
        patch.stopall()

    def mock_climatic_dataset(self, index, longitude=None, latitude=None, time=None):
        if longitude is None:
            longitude = np.arange(0, 360, 50.5)
        if latitude is None:
            latitude = np.arange(-90, 90, 50.5)
        if time is None:
            time = np.array(
                [
                    datetime.date(1950, 1, 1),
                    datetime.date(1951, 1, 1),
                    datetime.date(1979, 1, 1),
                    datetime.date(2022, 1, 1),
                ]
            ).astype("datetime64[ns]")

        climatic_data = np.random.uniform(
            -10, 130, (len(longitude), len(latitude), len(time))
        )

        return xr.Dataset(
            {index: (["longitude", "latitude", "time"], climatic_data)},
            coords={"longitude": longitude, "latitude": latitude, "time": time},
        )


class TestClimaticRegionalExtremes(TestRegionalExtremes):
    def test_load_data_climatic(self):
        processor = ClimaticRegionalExtremes()
        processor.load_data()
        # Check the data is not empty
        self.assertIsNotNone(processor.data)

    def test_climatic_apply_transformations_true_data(self):
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

    def test_climatic_apply_transformations_mock_data(self):
        processor = ClimaticRegionalExtremes(index="pei_180")
        # Replace the data by the mock dataset
        processor.data = self.mock_climatic_dataset(index="pei_180")

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

    def test_compute_pca_and_transform_mock_data(self):
        n_samples_training = 10
        n_components = 3
        processor = ClimaticRegionalExtremes(
            index="pei_180",
            step_msc=5,
            n_samples_training=n_samples_training,
            n_components=n_components,
        )
        t1 = time.time()
        # Use the mock dataset
        dates = [
            datetime.date(2017, 1, 1) + datetime.timedelta(days=x)
            for x in range(0, 366, 1)
        ]
        dates = np.array(dates).astype("datetime64[ns]")
        processor.data = self.mock_climatic_dataset(index="pei_180", time=dates)

        # Apply the transformations
        processor.apply_transformations()

        # Scale the data between 0 and 1.
        scaled_data = processor.compute_and_scale_the_msc(n_samples=10)

        self.assertTrue(
            ((scaled_data >= -1) & (scaled_data <= 1)).all(),
            "data are not scaled between 0 and 1.",
        )
        # Fit and apply the PCA
        pca_components = processor.compute_pca_and_transform(scaled_data)
        self.assertTrue(pca_components.shape[0] == n_samples_training)
        self.assertTrue(pca_components.shape[1] == n_components)

    def test_define_limits_bins_mock_data(self):
        n_components = 3
        n_samples = 30
        n_bins = 4

        # Range of the distribution of the mock dataset
        random_low = 0
        random_high = 5

        # Generate mock data
        mock_scaled_data = np.random.randint(
            random_low, random_high, (n_samples, n_components)
        )

        processor = ClimaticRegionalExtremes(
            index="pei_180",
            step_msc=5,
            n_samples_training=100,
            n_components=n_components,
        )
        # Define box
        limits_bins = processor.define_limits_bins(mock_scaled_data, n_bins=n_bins)

        # Assert shapes
        self.assertTrue(len(limits_bins) == n_components)
        self.assertTrue(limits_bins[0].shape[0] == n_bins - 1)

        # Assert values
        for limits_bin in limits_bins:
            self.assertTrue(
                (limits_bin == np.array(range(random_low + 1, random_high - 1))).all()
            )

    def test_find_bins_mock_data(self):
        n_components = 3
        n_samples = 30
        n_bins = 4

        # Range of the distribution of the mock dataset
        lower_value = 0
        upper_value = 5

        # Generate mock data
        mock_data = np.random.randint(
            lower_value, upper_value, (n_samples, n_components)
        )

        processor = ClimaticRegionalExtremes(
            index="pei_180",
            step_msc=5,
            n_samples_training=100,
            n_components=n_components,
        )
        # Define bins
        limits_bins = processor.define_limits_bins(mock_data, n_bins=n_bins)
        # Generate mock data
        new_mock_data = np.array(
            [
                [5, 5, 5],  # higher values than the ones seen during the define_limits
                [-1, -1, -1],  # smaller values
                [1, 1, 1],  # values on the limits
                [2.3, 1.2, 3.7],  # random
                [0.1, 2.4, 3.4],
            ]
        )
        # Attribute the right bins to the new data
        new_bins = processor.find_bins(new_mock_data, limits_bins)

        # Assert shapes
        self.assertTrue(new_bins.shape[0] == new_mock_data.shape[0])
        self.assertTrue(new_bins.shape[1] == n_components)

        # Assert values
        true_values = np.array([[3, 3, 3], [0, 0, 0], [1, 1, 1], [2, 1, 3], [0, 2, 3]])
        self.assertTrue((true_values == new_bins).all())


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
