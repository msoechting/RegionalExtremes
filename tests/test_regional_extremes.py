import sys
import os
import unittest
from unittest.mock import patch, MagicMock, Mock
import tempfile

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from regional_extremes import CLIMATIC_FILEPATH
CLIMATIC_FILEPATH = "/Net/Groups/BGI/scratch/mweynants/DeepExtremes/v3/PEICube.zarr"
from regional_extremes import *  # RegionalExtremes, DatasetHandler


class TestRegionalExtremes(unittest.TestCase):
    def setUp(self):
        #    """Set up mock datasets for testing."""
        index = "pei_180"
        step_msc = 5
        n_components = 3
        n_bins = 4

        self.data_small = DatasetHandler(
            index=index,
            n_samples=10,
            step_msc=step_msc,
            load_data=False,
        )

        self.data_big = DatasetHandler(
            index=index,
            n_samples=100,
            step_msc=step_msc,
            load_data=False,
        )

        self.processor = RegionalExtremes(
            index=index,
            step_msc=step_msc,
            n_components=n_components,
            n_bins=n_bins,
            saving_path="results/test/",
        )

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
        data = DatasetHandler(
            index="pei_180",
            n_samples=10,
            step_msc=5,
            load_data=True,
        )
        data.load_data()
        # Check the data is not empty
        self.assertIsNotNone(processor.data)

    def test_climatic_apply_transformations_true_data(self):
        processor = DatasetHandler(
            index="pei_180",
            n_samples=10,
            step_msc=5,
            load_data=True,
        )
        processor.load_data()
        expected_len_lonlat = (
            processor.data.sizes["longitude"] * processor.data.sizes["latitude"]
        )

        processor.apply_climatic_transformations()

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
        self.assertIn("lonlat", processor.data.sizes)
        self.assertIn("time", processor.data.sizes)
        self.assertEqual(processor.data.sizes["lonlat"], expected_len_lonlat)

    def test_climatic_apply_transformations_mock_data(self):
        processor = self.data_small
        # Replace the data by the mock dataset
        processor.data = self.mock_climatic_dataset(index="pei_180")

        # Apply the transformations
        processor.apply_climatic_transformations()

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
        self.assertIn("lonlat", processor.data.sizes)
        self.assertIn("time", processor.data.sizes)

        # Check if the dimensions lens are correct
        expected_len_lonlat = (
            self.mock_climatic_dataset.sizes["longitude"]
            * self.mock_climatic_dataset.sizes["latitude"]
        )
        self.assertEqual(processor.data.sizes["lonlat"], expected_len_lonlat)
        self.assertEqual(processor.data.sizes["time"], len(expected_times))

    def test_compute_and_scale_the_msc_mock_data(self):
        # Use the mock dataset
        dates = [
            datetime.date(2017, 1, 1) + datetime.timedelta(days=x)
            for x in range(0, 366, 1)
        ]
        dates = np.array(dates).astype("datetime64[ns]")
        self.data_small.data = self.mock_climatic_dataset(index="pei_180", time=dates)

        self.data_small.apply_climatic_transformations()
        self.data_small.randomly_select_data()
        self.assertTrue(
            self.data_small.data.sizes["lonlat"] == 10,
            "number of samples is different than n_samples",
        )
        expected_max = np.max(self.data_small.data)
        expected_min = np.min(self.data_small.data)
        # Scale the data between 0 and 1.
        scaled_data, (max_data, min_data) = self.data_small.compute_and_scale_the_msc()

        self.assertTrue(
            ((scaled_data >= -1) & (scaled_data <= 1)).all(),
            "data are not scaled between 1- and 1.",
        )
        self.assertTrue(
            ((max_data == expected_max) & (min_data == expected_min)),
            "data are not scaled between 0 and 1.",
        )

    #    def test_compute_pca_and_transform(self):
    #        # Fit and apply the PCA
    #        pca_components = processor.compute_pca_and_transform(scaled_data)
    #        self.assertTrue(pca_components.shape[0] == n_samples_training)
    #        self.assertTrue(pca_components.shape[1] == n_components)
    #
    def test_define_limits_bins_mock_data(self):
        n_samples = 100
        n_components = 3
        n_bins = 4

        # Range of the distribution of the mock dataset
        random_low = 0
        random_high = 5

        # Generate mock data
        self.data_big.data = np.random.randint(
            random_low, random_high, (n_samples, n_components)
        )

        # Define box
        limits_bins = processor.define_limits_bins(self.data_big.data)

        # Assert shapes
        self.assertTrue(len(limits_bins) == n_components)
        self.assertTrue(limits_bins[0].shape[0] == n_bins - 1)

        # Assert values
        for limits_bin in limits_bins:
            self.assertTrue(
                (limits_bin == np.array(range(random_low + 1, random_high - 1))).all()
            )

    def test_find_bins_mock_data(self):

        index = "pei_180"
        step_msc = 5
        n_components = 3
        n_bins = 4

        n_samples = 30

        # Range of the distribution of the mock dataset
        lower_value = 0
        upper_value = 5

        # Generate mock data
        mock_data = np.random.randint(
            lower_value, upper_value, (n_samples, n_components)
        )

        processor = RegionalExtremes(
            index=index,
            step_msc=step_msc,
            n_components=n_components,
            n_bins=n_bins,
            saving_path="results/test/",
        )

        # Define bins
        limits_bins = processor.define_limits_bins(mock_data)

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

    @unittest.mock.patch("argparse.ArgumentParser.parse_args")
    def test_save_experiment_mock(self, mock_parse_args):

        # Create a temporary directory for testing
        # with tempfile.TemporaryDirectory() as temp_dir:
        for i in range(1):
            # Mock the argparse.Namespace object returned by parse_args
            # mock_parse_args.return_value = mock_args
            mock_parse_args.return_value = argparse.Namespace(
                index="pei_180",
                n_components=2,
                step_msc=5,
                n_bins=4,
                saving_path="./experiments/",
            )
            # Mock data
            limits_bins = [np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2, 0.3])]
            min_data = 10
            max_data = 100
            processor = RegionalExtremes(
                index=mock_parse_args.return_value.index,
                step_msc=mock_parse_args.return_value.step_msc,
                n_components=mock_parse_args.return_value.n_components,
                n_bins=mock_parse_args.return_value.n_bins,
                saving_path=mock_parse_args.return_value.saving_path,
            )
            # Change the pca attribute by the mock pca
            processor.pca = PCA(n_components=processor.n_components)
            processor.save_experiment(
                mock_parse_args.return_value, limits_bins, min_data, max_data
            )
            # Create the saving path
            saving_path = Path(CURRENT_DIRECTORY_PATH) / Path(
                mock_parse_args.return_value.saving_path
            )
            date_of_today = datetime.datetime.today().strftime("%Y-%m-%-d")

            saving_path = (
                saving_path / f"{mock_parse_args.return_value.index}_{date_of_today}"
            )

            # Verify that files were created and saved correctly
            args_path = saving_path / "args.json"
            print("test", args_path)
            self.assertTrue(args_path.exists())

            min_max_data_path = saving_path / "min_max_data.json"
            self.assertTrue(min_max_data_path.exists())

            pca_path = saving_path / "pca_matrix.pkl"
            self.assertTrue(pca_path.exists())

            limits_bins_path = saving_path / "limits_bins.npy"
            self.assertTrue(limits_bins_path.exists())


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
