import unittest
from unittest.mock import patch, MagicMock
import xarray as xr
import numpy as np
import datetime
from regional_extremes import ClimaticRegionalExtremes


class TestRegionalExtreme(unittest.TestCase):
    def setUp(self):
        """Set up mock datasets for testing."""
        lon = np.arange(0, 360, 50.5)
        lat = np.arange(-90, 90, 50.5)

        time = np.array([
            np.datetime64('1950-01-01'),
            np.datetime64('1951-01-01'),
            np.datetime64('1979-01-01'),
            np.datetime64('2022-12-31')
        ])
        climatic_data = np.random.random((len(lon), len(lat), len(time)))

        lonchunck = np.arange(-179, 179, 50.5)
        latchunck = np.arange(-89, 89, 50.5)
        
        latstep_modis = np.arange(-0.975, 0.975, 0.1)
        lonstep_modis = np.arange(-0.975, 0.975, 0.1)

        ecological_data = np.random.random((len(lonchunck), len(lonstep_modis), len(latchunck), len(latstep_modis), len(time)))

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


class TestClimaticRegionalExtreme(TestRegionalExtreme):

    @patch('xarray.open_zarr')
    def test_load_data(self, mock_open_zarr):
        """Test the load_data method."""
        # Configure the mock to return the mock dataset
        mock_open_zarr.return_value = self.mock_dataset

        processor = ClimaticDataProcessor(self.filepath)
        processor.load_data()

        # Check if the data has been loaded
        self.assertIsNotNone(processor.data)

        # Check if the longitude transformation is correctly applied
        expected_longitudes = ((self.mock_dataset.longitude + 180) % 360) - 180
        np.testing.assert_array_equal(processor.data.longitude, expected_longitudes)

        # Check if the correct time range is selected
        expected_times = np.array([np.datetime64('1951-01-01'), np.datetime64('2022-12-31')])
        np.testing.assert_array_equal(processor.data.time, expected_times)

        # Check if the dimensions are stacked correctly
        self.assertIn('lonlat', processor.data.dims)
        self.assertIn('time', processor.data.dims)
        self.assertEqual(processor.data.dims['lonlat'], len(lon) * len(lat))
        self.assertEqual(processor.data.dims['time'], len(expected_times))

        # Verify print statement output (optional)
        with patch('builtins.print') as mocked_print:
            processor.load_data()
            mocked_print.assert_called_with(f"Climatic datas loaded with dimensions: {processor.data.dims}")

if __name__ == '__main__':
    unittest.main()



def test_climatic_regional_extremes(self):
    processor = ClimaticRegionalExtremes(self.climatic_filepath)
    processor.load_data()
    # Check the data is not empty
    self.assertIsNotNone(processor.data)
    # Check dimension and 
    self.assertEqual(processor.data.dims, {"lon": 2, "lat": 2, "time": 3})
    self.assertLess(processor , 

    processor.apply_transformations()
    self.assertIsNotNone(processor.transformed_data)
    self.assertEqual(processor.transformed_data.dims, {"lonlat": 4, "time": 3})
    

    processor.perform_pca(n_components=2)
    self.assertIsNotNone(processor.pca_result)
    self.assertEqual(processor.pca_result.shape[1], 2)

    # Mock plt.show() to avoid displaying the plot during tests
    with patch("matplotlib.pyplot.show"):
        processor.compute_box_plot()
