import xarray as xr
import random
import datetime
from sklearn.decomposition import PCA

# import Path
# import Union

climatic_data_filepath = (
    "/Net/Groups/BGI/scratch/mweynants/DeepExtremes/v3/PEICube.zarr"
)


class RegionalExtremes:
    def load_data(self):
        """Load the dataset. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses should implement this method")

    def randomly_select_samples(self, n_samples=10):
        # Select randomly n_samples to train the PCA:
        lat_indices = random.choices(ds_pei.latitude.values, k=1000)
        lon_indices = random.choices(ds_pei.longitude.values, k=1000)

        ds_pei = ds_pei.sel(latitude=lat_indices, longitude=lon_indices)

    def compute_msc(self):
        for index in self.data.data_vars:
            self.data[f"msc_{index}"] = (
                self.data[index].groupby("time.dayofyear").mean("time")
            )

    def perform_pca(self):
        pca = PCA(n_components=5)
        # pca.fit(self.data.msc_pei_30.isel(dayofyear=slice(1, 366, 5)))
        raise NotImplementedError("Subclasses should implement this method")


class ClimaticRegionalExtremes(RegionalExtremes):
    def __init__(
        self,
        # filepath: Union[Path, str],
        # isclimatic: bool,
    ):
        self.filepath = climatic_data_filepath

    def load_data(self):
        # Load the PEI-* drought indices
        ds = xr.open_zarr(self.filepath)

        # Transform the longitude coordinates to -180 and 180
        def coordstolongitude(x):
            return ((x + 180) % 360) - 180

        dsc = ds.roll(longitude=180 * 4, roll_coords=True)
        ds_pei = dsc.assign_coords(longitude=coordstolongitude(dsc.longitude))

        # Remove the year 1950 because the data are inconsistent
        ds_pei = ds_pei.sel(
            time=slice(datetime.date(1951, 1, 1), datetime.date(2022, 12, 31))
        )

        # Stack the longitude and latitude dimensions into a new dimension called lonlat
        ds_pei = ds_pei.stack(lonlat=("longitude", "latitude")).transpose(
            "lonlat", "time", ...
        )

        self.data = ds_pei
        print(f"Climatic datas loaded with dimensions: {self.data.dims}")


class EcologicalRegionalExtremes(RegionalExtremes):
    def __init__(
        self,
        vegetation_index,
        # isclimatic: bool,
    ):
        self.vegetation_index = vegetation_index
        self.filepath = f"/Net/Groups/BGI/work_1/scratch/fluxcom/upscaling_inputs/MODIS_VI_perRegion061/{vegetation_index}/Groups_dyn_{vegetation_index}_MSC_snowfrac.zarr"
        self.filename = f"Groups_dyn_{vegetation_index}_MSC_snowfrac"

    def load_data(self):
        # Load the MSC of MODIS
        ds = xr.open_zarr(self.filepath, consolidated=False)
        ds_msc = ds[self.filename].stack(
            {"lat": ["latchunk", "latstep_modis"], "lon": ["lonchunk", "lonstep_modis"]}
        )

        # Select k locations randomly to train the PCA:
        lat_indices = random.choices(ds_msc.lat.values, k=3000)
        lon_indices = random.choices(ds_msc.lon.values, k=3000)

        # Select the MSC of those locations:
        return


# For climatic data
climatic_processor = ClimaticRegionalExtremes()
climatic_processor.load_data()
climatic_processor.compute_msc()
climatic_processor.perform_pca()
climatic_processor.compute_box_plot()
