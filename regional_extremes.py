import xarray as xr
import random
import datetime
from sklearn.decomposition import PCA

from pathlib import Path
from typing import Union
import time

CLIMATIC_FILEPATH = "/Net/Groups/BGI/scratch/mweynants/DeepExtremes/v3/PEICube.zarr"


class RegionalExtremes:
    def __init__(
        self,
        filepath: Union[Path, str],
        index: str,
    ):
        self.filepath = filepath
        self.index = index
        self.data = None

    def load_data(self):
        # Load the PEI-* drought indices
        self.data = xr.open_zarr(self.filepath)[[self.index]]

    def apply_transformations(self):
        raise NotImplementedError("Subclasses should implement this method")

    def perform_pca_on_the_msc(self, n_samples=10, n_components=3, step_msc=5):
        """perform a PCA on the MSC of n samples. Each time step of the msc is considered as an independent component. nb of time_step used for the PCA computation = 366 / step_msc.

        Args:
            n_samples (int, optional): number of samples to fit the PCA. Defaults to 10.
            n_components (int, optional): number of components to compute the PCA. Defaults to 3.
            step_msc (int, optional): temporal resolution of the msc, to reduce computationnal workload. Defaults to 5.
        """
        assert (n_samples > 0) & (n_samples <= self.data.dims["lonlat"])
        assert (n_components > 0) & (
            n_components <= 366
        ), "n_components have to be in the range of days of a years"
        assert (step_msc > 0) & (
            step_msc <= 366
        ), "step_msc have to be in the range of days of a years."

        # Select randomly n_samples to train the PCA:
        # TODO for elecological, select valid values
        lonlat_indices = random.choices(self.data.lonlat.values, k=n_samples)
        selected_data = self.data.sel(lonlat=lonlat_indices)

        # Compute the MSC
        selected_data["msc"] = (
            selected_data[self.index].groupby("time.dayofyear").mean("time")
        )
        # Compute the PCA
        pca = PCA(n_components=n_components)
        pca.fit(selected_data.msc.isel(dayofyear=slice(1, 366, 5)).values)
        print(
            f"PCA performed. sum explained variance: {sum(pca.explained_variance_ratio_)}"
        )


#    def project_pca(self):
#        self.data
#        # Compute the MSC
#        selected_data["msc"] = (
#            selected_data[self.index].groupby("time.dayofyear").mean("time")
#        )
#        return


class ClimaticRegionalExtremes(RegionalExtremes):
    def __init__(self, index):
        assert index in [
            "pei_30",
            "pei_90",
            "pei_180",
        ], "index unavailable. Index available:'pei_30', 'pei_90', 'pei_180'."
        self.filepath = CLIMATIC_FILEPATH
        self.index = index
        self.data = None

    def apply_transformations(self):
        assert self.data is not None
        # Assert dimensions are as expected after loading and transformation
        assert all(
            dim in self.data.dims for dim in ("time", "latitude", "longitude")
        ), "Dimension missing"
        assert (
            (self.data.longitude >= 0) & (self.data.longitude <= 360)
        ).all(), "This function transform longitude values from 0, 360 to the range -180 to 180"

        # Transform the longitude coordinates to -180 and 180
        def coordstolongitude(x):
            return ((x + 180) % 360) - 180

        dsc = self.data.roll(longitude=180 * 4, roll_coords=True)
        ds_pei = dsc.assign_coords(longitude=coordstolongitude(dsc.longitude))

        # Stack the longitude and latitude dimensions into a new dimension called lonlat
        ds_pei = ds_pei.stack(lonlat=("longitude", "latitude")).transpose(
            "lonlat", "time", ...
        )

        # Remove the year 1950 because the data are inconsistent
        ds_pei = ds_pei.sel(
            time=slice(datetime.date(1951, 1, 1), datetime.date(2030, 12, 31))
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

    def apply_transformations(self):
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


if __name__ == "__main__":
    # For climatic data
    t0 = time.time()
    climatic_processor = ClimaticRegionalExtremes(index="pei_180")
    t1 = time.time()
    climatic_processor.load_data()
    t2 = time.time()
    print(t2 - t1)
    climatic_processor.apply_transformations()
    t3 = time.time()
    print(t3 - t2)
    climatic_processor.perform_pca_on_the_msc(n_samples=10, n_components=3)
    t5 = time.time()
    print(t5 - t3)

    # climatic_processor.compute_box_plot()
