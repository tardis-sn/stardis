import pandas as pd
import gzip
import re
from dataclasses import dataclass
from astropy import units as u
import numpy as np
import logging


from stardis.model.geometry.radial1d import Radial1DGeometry
from stardis.model.base import StellarModel
from tardis.model.matter.composition import Composition
from stardis.io.model.marcs_regex_patterns import (
    METADATA_PLANE_PARALLEL_RE_STR,
    METADATA_SPHERICAL_RE_STR,
)

logger = logging.getLogger(__name__)


@dataclass
class MARCSModel(object):
    """
    Class to hold a MARCS model. Holds a dict of the metadata information and a pandas dataframe of the contents.
    Metadata matches for an effective temperature, surface gravity, microturbulence, metallicity (both fe/h and alpha/fe),
    convective parameters, and X (hydrogen mass fraction), Y (helium mass fraction), and Z (heavy element mass fraction).
    """

    metadata: dict
    data: pd.DataFrame
    spherical: bool

    def to_geometry(self):
        """
        Returns a stardis.model.geometry.radial1d.Radial1DGeometry object from the MARCS model.

        Returns
        -------
        stardis.model.geometry.radial1d.Radial1DGeometry
        """

        reference_r = None
        r = (
            -self.data.depth.values[::-1] * u.cm
        )  # Flip data to move from innermost stellar point to surface
        if self.spherical:
            r += self.metadata["radius"]
            reference_r = self.metadata["radius"]
        return Radial1DGeometry(r, reference_r)

    def to_composition(self, atom_data, final_atomic_number):
        """
        Returns a stardis.model.composition.base.Composition object from the MARCS model.

        Parameters
        ----------
        atom_data : tardis.io.atom_data.base.AtomData
        final_atomic_number : int, optional
            Atomic number for the final element included in the model.

        Returns
        ----------
        stardis.model.composition.base.Composition
        """
        density = (
            self.data.density.values[::-1] * u.g / u.cm**3
        )  # Flip data to move from innermost stellar point to surface
        atomic_mass_fraction = self.convert_marcs_raw_abundances_to_mass_fractions(
            atom_data, final_atomic_number
        )

        atomic_mass_fraction["mass_number"] = -1
        atomic_mass_fraction.set_index("mass_number", append=True, inplace=True)

        return Composition(
            density,
            atomic_mass_fraction,
            raw_isotope_abundance=None,
            element_masses=atom_data.atom_data.mass.copy(),
        )

    def convert_marcs_raw_abundances_to_mass_fractions(
        self, atom_data, final_atomic_number
    ):
        marcs_chemical_mass_fractions = self.data.filter(
            regex="scaled_log_number"
        ).copy()

        num_of_chemicals_in_model = len(marcs_chemical_mass_fractions.columns)

        if atom_data.atom_data.index.max() < final_atomic_number and (
            len(marcs_chemical_mass_fractions.columns) > atom_data.atom_data.index.max()
        ):
            logging.warning(
                f"Final model chemical number is {num_of_chemicals_in_model} while final atom data chemical number is {atom_data.atom_data.index.max()} and final atomic number requested is {final_atomic_number}."
            )

        for atom_num, col in enumerate(marcs_chemical_mass_fractions.columns):
            if atom_num < len(atom_data.atom_data):
                marcs_chemical_mass_fractions[atom_num + 1] = (
                    10 ** marcs_chemical_mass_fractions[col]
                ) * atom_data.atom_data.mass.iloc[atom_num]

        # Remove scaled log number columns - leaves only masses
        dropped_cols = [
            c
            for c in marcs_chemical_mass_fractions.columns
            if "scaled_log_number" in str(c)
        ]
        marcs_chemical_mass_fractions.drop(columns=dropped_cols, inplace=True)

        # Divide by sum to leave mass densities
        marcs_chemical_mass_fractions = marcs_chemical_mass_fractions.div(
            marcs_chemical_mass_fractions.sum(axis=1), axis=0
        )
        # Truncate to final atomic number, if smaller than the number of chemicals in the model
        marcs_chemical_mass_fractions = marcs_chemical_mass_fractions.iloc[
            :,
            : np.min([final_atomic_number, num_of_chemicals_in_model]),
        ]

        # Convert to atom data format expected by tardis plasma
        marcs_atom_data = pd.DataFrame(
            columns=marcs_chemical_mass_fractions.index
            - 1,  # columns need to start from 0 to avoid tardis plasma crashing
            index=marcs_chemical_mass_fractions.columns,
            data=np.fliplr(
                marcs_chemical_mass_fractions.values.T
            ),  # Flip column order to move from deepest point to surface - Doesn't change data if abundances are uniform
        )

        marcs_atom_data.index.name = "atomic_number"

        return marcs_atom_data

    def to_stellar_model(self, atom_data, final_atomic_number=118):
        """
        Produces a stellar model readable by stardis.

        Parameters
        ----------
        atom_data : tardis.io.atom_data.base.AtomData
        final_atomic_number : int, optional
            Atomic number for the final element included in the model. Default
            is 118, an abitrarily large atomic number so as not to truncate by default.

        Returns
        -------
        stardis.model.base.StellarModel
        """
        marcs_geometry = self.to_geometry()
        marcs_composition = self.to_composition(
            atom_data=atom_data, final_atomic_number=final_atomic_number
        )
        temperatures = (
            self.data.t.values[::-1] * u.K
        )  # Flip data to move from innermost stellar point to surface
        return StellarModel(
            temperatures,
            marcs_geometry,
            marcs_composition,
            spherical=self.spherical,
            microturbulence=self.metadata["microturbulence"],
        )


def read_marcs_metadata(fpath, gzipped=True):
    """
    Grabs the metadata information from a gzipped MARCS model file and returns it in a python dictionary.
    Matches the metadata information and units using regex. Assumes file line structure of plane-parallel models.
    Fails if the file does not exist or is formatted unexpectedly.

    Parameters
    ----------
    fpath : str
            Path to model file
    gzipped : Bool
            Whether or not the file is gzipped
    spherical : Bool
            Whether or not the model is spherical

    Returns
    -------
    dict : dictionary
            metadata parameters of file
    """
    BYTES_THROUGH_METADATA = 550

    if gzipped:
        with gzip.open(fpath, "rt") as file:
            contents = file.readlines(BYTES_THROUGH_METADATA)

    else:
        with open(fpath, "rt") as file:
            contents = file.readlines(BYTES_THROUGH_METADATA)

    lines = list(contents)

    # Compile each of the regex pattern strings then open the file and match each of the patterns by line.
    # Then add each of the matched patterns as a key:value pair to the metadata dict.
    # Files are formatted a little differently depending on if the MARCS model is spherical or plane-parallel
    if "plane-parallel" in lines[5]:
        logger.info("Plane-parallel model detected.")
        spherical = False
        metadata_re = [
            re.compile(re_str[0]) for re_str in METADATA_PLANE_PARALLEL_RE_STR
        ]
        metadata_re_str = METADATA_PLANE_PARALLEL_RE_STR
    else:
        logger.info("Spherical model detected.")
        spherical = True
        metadata_re = [re.compile(re_str[0]) for re_str in METADATA_SPHERICAL_RE_STR]
        metadata_re_str = METADATA_SPHERICAL_RE_STR

    metadata = {}

    # Check each line against the regex patterns and add the matched values to the metadata dictionary
    for i in range(len(metadata_re_str)):
        line = lines[i]
        metadata_re_match = metadata_re[i].match(line)
        for j, metadata_name in enumerate(metadata_re_str[i][1:]):
            metadata[metadata_name] = metadata_re_match.group(j + 1)

    # clean up metadata dictionary by changing strings of numbers to floats and attaching parsed units where appropriate
    keys_to_remove = []
    for key in metadata:
        if "_units" in key:
            quantity_to_add_unit = key.split("_units")[0]
            metadata[quantity_to_add_unit] *= u.Unit(metadata[key])
            keys_to_remove.append(key)
        elif key != "fname":
            metadata[key] = float(metadata[key])
    metadata = {key: metadata[key] for key in metadata if key not in keys_to_remove}

    return metadata, spherical


def read_marcs_data(fpath, gzipped=True):
    """
    Parameters
    ----------
    fpath : str
            Path to model file
    gzipped : Bool
            Whether or not the file is gzipped

    Returns
    -------
    data : pd.DataFrame
        data contents of the MARCS model file
    """

    # Interior model file contents are split into two tables vertically. Each needs to be read
    # in separately and then joined on shared planes (k for plane number or lgTauR for optical depth.)
    MARCS_MODEL_SHELLS = 56
    LINES_BEFORE_UPPER_TABLE = 24
    LINES_BEFORE_LOWER_TABLE = 81

    BYTES_THROUGH_ABUNDANCES = 1290
    LINES_BEFORE_ABUNDANCES = 12
    LINES_THROUGH_ABUNDANCES = 22

    marcs_model_data_upper_split = pd.read_csv(
        fpath,
        skiprows=LINES_BEFORE_UPPER_TABLE,
        nrows=MARCS_MODEL_SHELLS,
        sep=r"\s+",
        index_col="k",
    )
    marcs_model_data_lower_split = pd.read_csv(
        fpath,
        skiprows=LINES_BEFORE_LOWER_TABLE,
        nrows=MARCS_MODEL_SHELLS,
        index_col="k",
        sep=r"(?:\s+)|(?<=\+\d{2})(?=-)",
        engine="python",
    )

    marcs_model_data = pd.merge(
        marcs_model_data_upper_split, marcs_model_data_lower_split, on=["k", "lgTauR"]
    )
    marcs_model_data.columns = [item.lower() for item in marcs_model_data.columns]

    if gzipped:
        with gzip.open(fpath, "rt") as file:
            contents = file.readlines(BYTES_THROUGH_ABUNDANCES)
    else:
        with open(fpath, "rt") as file:
            contents = file.readlines(BYTES_THROUGH_ABUNDANCES)

    marcs_abundance_scale_str = " ".join(
        [
            item.strip()
            for item in contents[LINES_BEFORE_ABUNDANCES:LINES_THROUGH_ABUNDANCES]
        ]
    )

    for i, abundance in enumerate(marcs_abundance_scale_str.split()):
        marcs_model_data[f"scaled_log_number_fraction_{i+1}"] = float(abundance)

    # NOTE: Replace empty values with 0.0 to avoid issues with tardis plasma.
    marcs_model_data.replace({-99.00: 0.0}, inplace=True)

    return marcs_model_data


def read_marcs_model(fpath, gzipped=True):
    """
    Parameters
    ----------
    fpath : str
            Path to model file
    gzipped : Bool
            Whether or not the file is gzipped
    spherical : Bool
            Whether or not the model is spherical

    Returns
    -------
    model : MARCSModel
        Assembled metadata and data pair of a MARCS model
    """
    try:
        metadata, spherical = read_marcs_metadata(fpath, gzipped=gzipped)
    except:
        raise ValueError(
            "Failed to read metadata from MARCS model file. Make sure that you are specifying if the file is gzipped appropriately."
        )
    data = read_marcs_data(fpath, gzipped=gzipped)

    return MARCSModel(metadata, data, spherical=spherical)
