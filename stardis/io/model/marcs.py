import pandas as pd
import gzip
import re
from dataclasses import dataclass
from astropy import units as u
import numpy as np
import logging


from stardis.model.geometry.radial1d import Radial1DGeometry
from stardis.model.composition.base import Composition

from stardis.model.base import StellarModel


@dataclass
class MARCSModel(object):
    """
    Class to hold a MARCS model. Holds a dict of the metadata information and a pandas dataframe of the contents.
    Metadata matches for an effective temperature, surface gravity, microturbulence, metallicity (both fe/h and alpha/fe),
    convective parameters, and X (hydrogen mass fraction), Y (helium mass fraction), and Z (heavy element mass fraction).
    """

    metadata: dict
    data: pd.DataFrame

    def to_geometry(self):
        """
        Returns a stardis.model.geometry.radial1d.Radial1DGeometry object from the MARCS model.

        Returns
        -------
        stardis.model.geometry.radial1d.Radial1DGeometry
        """
        r = (
            -self.data.depth.values[::-1] * u.cm
        )  # Flip data to move from innermost stellar point to surface
        return Radial1DGeometry(r)

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
        return Composition(density, atomic_mass_fraction)

    def convert_marcs_raw_abundances_to_mass_fractions(
        self, atom_data, final_atomic_number
    ):
        marcs_chemical_mass_fractions = self.data.filter(
            regex="scaled_log_number"
        ).copy()

        num_of_chemicals_in_model = len(marcs_chemical_mass_fractions.columns)

        if atom_data.atom_data.index.max() < final_atomic_number:
            if (
                len(marcs_chemical_mass_fractions.columns)
                > atom_data.atom_data.index.max()
            ):
                logging.warning(
                    f"Final model chemical number is {num_of_chemicals_in_model} while final atom data chemical number is {atom_data.atom_data.index.max()} and final atomic number requested is {final_atomic_number}."
                )

        for atom_num, col in enumerate(marcs_chemical_mass_fractions.columns):
            if atom_num < len(atom_data.atom_data):
                marcs_chemical_mass_fractions[atom_num + 1] = (
                    10 ** marcs_chemical_mass_fractions[col]
                ) * atom_data.atom_data.mass.iloc[atom_num]
            else:
                for atoms_not_in_atom_data in range(
                    atom_num, marcs_chemical_mass_fractions.shape[1]
                ):
                    marcs_chemical_mass_fractions[atoms_not_in_atom_data + 1] = np.nan
                break

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
        # First two none values are old fv_geometry and abundances which are replaced by the new structures.
        return StellarModel(None, None, temperatures, marcs_geometry, marcs_composition)


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

    Returns
    -------
    dict : dictionary
            metadata parameters of file
    """

    METADATA_RE_STR = [
        ("(.+)\\n", "fname"),
        (
            "  (\d+\.)\s+Teff \[(.+)\]\.\s+Last iteration; yyyymmdd=\d+",
            "teff",
            "teff_units",
        ),
        ("  (\d+\.\d+E\+\d+) Flux \[(.+)\]", "flux", "flux_units"),
        (
            "  (\d+.\d+E\+\d+) Surface gravity \[(.+)\]",
            "surface_grav",
            "surface_grav_units",
        ),
        (
            "  (\d+\.\d+)\W+Microturbulence parameter \[(.+)\]",
            "microturbulence",
            "microturbulence_units",
        ),
        ("  (\d+\.\d+)\s+(No mass for plane-parallel models)", "plane_parallel_mass"),
        (
            " (\+?\-?\d+.\d+) (\+?\-?\d+.\d+) Metallicity \[Fe\/H] and \[alpha\/Fe\]",
            "feh",
            "afe",
        ),
        (
            "  (\d+\.\d+E\+00) (1 cm radius for plane-parallel models)",
            "radius for plane-parallel model",
        ),
        ("  (\d.\d+E-\d+) Luminosity \[(.+)\]", "luminosity", "luminosity_units"),
        (
            "  (\d+.\d+) (\d+.\d+) (\d+.\d+) (\d+.\d+) are the convection parameters: alpha, nu, y and beta",
            "conv_alpha",
            "conv_nu",
            "conv_y",
            "conv_beta",
        ),
        (
            "  (0.\d+) (0.\d+) (\d.\d+E-\d+) are X, Y and Z, 12C\/13C=(\d+.?\d+)",
            "x",
            "y",
            "z",
            "12C/13C",
        ),
    ]
    BYTES_THROUGH_METADATA = 550

    # Compile each of the regex pattern strings then open the file and match each of the patterns by line.
    # Then add each of the matched patterns as a key:value pair to the metadata dict.
    metadata_re = [re.compile(re_str[0]) for re_str in METADATA_RE_STR]
    metadata = {}

    if gzipped:
        with gzip.open(fpath, "rt") as file:
            contents = file.readlines(BYTES_THROUGH_METADATA)

    else:
        with open(fpath, "rt") as file:
            contents = file.readlines(BYTES_THROUGH_METADATA)

    lines = [line for line in contents]

    for i, line in enumerate(lines):
        metadata_re_match = metadata_re[i].match(line)

        for j, metadata_name in enumerate(METADATA_RE_STR[i][1:]):
            metadata[metadata_name] = metadata_re_match.group(j + 1)

    # clean up metadata dictionary by changing strings of numbers to floats and attaching parsed units where appropriate
    keys_to_remove = []
    for i, key in enumerate(metadata.keys()):
        if "_units" in key:
            quantity_to_add_unit = key.split("_units")[0]
            metadata[quantity_to_add_unit] *= u.Unit(metadata[key])
            keys_to_remove.append(key)
        elif key == "fname":
            pass
        else:
            metadata[key] = float(metadata[key])
    metadata = {key: metadata[key] for key in metadata if key not in keys_to_remove}

    return metadata


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
        delim_whitespace=True,
        index_col="k",
    )
    marcs_model_data_lower_split = pd.read_csv(
        fpath,
        skiprows=LINES_BEFORE_LOWER_TABLE,
        nrows=MARCS_MODEL_SHELLS,
        index_col="k",
        sep="(?:\s+)|(?<=\+\d{2})(?=-)",
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

    marcs_model_data.replace({-99.00: np.nan}, inplace=True)

    return marcs_model_data


def read_marcs_model(fpath, gzipped=True):
    """
    Parameters
    ----------
    fpath : str
            Path to model file
    gzipped : Bool
            Whether or not the file is gzipped

    Returns
    -------
    model : MARCSModel
        Assembled metadata and data pair of a MARCS model
    """
    metadata = read_marcs_metadata(fpath, gzipped=gzipped)
    data = read_marcs_data(fpath, gzipped=gzipped)

    return MARCSModel(metadata, data)
