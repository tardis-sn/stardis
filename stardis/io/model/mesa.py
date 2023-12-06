import pandas as pd
import re
from dataclasses import dataclass
from astropy import units as u
import numpy as np
import logging

from stardis.model.geometry.radial1d import Radial1DGeometry
from stardis.model.composition.base import Composition

from stardis.model.base import StellarModel


@dataclass
class MESAModel(object):
    """
    Class to hold a MESA model. Holds a dict of the metadata information and a pandas dataframe of the contents.
    """

    metadata: dict
    data: pd.DataFrame

    def truncate_model(self, shell_number: int = None):
        """
        Truncates the model to some specified number of shells (starting from outermost).

        Args:
            shell_number: The number of shells to keep.
        """
        self.data = self.data[self.data.index <= shell_number]

    def to_geometry(self):
        """
        Returns a stardis.model.geometry.radial1d.Radial1DGeometry object from the MESA model.

        Returns
        -------
        stardis.model.geometry.radial1d.Radial1DGeometry
        """
        r = (
            np.exp(self.data.lnR[::-1]) * u.cm
        )  # Flip the order of the shells to move from innermost point to surface
        return Radial1DGeometry(r)

    def to_composition(self, atom_data, final_atomic_number):

def read_mesa_metadata(fpath):
    """
    Reads the metadata from a MESA file.

    Args:
        fpath: The path to the MESA file.

    Returns:
        A dictionary containing the metadata extracted from the MESA file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    # The regular expression pattern
    METADATA_RE_STR = [
        ("\s*version_number\s+(\S+)\s*\n", "Version number"),
        ("\s*M/Msun\s+(\S+)\s*\n", "Mass"),
        ("\s*model\_number\s+(\S+)\s*\n", "Model Number"),
        ("\s*star_age\s+(\S+)\s*\n", "Star Age"),
        ("\s*initial_z\s+(\S+)\s*\n", "Initial Z"),
        ("\s*n_shells\s+(\S+)\s*\n", "Number of Shells"),
        ("\s*net_name\s+(\S+)\s*\n", "Net Name"),
        ("\s*species\s+(\S+)\s*\n", "Number of Species"),
        ("\s*Teff\s+(\S+)\s*\n", "Effective Temperature"),
    ]

    metadata_re = [re.compile(re_str[0]) for re_str in METADATA_RE_STR]
    metadata = {}

    with open(fpath, "rt") as file:
        contents = file.readlines()

    lines = list(contents)

    for line_number, line in enumerate(lines):
        match = metadata_re[0].match(line)
        if match is not None:
            STARTING_OF_METADATA = line_number
            for i, line in enumerate(
                lines[
                    STARTING_OF_METADATA : STARTING_OF_METADATA + len(METADATA_RE_STR)
                ]
            ):
                metadata_re_match = metadata_re[i].match(line)
                for j, metadata_name in enumerate(METADATA_RE_STR[i][1:]):
                    metadata[metadata_name] = metadata_re_match.group(j + 1)
            break

    for key in metadata:
        metadata[key] = metadata[key].replace("D", "e")

    metadata["Mass"] = float(metadata["Mass"]) * u.M_sun
    metadata["Number of Shells"] = int(metadata["Number of Shells"])
    metadata["Star Age"] = float(metadata["Star Age"]) * u.yr
    metadata["Effective Temperature"] = float(metadata["Effective Temperature"]) * u.K
    metadata["Initial Z"] = float(metadata["Initial Z"])
    metadata["Number of Species"] = int(metadata["Number of Species"])
    metadata["Model Number"] = int(metadata["Model Number"])

    return metadata


def read_mesa_data(fpath, mesa_shells):
    """
    Parameters
    ----------
    fpath : str
            Path to model file


    Returns
    -------
    data : pd.DataFrame
        data contents of the mesa model file
    """

    ROWS_TO_SKIP = 23

    mesa_model = pd.read_csv(
        fpath,
        skiprows=ROWS_TO_SKIP,
        delim_whitespace=True,
        nrows=mesa_shells,
        index_col=0,
        comment="!",
    )

    mesa_model = mesa_model.apply(
        lambda number: number.str.replace("D", "E").astype(float)
    )

    return mesa_model


def read_mesa_model(fpath):
    """
    Parameters
    ----------
    fpath : str
            Path to model file

    Returns
    -------
    model : MESAModel
        Assembled metadata and data pair of a MARCS model
    """
    metadata = read_mesa_metadata(fpath)
    data = read_mesa_data(fpath, mesa_shells=metadata["Number of Shells"])

    return MESAModel(metadata, data)
