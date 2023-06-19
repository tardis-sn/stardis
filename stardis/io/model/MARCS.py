import pandas as pd
import gzip
import re
from dataclasses import dataclass
from astropy import units as u


@dataclass
class MARCSModel(object):
    """
    Class to hold a MARCS model. Holds a dict of the metadata information and a pandas dataframe of the contents.

    """

    metadata: dict
    data: pd.DataFrame


def read_marcs_metadata(fpath):
    """
    Grabs the metadata information from a gzipped MARCS model file and returns it in a python dictionary.
    Matches the metadata information and units using regex. Assumes line structure of plane-parallel models.
    Fails if the file does not exist or is formatted unexpectedly.

    Parameters
    ----------
    fpath : str
            Path to model file

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

    # Compile each of the regex pattern strings then open the file and match each of the patterns by line.
    # Then add each of the matched patterns as a key:value pair to the metadata dict.
    metadata_re = [re.compile(re_str[0]) for re_str in METADATA_RE_STR]
    metadata = {}
    with gzip.open(fpath, "rt") as file:
        contents = file.readlines(550)
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


def read_marcs_data(fpath):
    """
    Parameters
    ----------
    fpath : str
            Path to model file

    Returns
    -------
    data : pd.DataFrame
        data contents of the MARCS model file
    """

    # Interior model file contents are split in to two tables vertically. Each needs to be read
    # in separately and then joined on shared planes (k for plane number or lgTauR for optical depth.)
    marcs_model_data_upper_split = pd.read_csv(
        fpath, skiprows=24, nrows=56, delim_whitespace=True, index_col="k"
    )
    marcs_model_data_lower_split = pd.read_csv(
        fpath,
        skiprows=81,
        nrows=56,
        index_col="k",
        sep="(?:\s+)|(?<=\+\d{2})(?=-)",
    )

    marcs_model_data = pd.merge(
        marcs_model_data_upper_split, marcs_model_data_lower_split, on=["k", "lgTauR"]
    )
    marcs_model_data.columns = [item.lower() for item in marcs_model_data.columns]

    return marcs_model_data


def read_marcs_model(fpath):
    """
    Parameters
    ----------
    fpath : str
            Path to model file

    Returns
    -------
    model : MARCSModel
        Assembled metadata and data pair of a MARCS model
    """
    metadata = read_marcs_metadata(fpath)
    data = read_marcs_data(fpath)

    return MARCSModel(metadata, data)
