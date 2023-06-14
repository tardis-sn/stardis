import numpy as np
import pandas as pd
import gzip
from pathlib import Path
import re


class MARCSmod(object):
    """
    Class to hold a MARCS model. Holds a dict of the meta information and a pandas dataframe of the contents.

    """

    def __init__(self, meta, contents):
        pass

    def get_meta(fname):
        """
        Grabs the meta information from a gzipped MARCS model file and returns it in a python dictionary.
        Matches the meta information and units using regex. Fails if the file does not exist or is formatted unexpectedly.

        Parameters
        ----------
        fname : string
                Path to model file

        Returns
        -------
        dict : dictionary
                parameters of file
        """

        with gzip.open(fname, "r") as file:
            contents = file.readlines()
            lines = [line.decode() for line in contents]

            # regex is used to search for the lines.
            teff, teff_units = re.search(
                "(\d+\.)\s+Teff (\[K\])\.\s+Last iteration; yyyymmdd=\d+", lines[1]
            ).group(1, 2)
            flux, flux_units = re.search(
                "(\d+\.\d+E\+\d+) Flux (\[erg.cm2.s\])", lines[2]
            ).group(1, 2)
            surface_grav, surface_grav_units = re.search(
                "(\d+.\d+E\+\d+) Surface gravity (\[cm\/s2\])", lines[3]
            ).group(1, 2)
            microturbulence, microturbulence_units = re.search(
                "(\d+\.\d+)\W+Microturbulence parameter (\[km\/s])", lines[4]
            ).group(1, 2)
            feh, afe, feh_units, afe_units = re.search(
                "(\+?\-?\d+.\d+) (\+?\-?\d+.\d+) Metallicity (\[Fe\/H]) and (\[alpha\/Fe\])",
                lines[6],
            ).group(1, 2, 3, 4)
            Luminosity, Lumonsity_units = re.search(
                "(\d.\d+E-\d+) Luminosity (\[Lsun\])", lines[8]
            ).group(1, 2)
            conv_alpha, conv_nu, conv_y, conv_beta = re.search(
                "(\d+.\d+) (\d+.\d+) (\d+.\d+) (\d+.\d+) are the convection parameters: alpha, nu, y and beta",
                lines[9],
            ).group(1, 2, 3, 4)

            param_dict = {
                "teff": float(teff),
                "teff_units": teff_units,
                "flux": float(flux),
                "flux_units": flux_units,
                "surface_grav": float(surface_grav),
                "surface_grav_units": surface_grav_units,
                "microturbulence": float(microturbulence),
                "microturbulence": microturbulence_units,
                "feh": float(feh),
                "feh_units": feh_units,
                "afe": float(afe),
                "afe_units": afe_units,
            }

        return param_dict

    def get_data(fname):
        """
        Parameters
        ----------
        fname : string
                Path to model file

        Returns
        -------
        dict : pd.DataFrame
                data inside model
        """
        pass
