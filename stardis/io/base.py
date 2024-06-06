from pathlib import Path
import logging
import numpy as np

from tardis.io.atom_data import AtomData
from tardis.io.configuration.config_validator import validate_yaml
from tardis.io.configuration.config_reader import Configuration

from stardis.io.model.marcs import read_marcs_model
from stardis.io.model.mesa import read_mesa_model


BASE_DIR = Path(__file__).parent.parent
SCHEMA_PATH = BASE_DIR / "config_schema.yml"


def parse_config_to_model(config_fname):
    """
    Parses the config and model files and outputs python objects to be passed into run stardis so they can be individually modified in python.

    Parameters
    ----------
    config_fname : str
        Filepath to the STARDIS configuration. Must be a YAML file.

    Returns
    -------
    config : stardis.io.configuration.config_reader.Configuration
        Configuration object.
    adata : tardis.io.atom_data.AtomData
        AtomData object.
    stellar_model : stardis.io.model.marcs.MarcsModel or stardis.io.model.mesa.MesaModel
        Stellar model object.
    """

    try:
        config_dict = validate_yaml(config_fname, schemapath=SCHEMA_PATH)
        config = Configuration(config_dict)
    except:
        raise ValueError("Config failed to validate. Check the config file.")

    adata = AtomData.from_hdf(config.atom_data)

    # model
    logging.info("Reading model")
    if config.model.type == "marcs":
        raw_marcs_model = read_marcs_model(
            Path(config.model.fname), gzipped=config.model.gzipped
        )
        stellar_model = raw_marcs_model.to_stellar_model(
            adata, final_atomic_number=config.model.final_atomic_number
        )

    elif config.model.type == "mesa":
        raw_mesa_model = read_mesa_model(Path(config.model.fname))
        if config.model.truncate_to_shell != -99:
            raw_mesa_model.truncate_model(config.model.truncate_to_shell)
        elif config.model.truncate_to_shell < 0:
            raise ValueError(
                f"{config.model.truncate_to_shell} shells were requested for mesa model truncation. -99 is default for no truncation."
            )

        stellar_model = raw_mesa_model.to_stellar_model(
            adata, final_atomic_number=config.model.final_atomic_number
        )

    else:
        raise ValueError("Model type not recognized. Must be either 'marcs' or 'mesa'")

    # Handle case of when there are fewer elements requested vs. elements in the atomic mass fraction table.
    adata.prepare_atom_data(
        np.arange(
            1,
            np.min(
                [
                    len(
                        stellar_model.composition.atomic_mass_fraction.columns.tolist()
                    ),
                    config.model.final_atomic_number,
                ]
            )
            + 1,
        ),
        line_interaction_type="macroatom",
        nlte_species=[],
        continuum_interaction_species=[],
    )

    if (
        not config.model.elemental_rescaling_dict
    ):  # Pass if no rescaling is requested, else rescale by dictionary values provided
        pass
    else:
        stellar_model.composition.rescale_elements(
            list(config.model.elemental_rescaling_dict.keys()),
            list(config.model.elemental_rescaling_dict.values()),
        )

    return config, adata, stellar_model
