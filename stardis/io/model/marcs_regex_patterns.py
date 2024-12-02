METADATA_PLANE_PARALLEL_RE_STR = [
    (r"(.+)\n", "fname"),
    (
        r"  (\d+\.)\s+Teff \[(.+)\]\.\s+Last iteration; yyyymmdd=\d+",
        "teff",
        "teff_units",
    ),
    (r"  (\d+\.\d+E\+\d+) Flux \[(.+)\]", "flux", "flux_units"),
    (
        r"  (\d+.\d+E\+\d+) Surface gravity \[(.+)\]",
        "surface_grav",
        "surface_grav_units",
    ),
    (
        r"  (\d+\.\d+)\W+Microturbulence parameter \[(.+)\]",
        "microturbulence",
        "microturbulence_units",
    ),
    (r"  (\d+\.\d+)\s+(No mass for plane-parallel models)", "plane_parallel_mass"),
    (
        r" (\+?\-?\d+.\d+) (\+?\-?\d+.\d+) Metallicity \[Fe\/H] and \[alpha\/Fe\]",
        "feh",
        "afe",
    ),
    (
        r"  (\d+\.\d+E\+00) (1 cm radius for plane-parallel models)",
        "radius for plane-parallel model",
    ),
    (r"  (\d.\d+E-\d+) Luminosity \[(.+)\]", "luminosity", "luminosity_units"),
    (
        r"  (\d+.\d+) (\d+.\d+) (\d+.\d+) (\d+.\d+) are the convection parameters: alpha, nu, y and beta",
        "conv_alpha",
        "conv_nu",
        "conv_y",
        "conv_beta",
    ),
    (
        r"  (0.\d+) (0.\d+) (\d.\d+E-\d+) are X, Y and Z, 12C\/13C=(\d+.?\d+)",
        "x",
        "y",
        "z",
        "12C/13C",
    ),
]

METADATA_SPHERICAL_RE_STR = [
    (r"(.+)\n", "fname"),
    (
        r"  (\d+\.)\s+Teff \[(.+)\]\.\s+Last iteration; yyyymmdd=\d+",
        "teff",
        "teff_units",
    ),
    (r"  (\d+\.\d+E\+\d+) Flux \[(.+)\]", "flux", "flux_units"),
    (
        r"  (\d+.\d+E\+\d+) Surface gravity \[(.+)\]",
        "surface_grav",
        "surface_grav_units",
    ),
    (
        r"  (\d+\.\d+)\W+Microturbulence parameter \[(.+)\]",
        "microturbulence",
        "microturbulence_units",
    ),
    (
        r"\s+(\d+\.\d+)\s+Mass \[(.+)\]",
        "mass",
        "mass_units",
    ),
    (
        r" (\+?\-?\d+.\d+) (\+?\-?\d+.\d+) Metallicity \[Fe\/H] and \[alpha\/Fe\]",
        "feh",
        "afe",
    ),
    (
        r"  (\d+.\d+E\+\d\d) Radius \[(.+)\] at Tau",
        "radius",
        "radius_units",
    ),
    (
        r"\s+(\d+\.\d+(?:E[+-]?\d+)?) Luminosity \[(.+)\]",
        "luminosity",
        "luminosity_units",
    ),
    (
        r"  (\d+.\d+) (\d+.\d+) (\d+.\d+) (\d+.\d+) are the convection parameters: alpha, nu, y and beta",
        "conv_alpha",
        "conv_nu",
        "conv_y",
        "conv_beta",
    ),
    (
        r"  (0.\d+) (0.\d+) (\d.\d+E-\d+) are X, Y and Z, 12C\/13C=(\d+.?\d+)",
        "x",
        "y",
        "z",
        "12C/13C",
    ),
]
