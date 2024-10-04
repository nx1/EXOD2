from tqdm import tqdm
import pandas as pd 
from exod.utils.path import data_util

# The GLADE+ catalogue can be downloaded from
# http://elysium.elte.hu/~dalyag/GLADE+.txt
# Requires fastparquet

columns = [
    "GLADE_no",
    "PGC_no",
    "GWGC_name",
    "HyperLEDA_name",
    "2MASS_name",
    "WISExSCOS_name",
    "SDSS_DR16Q_name",
    "Object_type_flag",
    "RA",
    "Dec",
    "B",
    "B_err",
    "B_flag",
    "B_Abs",
    "J",
    "J_err",
    "H",
    "H_err",
    "K",
    "K_err",
    "W1",
    "W1_err",
    "W2",
    "W2_err",
    "W1_flag",
    "B_J",
    "B_J_err",
    "z_helio",
    "z_cmb",
    "z_flag",
    "v_err",
    "z_err",
    "d_L",
    "d_L_err",
    "dist_flag",
    "M*",
    "M*_err",
    "M*_flag",
    "Merger_rate",
    "Merger_rate_error"
]


dtypes_optimized = {
    "GLADE_no": "Int32",               # Integer IDs that are not too large.
    "PGC_no": "Int32",                 # Similar to GLADE_no, assuming non-negative IDs.
    "GWGC_name": "object",             # String identifiers for galaxy names.
    "HyperLEDA_name": "object",        # String identifiers.
    "2MASS_name": "object",            # String identifiers.
    "WISExSCOS_name": "object",        # String identifiers.
    "SDSS_DR16Q_name": "object",       # Mixed strings and nulls.
    "Object_type_flag": "category",    # Likely a small number of categories (e.g., "G").
    "RA": "float64",                   # Right ascension, needs high precision.
    "Dec": "float64",                  # Declination, needs high precision.
    "B": "float32",                    # Magnitude, can use `float32` for less memory usage.
    "B_err": "float32",                # Error on magnitude, doesn't require `float64`.
    "B_flag": "Int8",                  # Binary or small integers, can be stored as `int8`.
    "B_Abs": "float32",                # Absolute magnitude, `float32` is sufficient.
    "J": "float32",                    # Near-IR magnitudes.
    "J_err": "float32",                # Error on J magnitude.
    "H": "float32",                    # Magnitude in H band.
    "H_err": "float32",                # Error in H magnitude.
    "K": "float32",                    # Magnitude in K band.
    "K_err": "float32",                # Error in K magnitude.
    "W1": "float32",                   # WISE W1 band.
    "W1_err": "float32",               # Error in W1 band.
    "W2": "float32",                   # WISE W2 band.
    "W2_err": "float32",               # Error in W2 band.
    "W1_flag": "Int8",                 # Likely a binary or categorical flag.
    "B_J": "float32",                  # Photometry value, `float32` is adequate.
    "B_J_err": "float32",              # Error on B_J.
    "z_helio": "float64",              # Redshift in heliocentric frame, high precision.
    "z_cmb": "float64",                # Redshift in CMB frame, high precision.
    "z_flag": "Int8",                  # Likely a binary or small integer flag.
    "v_err": "float32",                # Error on velocity, `float32` is sufficient.
    "z_err": "float32",                # Redshift error.
    "d_L": "float64",                  # Luminosity distance, requires high precision.
    "d_L_err": "float64",              # Error on luminosity distance.
    "dist_flag": "Int8",               # Binary or small integer flag.
    "M*": "float64",                   # Stellar mass, requires high precision.
    "M*_err": "float64",               # Error on stellar mass.
    "M*_flag": "Int8",                 # Likely a flag for mass measurements.
    "Merger_rate": "float32",          # Merger rate, can use `float32`.
    "Merger_rate_error": "float32"     # Error on merger rate.
}


for i, chunk in tqdm(enumerate(pd.read_csv(data_util / "GLADE+.txt", sep=' ', names=columns, dtype=dtypes_optimized, skiprows=1, chunksize=10000000))):
    if i == 0:
        chunk.to_parquet(data_util / "GLADE+.parquet", engine='fastparquet', index=False, compression='snappy')
    else:

        chunk.to_parquet(data_util / "GLADE+.parquet", engine='fastparquet', index=False, compression='snappy', append=True)

