"""Simple loader/transform for the freMTPL2 insurance dataset.

Design notes:
- The loader tries multiple sources in order: local files > HF resolve URL
    > huggingface_hub helper > OpenML. The goal is a robust script that works
    offline (local CSVs) or in automated environments with HF caching.
- The severity table uses policy ID as its index; so we read it with the
    first column as index (index_col=0) and then aggregate per policy before
    joining back to the frequency table.
"""

import os  # file/directory operations
import numpy as np  # numerical ops (arrays, nan handling)
import pandas as pd  # DataFrame operations

try:
    # Optional: huggingface_hub provides hf_hub_download for dataset caching
    from huggingface_hub import hf_hub_download
    # Import succeeded: hf_hub_download will be available. Note: importing
    # the helper doesn't imply you are authenticated to private HF repos; it
    # only means the Python package is present. To access gated HF repos you
    # may need to set 'HUGGINGFACE_HUB_TOKEN' (or HF_TOKEN) in your env.
    _HAS_HF = True  # True means the hf_hub_download helper is importable in this env
except Exception:
    # huggingface_hub not installed; we'll fall back to other sources
    hf_hub_download = None
    _HAS_HF = False  # No huggingface_hub: we will try other download strategies


def _read_local_or_remote(freq_local, sev_local):
    """
    Helper to load the 'frequency' and 'severity' CSVs.

    Returns a triple (df_freq, df_sev, source_str), where 'source_str' is a
    short label showing which fallback path succeeded: 'local', 'hf_url',
    'hf_hub', or 'openml'. Returning both dataframes is convenient for the
    caller because the severity table uses a different index scheme.
    """
    # 1) local files are preferred (offline, fastest)
    # 1) Prefer local CSVs placed next to this script (offline, fastest).
    # If both CSVs exist, we load them and return immediately.
    if os.path.exists(freq_local) and os.path.exists(sev_local):
        # We intentionally read the severity table with index_col=0 because
        # the CSV's first column is the policy ID, and the severity dataset
        # is organized as one row per claim, referencing the policy ID.
        df = pd.read_csv(freq_local)
        df_sev = pd.read_csv(sev_local, index_col=0)
        return df, df_sev, "local"

    # 2) Try the Hugging Face dataset direct 'resolve' URL. These URLs will
    # redirect to an S3 URL if the dataset is public and allow a simple
    # HTTP-based read from pandas.read_csv without requiring huggingface_hub.
    hf_freq_url = "https://huggingface.co/datasets/mabilton/fremtpl2/resolve/main/freMTPL2freq.csv"
    hf_sev_url = "https://huggingface.co/datasets/mabilton/fremtpl2/resolve/main/freMTPL2sev.csv"
    try:
        # Download public CSV via HF dataset direct URL (fast path)
        # Here we rely on simple HTTP GETs; some HF datasets redirect and
        # may require authentication. This path sometimes works for public
        # datasets and doesn't require huggingface_hub.
        df = pd.read_csv(hf_freq_url)
        df_sev = pd.read_csv(hf_sev_url, index_col=0)
        return df, df_sev, "hf_url"
    except Exception:
        pass

    # 3) Use huggingface_hub if installed. This gives caching and retries and
    # can be used even if the file isn't public as long as the environment
    # has proper HF authentication set up. This path uses the local HF cache.
    if _HAS_HF:
        try:
            # Pull files via HF hub API; this uses the cache if possible
            # hf_hub_download caches artifacts locally and will download
            # them if necessary; it needs proper HF authentication if the
            # dataset is gated/private.
            freq_path = hf_hub_download(repo_id="mabilton/fremtpl2", filename="freMTPL2freq.csv", repo_type="dataset")
            sev_path = hf_hub_download(repo_id="mabilton/fremtpl2", filename="freMTPL2sev.csv", repo_type="dataset")
            df = pd.read_csv(freq_path)
            df_sev = pd.read_csv(sev_path, index_col=0)
            return df, df_sev, "hf_hub"
        except Exception:
            pass

    # 4) Last fallback: OpenML. This uses the public OpenML-hosted CSV (if
    # available). Note that different mirrors may create variant encodings.
    try:
        # Read OpenML dataset as final fallback (uses HTTP arff csv link).
        # OpenML datasets are public and often mirror the same CSVs used
        # across other hosting sites.
        # We pass quotechar="'" because the arff->csv converter on OpenML
        # uses single quotes for categorical values; telling pandas to use
        # that quote character avoids issues with comma-separated values.
        df = pd.read_csv("https://www.openml.org/data/get_csv/20649148/freMTPL2freq.arff", quotechar="'")
        df_sev = pd.read_csv("https://www.openml.org/data/get_csv/20649149/freMTPL2sev.arff", index_col=0)
        return df, df_sev, "openml"
    except Exception as e:
        raise RuntimeError("Could not load dataset. Please download the CSVs and place them next to this script or ensure HF/OpenML access.") from e


def load_transform():
    """Load and transform data from OpenML or Hugging Face dataset.

    - Tries local files next to this script
    - Tries HF dataset HTTP resolve URL (no auth required for public datasets)
    - Tries huggingface_hub to fetch files (requires package and access)
    - Falls back to OpenML URL (if available)
    """

    # Local fallback paths (script directory)
    this_dir = os.path.dirname(__file__) # __file__ is the path to this script
    local_freq = os.path.join(this_dir, "freMTPL2freq.csv") #this gets the path to the freq csv
    local_sev = os.path.join(this_dir, "freMTPL2sev.csv") #this gets the path to the sev csv

    # Load the data using the helper that tries several fallbacks.
    # df: frequency dataset (one row per policy)
    # df_sev: severity dataset (one row per claim; index refers to policy)
    # source: a short string explaining which path was used (for debugging)
    df, df_sev, source = _read_local_or_remote(local_freq, local_sev) #three variables called in one go 

    # Keep a module-level record of where the data came from for debugging
    global _LAST_LOAD_SOURCE #
    _LAST_LOAD_SOURCE = source

    # Clean column names: remove surrounding quotes if present. The raw files
    # sometimes include quotes around the column names (e.g., "IDpol").
    # This replacement only affects column headers, not values.
    # Example: '"IDpol"' -> 'IDpol'
    # We also cast IDpol to an integer type and set it as the index. We keep
    # IDpol as the index since the severity data refers to policies by that id.
    # Setting a numeric index improves join performance and makes groupby
    # operations more intuitive later.
    # (The frequency dataset is one row per policy)
    df = df.rename(lambda x: x.replace('"', ''), axis="columns")
    df["IDpol"] = df["IDpol"].astype(np.int64)
    df.set_index("IDpol", inplace=True)

    # Join the severity table to the frequency table. df_sev contains claim
    # rows for each policy; we group by the policy ID (index) and sum to get
    # aggregate ClaimAmount per policy, then join these aggregated columns
    # into the main frequency table. We also clip very large ClaimAmount
    # values to guard against extreme outliers or errors.
    df_sev["ClaimAmountCut"] = df_sev["ClaimAmount"].clip(upper=100_000)
    df = df.join(df_sev.groupby(level=0).sum(), how="left")
    # After the left-join, policies with no claims will have NaN in the
    # severity columns. Replace these with zeros because no claim -> 0.
    df.fillna(value={"ClaimAmount": 0, "ClaimAmountCut": 0}, inplace=True)

    # Correct inconsistent rows: if the aggregated ClaimAmount is zero but
    # ClaimNb (number of claims) is positive, then set ClaimNb to zero. This
    # typically fixes cases where the severity dataset had zero-priced claims
    # or where records were corrupted. Severity models normally ignore
    # zero-amount claims, so this harmonises the two columns.
    df.loc[(df.ClaimAmount <= 0) & (df.ClaimNb >= 1), "ClaimNb"] = 0

        # correct for unreasonable observations (that might be data error)

    # Feature engineering: we clip or bucketize features (VehPower, VehAge,
    # DrivAge) to reduce the number of categories (for easier modelling)
    # and to guard against improbable values. Example: VehPower > 9 is
    # replaced with 9 (a typical categorical limit).
    # VehAge: bucket the age into 1-9 categories; replace 10 with 9 first so
    # it falls into the final bin (this was observed from dataset quirks).
    df["VehPower"] = np.minimum(df["VehPower"], 9)
    df["VehAge"] = np.digitize(
        np.where(df["VehAge"] == 10, 9, df["VehAge"]), bins=[1, 10]
    )
    df["DrivAge"] = np.digitize(df["DrivAge"], bins=[21, 26, 31, 41, 51, 71])

    # Reset index: return the DataFrame with a fresh RangeIndex and with
    # IDpol as a normal column. This is often more convenient for downstream
    # use (e.g., merging with other data or saving to CSV).
    # Return with a conventional RangeIndex instead of IDpol as the index.
    # Downstream code often prefers IDpol as a normal column, e.g., when
    # merging with other policy-level datasets or saving to CSV.
    df = df.reset_index()
    return df


if __name__ == "__main__":
    data = load_transform()
    pd.set_option('display.max_columns', None)

    # A quick summary for a developer: show all columns and statistics.
    # This is useful during development to sanity check the dataset shape
    # and basic distribution attributes.
    print(data.describe(include='all'))
    try:
        print(f"Loaded from: {_LAST_LOAD_SOURCE}")
    except NameError:
        pass

        # no duplicate content below