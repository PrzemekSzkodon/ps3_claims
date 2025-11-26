import os
import numpy as np
import pandas as pd

try:
    from huggingface_hub import hf_hub_download
    _HAS_HF = True
except Exception:
    hf_hub_download = None
    _HAS_HF = False


def _read_local_or_remote(freq_local, sev_local):
    """Helper to try local, HF dataset URL, hf_hub_download, then OpenML."""
    # 1) local
    if os.path.exists(freq_local) and os.path.exists(sev_local):
        df = pd.read_csv(freq_local)
        df_sev = pd.read_csv(sev_local, index_col=0)
        return df, df_sev, "local"

    # 2) direct HF dataset URL (public)
    hf_freq_url = "https://huggingface.co/datasets/mabilton/fremtpl2/resolve/main/freMTPL2freq.csv"
    hf_sev_url = "https://huggingface.co/datasets/mabilton/fremtpl2/resolve/main/freMTPL2sev.csv"
    try:
        df = pd.read_csv(hf_freq_url)
        df_sev = pd.read_csv(hf_sev_url, index_col=0)
        return df, df_sev, "hf_url"
    except Exception:
        pass

    # 3) huggingface_hub (if installed and allowed)
    if _HAS_HF:
        try:
            freq_path = hf_hub_download(repo_id="mabilton/fremtpl2", filename="freMTPL2freq.csv", repo_type="dataset")
            sev_path = hf_hub_download(repo_id="mabilton/fremtpl2", filename="freMTPL2sev.csv", repo_type="dataset")
            df = pd.read_csv(freq_path)
            df_sev = pd.read_csv(sev_path, index_col=0)
            return df, df_sev, "hf_hub"
        except Exception:
            pass

    # 4) fallback to OpenML URL
    try:
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
    this_dir = os.path.dirname(__file__)
    local_freq = os.path.join(this_dir, "freMTPL2freq.csv")
    local_sev = os.path.join(this_dir, "freMTPL2sev.csv")

    df, df_sev, source = _read_local_or_remote(local_freq, local_sev)

    # rename column names '"name"' => 'name'
    df = df.rename(lambda x: x.replace('"', ''), axis="columns")
    df["IDpol"] = df["IDpol"].astype(np.int64)
    df.set_index("IDpol", inplace=True)

    # join ClaimAmount from df_sev to df:
    df_sev["ClaimAmountCut"] = df_sev["ClaimAmount"].clip(upper=100_000)
    df = df.join(df_sev.groupby(level=0).sum(), how="left")
    df.fillna(value={"ClaimAmount": 0, "ClaimAmountCut": 0}, inplace=True)

    # Note: Zero claims must be ignored in severity models,
    df.loc[(df.ClaimAmount <= 0) & (df.ClaimNb >= 1), "ClaimNb"] = 0

        # correct for unreasonable observations (that might be data error)

    # Clip and/or digitize predictors into bins
    df["VehPower"] = np.minimum(df["VehPower"], 9)
    df["VehAge"] = np.digitize(
        np.where(df["VehAge"] == 10, 9, df["VehAge"]), bins=[1, 10]
    )
    df["DrivAge"] = np.digitize(df["DrivAge"], bins=[21, 26, 31, 41, 51, 71])

    df = df.reset_index()
    return df


if __name__ == "__main__":
    data = load_transform()
    pd.set_option('display.max_columns', None)

    print(data.describe(include='all'))

        # no duplicate content below