"""Shared feature engineering: used by the notebook/training script AND the API,
so a new account is transformed exactly like the training data."""
import numpy as np
import pandas as pd

RAW = ["followers", "following", "posts", "account_age_days",
       "has_profile_pic", "has_external_url", "is_verified"]

FEATURES = ["log_followers", "log_following", "log_posts", "log_age_days",
            "follow_ratio_log", "has_profile_pic", "has_external_url", "is_verified"]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["log_followers"] = np.log1p(df["followers"])
    out["log_following"] = np.log1p(df["following"])
    out["log_posts"] = np.log1p(df["posts"])
    out["log_age_days"] = np.log1p(df["account_age_days"])
    # followers/following ratio (log scale so extreme ratios don't dominate)
    out["follow_ratio_log"] = np.log10((df["followers"] + 1) / (df["following"] + 1))
    for c in ["has_profile_pic", "has_external_url", "is_verified"]:
        out[c] = df[c].astype(int)
    return out[FEATURES]
