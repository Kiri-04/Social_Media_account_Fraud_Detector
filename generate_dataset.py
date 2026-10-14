"""Creates a SYNTHETIC labeled dataset so the whole pipeline runs end to end.
Replace data/accounts.csv with a real labeled dataset (same column names) for your final results."""
import numpy as np, pandas as pd

rng = np.random.default_rng(42)

def make(n, is_fake, kind):
    p = kind
    followers = rng.lognormal(p["fol"][0], p["fol"][1], n)
    following = rng.lognormal(p["fng"][0], p["fng"][1], n)
    posts = rng.lognormal(p["pst"][0], p["pst"][1], n)
    age = rng.lognormal(p["age"][0], p["age"][1], n)
    return pd.DataFrame({
        "followers": followers.astype(int),
        "following": following.astype(int),
        "posts": posts.astype(int),
        "account_age_days": np.maximum(age.astype(int), 1),
        "has_profile_pic": rng.binomial(1, p["pic"], n),
        "has_external_url": rng.binomial(1, p["url"], n),
        "is_verified": rng.binomial(1, p["ver"], n),
        "is_fake": is_fake,
    })

real = dict(fol=(5.5, 1.5), fng=(5.6, 0.9), pst=(3.8, 1.4), age=(6.3, 1.0), pic=0.97, url=0.15, ver=0.04)
fake = dict(fol=(3.8, 1.5), fng=(6.6, 0.8), pst=(1.6, 1.5), age=(3.4, 1.4), pic=0.62, url=0.45, ver=0.002)

parts = [
    make(1650, 0, real),
    make(150, 0, dict(real, fol=(3.5, 1.2), pst=(1.8, 1.2), age=(3.6, 1.0))),   # new but genuine users
    make(1100, 1, fake),
    make(100, 1, dict(fake, fol=(5.2, 1.3), fng=(5.9, 0.9), pst=(3.2, 1.3), age=(5.5, 1.0), pic=0.95)),  # convincing fakes
]
df = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
df.insert(0, "account_id", [f"acc_{i:05d}" for i in range(len(df))])

# add realistic mess: missing values and duplicate rows (to practise cleaning)
for col in ["followers", "posts"]:
    df.loc[rng.choice(len(df), 25, replace=False), col] = np.nan
df = pd.concat([df, df.sample(20, random_state=1).assign(account_id=lambda d: d.account_id)], ignore_index=True)

df.to_csv("data/accounts.csv", index=False)
print(df.shape, df.is_fake.value_counts().to_dict())
