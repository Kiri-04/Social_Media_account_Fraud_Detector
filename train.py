# Auto-generated from fake_account_detection.ipynb. Run: python train.py

import matplotlib; matplotlib.use('Agg')


# Fake Social Media Account Detection

import os, json, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)

from features import FEATURES, build_features

os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)


# Step 1: Collect dataset

df = pd.read_csv("data/accounts.csv")
df.head()


# Step 2: Understand the data

print("Shape:", df.shape)
print("\nMissing values:\n", df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nClass distribution (0 = real, 1 = fake):\n", df["is_fake"].value_counts())
df.describe().round(2)


# Step 3: Data preprocessing

df = df.drop_duplicates().reset_index(drop=True)
for col in ["followers", "posts"]:
    df[col] = df[col].fillna(df[col].median())
df = df.drop(columns=["account_id"])
df.to_csv("data/accounts_clean.csv", index=False)

X = build_features(df)   # adds log features and the followers/following ratio
y = df["is_fake"]
print("After cleaning:", df.shape, "| missing:", int(df.isna().sum().sum()))
X.head()


# Step 4: Exploratory data analysis

fig, ax = plt.subplots(figsize=(4.5, 3.5))
counts = y.value_counts().sort_index()
ax.bar(["Real", "Fake"], counts.values, color=["#0f766e", "#c2255c"])
ax.set_title("Class distribution"); ax.set_ylabel("Accounts")
plt.tight_layout(); plt.savefig("reports/eda_class_distribution.png", dpi=150); plt.close('all')

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, col, title in zip(axes.ravel(),
        ["log_followers", "log_following", "log_posts", "log_age_days", "follow_ratio_log"],
        ["Followers (log)", "Following (log)", "Posts (log)", "Account age in days (log)", "Followers/following ratio (log10)"]):
    ax.hist(X.loc[y == 0, col], bins=30, alpha=.6, label="Real", color="#0f766e")
    ax.hist(X.loc[y == 1, col], bins=30, alpha=.6, label="Fake", color="#c2255c")
    ax.set_title(title)
axes[0, 0].legend()
axes.ravel()[-1].axis("off")
plt.tight_layout(); plt.savefig("reports/eda_distributions.png", dpi=150); plt.close('all')

flags = ["has_profile_pic", "has_external_url", "is_verified"]
rates = df.groupby("is_fake")[flags].mean().T
rates.columns = ["Real", "Fake"]
ax = rates.plot(kind="bar", color=["#0f766e", "#c2255c"], figsize=(6, 3.8), rot=0)
ax.set_ylabel("Share of accounts"); ax.set_title("Profile flags by class")
plt.tight_layout(); plt.savefig("reports/eda_flags.png", dpi=150); plt.close('all')

corr = X.assign(is_fake=y).corr()
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=60, ha="right")
ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
fig.colorbar(im); ax.set_title("Correlation heatmap")
plt.tight_layout(); plt.savefig("reports/eda_correlation.png", dpi=150); plt.close('all')


# Step 5: Split the data (80% train, 20% test)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
print(X_train.shape, X_test.shape)


# Step 6: Train the models

models = {
    "Logistic Regression": Pipeline([("scale", StandardScaler()),
                                     ("clf", LogisticRegression(max_iter=1000))]),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}
for name, m in models.items():
    m.fit(X_train, y_train)
    print("trained:", name)


# Step 7: Evaluate (accuracy, precision, recall, F1, confusion matrix)

rows = {}
for name, m in models.items():
    pred = m.predict(X_test)
    rows[name] = {
        "Train accuracy": accuracy_score(y_train, m.predict(X_train)),
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
    }
results = pd.DataFrame(rows).T.round(3)
results.to_csv("reports/model_comparison.csv")
results

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (name, m) in zip(axes, models.items()):
    cm = confusion_matrix(y_test, m.predict(X_test))
    ConfusionMatrixDisplay(cm, display_labels=["Real", "Fake"]).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name)
plt.tight_layout(); plt.savefig("reports/confusion_matrices.png", dpi=150); plt.close('all')


# Step 8: Select the model

best_name = results["F1"].idxmax()
best_model = models[best_name]
print("Selected model:", best_name)
print(results.loc[best_name])


# Step 9: Save the final model

final = best_model.named_steps["clf"] if hasattr(best_model, "named_steps") else best_model
if hasattr(final, "feature_importances_"):
    imp = final.feature_importances_
else:
    imp = np.abs(final.coef_[0]) / np.abs(final.coef_[0]).sum()
importance = dict(zip(FEATURES, [round(float(v), 4) for v in imp]))

joblib.dump({"model": best_model, "name": best_name, "features": FEATURES}, "models/best_model.joblib")
with open("models/metrics.json", "w") as f:
    json.dump({"model": best_name, "metrics": results.loc[best_name].to_dict(),
               "feature_importance": importance}, f, indent=2)
print("Saved models/best_model.joblib")
importance


# Step 10: Try a prediction (the example account from the project brief)

new = pd.DataFrame([{"followers": 120, "following": 1800, "posts": 3, "account_age_days": 20,
                     "has_profile_pic": 1, "has_external_url": 1, "is_verified": 0}])
proba = best_model.predict_proba(build_features(new))[0, 1]
print("Prediction:", "Likely Fake Account" if proba >= 0.5 else "Real Account", f"({proba:.1%} fake probability)")
