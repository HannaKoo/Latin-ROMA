```python
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA


# Configuration
experiment = "Htarg"
tokens_per_chunk = 500

workdir = Path("Results") / f"{experiment}{tokens_per_chunk}"
input_file = workdir / "chunks_norm.tsv"


# Plot style
plt.style.use("seaborn-v0_8-whitegrid")


# Load data
data = pd.read_csv(
    input_file,
    sep="\t",
)

X = data.drop(columns=["book", "Register", "Tokens"])
labels = data["Register"]


# Center the feature matrix
X = X - X.mean()


# Full PCA
pca_full = PCA()
pca_full.fit(X)


# Export PCA loadings
with open(
    workdir / "PCA-components.tsv",
    "w",
    newline="",
    encoding="utf-8",
) as tsv:
    writer = csv.writer(tsv, delimiter="\t")
    writer.writerow(X.columns)
    writer.writerows(pca_full.components_)


# Inspect the strongest loadings for PC1 and PC2
for pc in [0, 1]:
    loadings = pd.Series(
        pca_full.components_[pc],
        index=X.columns,
    )

    loadings = loadings.reindex(
        loadings.abs().sort_values(ascending=False).index
    )

    print(f"\nPC{pc + 1}")
    print(loadings.head(20))


# Cumulative explained variance
plt.figure()

plt.plot(
    np.cumsum(pca_full.explained_variance_ratio_)[:30]
)

plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")

plt.tight_layout()

plt.savefig(
    workdir / f"pca-{experiment}{tokens_per_chunk}_0-30.svg",
    format="svg",
)

plt.show()


# Scree plot
n_components = 16
pc_values = np.arange(1, n_components + 1)

plt.figure()

plt.plot(
    pc_values,
    pca_full.explained_variance_ratio_[:n_components],
    "o-",
    linewidth=2,
    color="black",
)

plt.title("Scree Plot")
plt.xlabel("Principal Component")
plt.ylabel("Variance Explained")

plt.tight_layout()

plt.savefig(
    workdir / f"pca-{experiment}{tokens_per_chunk}_scree.svg",
    format="svg",
)

plt.show()


# Two-component PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)


# Register order and markers
hue_order = [
    "kirje_antiikki",
    "tutkielma_antiikki",
    "tutkielma_humanismi",
    "kirje_humanismi",
]

markers = {
    "kirje_antiikki": "D",
    "tutkielma_antiikki": "X",
    "tutkielma_humanismi": "s",
    "kirje_humanismi": "P",
}


# PCA plot
plot = sns.relplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=labels,
    hue_order=hue_order,
    style=labels,
    markers=markers,
    alpha=0.4,
    palette="colorblind",
)

plot.legend.set_title("Rekisteri")
plot.set_axis_labels("PC1", "PC2")

plt.savefig(
    workdir / f"pca-{experiment}{tokens_per_chunk}.svg",
    format="svg",
)

plt.show()
```
