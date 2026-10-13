"""
UE25MA242A Mini Project - Problem 13
Movie taste clustering on MovieLens 100K using Linear Algebra

Install once:  pip install numpy scipy pandas sympy scikit-learn matplotlib
Run:           python movie_spectral_clustering.py
(The MovieLens 100K dataset is downloaded automatically on first run.)
"""
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import sympy as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import qr, eigh
from sklearn.cluster import KMeans

np.set_printoptions(precision=3, suppress=True, linewidth=120)
RNG = np.random.default_rng(42)

N_USERS, N_MOVIES = 200, 200
K_BASIS = 10
K_CLUSTERS = 3
K_NN = 15


def header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


header("STAGE 1: Real-world data -> user x movie ratings matrix")

URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
if not os.path.exists("ml-100k/u.data"):
    print("Downloading MovieLens 100K ...")
    urllib.request.urlretrieve(URL, "ml-100k.zip")
    with zipfile.ZipFile("ml-100k.zip") as z:
        z.extractall(".")

ratings = pd.read_csv("ml-100k/u.data", sep="\t",
                      names=["user", "movie", "rating", "ts"])
titles = pd.read_csv("ml-100k/u.item", sep="|", encoding="latin-1",
                     usecols=[0, 1], names=["movie", "title"], header=None)
title_of = dict(zip(titles.movie, titles.title))


top_movies = ratings.movie.value_counts().head(N_MOVIES).index
ratings = ratings[ratings.movie.isin(top_movies)]
top_users = ratings.user.value_counts().head(N_USERS).index
ratings = ratings[ratings.user.isin(top_users)]

table = ratings.pivot(index="user", columns="movie", values="rating")
user_ids = table.index.to_numpy()
movie_ids = table.columns.to_numpy()

mask = table.notna().to_numpy()
R = table.fillna(0).to_numpy(dtype=float)

print("Ratings matrix R shape (users x movies):", R.shape)
print("Observed ratings:", int(mask.sum()),
      f"({100 * mask.mean():.1f}% of the matrix is filled)")


user_mean = R.sum(axis=1) / mask.sum(axis=1)
Rc = (R - user_mean[:, None]) * mask
print("Centered matrix Rc built. First 5x5 block:\n", Rc[:5, :5])


header("STAGE 2: RREF on a small dense block of the ratings matrix")

busy_users = np.argsort(-mask.sum(axis=1))[:6]
popular_movies = np.argsort(-mask.sum(axis=0))[:6]
block = R[np.ix_(busy_users, popular_movies)].astype(int)

M = sp.Matrix(block.tolist())
print("6x6 block of raw ratings:")
sp.pprint(M)
rref_M, pivots = M.rref()
print("\nRREF:")
sp.pprint(rref_M)
print("Pivot columns:", pivots)
print("Rank of this block =", len(pivots))
if len(pivots) == block.shape[1]:
    print("Meaning: all 6 movie columns are independent in this block (no\n"
          "redundancy here); try a larger block to see dependent columns.")
else:
    print("Meaning: pivot columns are independent movies; non-pivot columns are\n"
          "combinations of the pivot ones (redundant taste information).")


header("STAGE 3: Rank and nullity")

rank_R = np.linalg.matrix_rank(Rc)
print("rank(Rc)    =", rank_R)
print("nullity(Rc) = columns - rank =", Rc.shape[1] - rank_R)

sv = np.linalg.svd(Rc, compute_uv=False)
print("Singular values (first 10):", sv[:10])
print("Energy captured by the first 10 directions:",
      f"{100 * (sv[:10] ** 2).sum() / (sv ** 2).sum():.1f}%")
print("Meaning: the larger the energy share of the first few directions, the\n"
      "more the 200 ratings per user can be summarised by a few taste patterns.\n"
      "(Real ratings are noisy, so expect a modest share, not 100%.)")


header("STAGE 4: Basis selection (most independent movie columns)")


_, _, piv = qr(Rc, pivoting=True, mode="economic")
basis_idx = piv[:K_BASIS]
B = Rc[:, basis_idx]
print("Chosen basis movies:")
for j in basis_idx:
    print("  -", title_of[movie_ids[j]])
print("rank(B) =", np.linalg.matrix_rank(B), "(= K_BASIS, so they are independent)")


header("STAGE 5: Gram-Schmidt orthogonalization")


def gram_schmidt(A):
    """Return Q with orthonormal columns spanning the same space as A."""
    Q = np.zeros_like(A, dtype=float)
    for j in range(A.shape[1]):
        v = A[:, j].astype(float).copy()
        for i in range(j):
            v -= (Q[:, i] @ v) * Q[:, i]
        Q[:, j] = v / np.linalg.norm(v)
    return Q


Q = gram_schmidt(B)
print("Q^T Q (should be the identity):\n", Q.T @ Q)
print("Max deviation from identity:", np.abs(Q.T @ Q - np.eye(K_BASIS)).max())


header("STAGE 6: Projection onto the span of the basis")

non_basis = [j for j in np.argsort(-mask.sum(axis=0)) if j not in set(basis_idx)]
target = non_basis[0]
y = Rc[:, target]
proj = Q @ (Q.T @ y)
resid = y - proj
print("Target movie:", title_of[movie_ids[target]])
print("||y||        =", np.linalg.norm(y))
print("||projection||=", np.linalg.norm(proj))
print("||residual|| =", np.linalg.norm(resid))
print("Residual orthogonal to subspace? max|Q^T r| =", np.abs(Q.T @ resid).max())
print("Meaning: the part of this movie's rating pattern explained by the\n"
      "basis movies is the projection; the rest is the residual.")


header("STAGE 7: Least squares prediction of held-out ratings")


w_ls, *_ = np.linalg.lstsq(B, y, rcond=None)
print("Least squares fit equals projection onto span(B)?",
      np.allclose(B @ w_ls, proj))

targets = non_basis[:30]
rmse_ls, rmse_base = [], []
for t in targets:
    rated = np.where(mask[:, t])[0]
    RNG.shuffle(rated)
    cut = int(0.8 * len(rated))
    train, test = rated[:cut], rated[cut:]
    yt = Rc[:, t]
    w, *_ = np.linalg.lstsq(B[train], yt[train], rcond=None)
    pred = np.clip(B[test] @ w + user_mean[test], 1, 5)
    actual = R[test, t]
    base = user_mean[test]
    rmse_ls.append(np.sqrt(np.mean((pred - actual) ** 2)))
    rmse_base.append(np.sqrt(np.mean((base - actual) ** 2)))

print(f"Average RMSE, baseline (user's mean rating): {np.mean(rmse_base):.3f}")
print(f"Average RMSE, least squares on basis       : {np.mean(rmse_ls):.3f}")
if np.mean(rmse_ls) < np.mean(rmse_base):
    print("Meaning: using the basis movies predicts unseen ratings better than\n"
          "just guessing each user's average (lower RMSE = better).")
else:
    print("Meaning: here the basis did not beat the baseline. Try a smaller\n"
          "K_BASIS or more users (fewer weights / more training rows).")


header("STAGE 8: User similarity graph and Laplacian eigenvalues")


norms = np.linalg.norm(Rc, axis=1, keepdims=True) + 1e-12
Rn = Rc / norms
S = Rn @ Rn.T
print("Similarity matrix S shape:", S.shape, "| symmetric:", np.allclose(S, S.T))


W = np.clip(S, 0, None)
np.fill_diagonal(W, 0)


W_knn = np.zeros_like(W)
nearest = np.argsort(-W, axis=1)[:, :K_NN]
for i in range(len(W)):
    W_knn[i, nearest[i]] = W[i, nearest[i]]
W = np.maximum(W_knn, W_knn.T)
print(f"k-NN graph built: each user keeps {K_NN} nearest neighbours.")

d = W.sum(axis=1)
D_inv_sqrt = 1.0 / np.sqrt(d + 1e-12)


L = np.eye(len(W)) - D_inv_sqrt[:, None] * W * D_inv_sqrt[None, :]
print("Laplacian symmetric:", np.allclose(L, L.T))


header("STAGE 9: Diagonalization  L = Q Lambda Q^T")

vals, vecs = eigh(L)
print("Smallest 10 eigenvalues:", vals[:10])
print("L = Q Lambda Q^T reconstructs L?",
      np.allclose(vecs @ np.diag(vals) @ vecs.T, L))
print("Eigenvectors orthonormal?", np.allclose(vecs.T @ vecs, np.eye(len(vals))))
print("All eigenvalues >= 0 (positive semidefinite)?", vals.min() > -1e-9)

plt.figure(figsize=(6, 3.5))
plt.plot(range(1, 16), vals[:15], "o-")
plt.axvline(K_CLUSTERS + 0.5, color="r", ls="--", label=f"k = {K_CLUSTERS}")
plt.xlabel("index"); plt.ylabel("eigenvalue")
plt.title("Smallest Laplacian eigenvalues (eigengap)")
plt.legend(); plt.tight_layout()
plt.savefig("eigenvalues.png", dpi=150)
print("Saved eigenvalues.png  (a big jump after k values suggests k clusters)")


header("STAGE 10: Spectral clustering of users + recommendations")


U = vecs[:, :K_CLUSTERS]
U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)

labels = KMeans(n_clusters=K_CLUSTERS, n_init=10, random_state=42).fit_predict(U)
print("Cluster sizes:", np.bincount(labels))


plt.figure(figsize=(6, 5))
sc = plt.scatter(U[:, 1], U[:, 2], c=labels, cmap="tab10", s=18)
plt.xlabel("eigenvector 2"); plt.ylabel("eigenvector 3")
plt.title("Users in spectral space, coloured by cluster")
plt.legend(*sc.legend_elements(), title="cluster")
plt.tight_layout(); plt.savefig("clusters.png", dpi=150)
print("Saved clusters.png")


for c in range(K_CLUSTERS):
    members = labels == c
    n_rate = mask[members].sum(axis=0)
    mean_rate = np.where(n_rate > 0, R[members].sum(axis=0) / np.maximum(n_rate, 1), 0)
    ok = n_rate >= 5
    ranked = np.argsort(-(mean_rate * ok))[:5]
    print(f"\nCluster {c}  ({members.sum()} users) - top movies:")
    for j in ranked:
        print(f"   {mean_rate[j]:.2f}  {title_of[movie_ids[j]]}")


def recommend(user_row, n=5):
    """Top-n unseen movies for a user, ranked by their cluster's mean rating."""
    c = labels[user_row]
    members = labels == c
    n_rate = mask[members].sum(axis=0)
    mean_rate = np.where(n_rate > 0, R[members].sum(axis=0) / np.maximum(n_rate, 1), 0)
    score = mean_rate * (n_rate >= 5) * (~mask[user_row])
    return [title_of[movie_ids[j]] for j in np.argsort(-score)[:n]]


row_of = {int(u): i for i, u in enumerate(user_ids)}

print("\n--- Interactive recommender ---")
print("Sample valid user IDs:", [int(u) for u in user_ids[:10]], "...")
while True:
    text = input("\nEnter a user ID (or 'q' to quit): ").strip()
    if text.lower() in ("q", "quit", "exit", ""):
        break
    if not text.isdigit() or int(text) not in row_of:
        print("  That ID is not in our 200-user subset. Try one of the sample IDs.")
        continue
    r = row_of[int(text)]
    liked = np.argsort(-(R[r] * mask[r]))[:3]
    print(f"User {text} belongs to cluster {labels[r]} "
          f"(cluster size {np.sum(labels == labels[r])}).")
    print("Movies this user rated highest:")
    for j in liked:
        print(f"   {R[r, j]:.0f}  {title_of[movie_ids[j]]}")
    print("Recommended for this user:")
    for t in recommend(r):
        print("   -", t)

print("\nDone. Files created: eigenvalues.png, clusters.png")