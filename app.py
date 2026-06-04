import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Movie Recommender", page_icon="🎬")
st.title("🎬 Movie Recommender")
st.write("Rate a few movies you've seen, then click *Get Recommendations*.")
st.caption("Leave a movie at 0 if you haven't seen it — we'll skip it.")

# ── Model architecture (must match exactly what was trained) ──────────────────
class UserTower(nn.Module):
    def _init_(self, U, D):
        super()._init_()
        self.emb = nn.Embedding(U, D)
        self.mlp = nn.Sequential(nn.Linear(D, D), nn.ReLU())
    def forward(self, u):
        return self.mlp(self.emb(u))

class ItemTower(nn.Module):
    def _init_(self, I, D):
        super()._init_()
        self.emb = nn.Embedding(I, D)
        self.mlp = nn.Sequential(nn.Linear(D, D), nn.ReLU())
    def forward(self, i):
        return self.mlp(self.emb(i))

class TwoTowerRegressor(nn.Module):
    def _init_(self, U, I, D=64):
        super()._init_()
        self.user_tower  = UserTower(U, D)
        self.item_tower  = ItemTower(I, D)
        self.user_bias   = nn.Embedding(U, 1)
        self.item_bias   = nn.Embedding(I, 1)
        self.global_bias = nn.Parameter(torch.tensor([0.0]))
    def forward(self, u, i):
        u_vec = self.user_tower(u)
        i_vec = self.item_tower(i)
        dot   = (u_vec * i_vec).sum(dim=1)
        bias  = self.user_bias(u).squeeze() + self.item_bias(i).squeeze() + self.global_bias
        return dot + bias

# ── Load everything once ──────────────────────────────────────────────────────
@st.cache_resource
def load_all():
    device = torch.device("cpu")

    model = torch.load("two_tower_model.pt", map_location=device)
    model.eval()

    ratings_df = pd.read_csv(
        "ml-1m/ratings.dat",
        sep="::", header=None,
        names=["userId", "movieId", "rating", "timestamp"],
        engine="python"
    )
    movies_df = pd.read_csv(
        "ml-1m/movies.dat",
        sep="::", header=None,
        names=["movieId", "title", "genres"],
        engine="python", encoding="latin-1"
    )
    item_categories = ratings_df.movieId.astype("category").cat.categories
    return model, device, item_categories, movies_df

model, device, item_categories, movies_df = load_all()

# ── Helper functions (from notebook) ─────────────────────────────────────────
def get_new_user_vector(rated_items, model, device, D=64, steps=300):
    model.eval()
    u_vec = nn.Parameter(torch.randn(1, D, device=device))
    opt   = optim.Adam([u_vec], lr=0.05)

    for param in model.parameters():
        param.requires_grad = False

    for _ in range(steps):
        opt.zero_grad()
        loss = torch.tensor(0.0, device=device)
        for item_idx, rating in rated_items:
            i     = torch.tensor([item_idx], device=device)
            i_vec = model.item_tower(i)
            pred  = (u_vec * i_vec).sum()
            loss += (pred - rating) ** 2
        loss.backward()
        opt.step()

    for param in model.parameters():
        param.requires_grad = True

    return u_vec.detach()


def recommend_new_user(u_vec, model, device, item_categories, movies_df, k=10):
    model.eval()
    with torch.no_grad():
        all_i  = torch.arange(len(item_categories), device=device)
        i_vecs = model.item_tower(all_i)
        preds  = (u_vec * i_vecs).sum(dim=1).cpu().numpy()

    topk = np.argsort(preds)[-k:][::-1]
    results = []
    for idx in topk:
        raw_id = item_categories[idx]
        title  = movies_df.loc[movies_df.movieId == raw_id, "title"].values[0]
        genre  = movies_df.loc[movies_df.movieId == raw_id, "genres"].values[0]
        results.append({"title": title, "score": float(preds[idx]), "genre": genre})
    return results

# ── Top 20 popular movies for rating ─────────────────────────────────────────
MOVIES = [
    (2689, "American Beauty (1999)"),
    (2762, "The Sixth Sense (1999)"),
    (260,  "Star Wars: Episode IV (1977)"),
    (1196, "Star Wars: Episode V (1980)"),
    (1210, "Star Wars: Episode VI (1983)"),
    (589,  "Terminator 2: Judgment Day (1991)"),
    (593,  "The Silence of the Lambs (1991)"),
    (527,  "Schindler's List (1993)"),
    (480,  "Jurassic Park (1993)"),
    (608,  "Fargo (1996)"),
    (50,   "The Usual Suspects (1995)"),
    (318,  "The Shawshank Redemption (1994)"),
    (858,  "The Godfather (1972)"),
    (1,    "Toy Story (1995)"),
    (3114, "Toy Story 2 (1999)"),
    (2396, "Shakespeare in Love (1998)"),
    (2571, "The Matrix (1999)"),
    (1265, "Groundhog Day (1993)"),
    (1580, "Men in Black (1997)"),
    (2028, "Saving Private Ryan (1998)"),
]

# ── Rating sliders ────────────────────────────────────────────────────────────
st.divider()
ratings = {}
for item_idx, title in MOVIES:
    score = st.slider(title, min_value=0, max_value=5, value=0, step=1)
    if score > 0:
        ratings[item_idx] = score

# ── Submit ────────────────────────────────────────────────────────────────────
st.divider()
if st.button("🎯 Get Recommendations", use_container_width=True):
    if len(ratings) == 0:
        st.warning("Please rate at least one movie first.")
    else:
        with st.spinner("Finding your recommendations..."):
            rated_items = list(ratings.items())
            u_vec = get_new_user_vector(rated_items, model, device)
            recs  = recommend_new_user(u_vec, model, device,
                                       item_categories, movies_df, k=10)

        st.subheader("🍿 Your Top 10 Recommendations")
        for i, rec in enumerate(recs, 1):
            st.write(f"*{i}. {rec['title']}* — {rec['genre']}")