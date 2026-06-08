import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import requests
import pickle



# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("Movie Recommender")
st.write("Rate the movies you've seen using the stars, then click *Get Recommendations*.")
st.caption("Leave a movie unrated if you haven't seen it — we'll skip it.")

# ── Model architecture ────────────────────────────────────────────────────────
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

# ── Load model + data ─────────────────────────────────────────────────────────
@st.cache_resource
def load_all():
    device = torch.device("cpu")
    model = torch.load("two_tower_model.pt", map_location=device, weights_only=False)
    model.eval()
    with open("item_categories.pkl", "rb") as f:
        item_categories = pickle.load(f)
    movies_df = pd.read_csv("movies.csv")
    return model, device, item_categories, movies_df

model, device, item_categories, movies_df = load_all()

# ── TMDB poster fetching ──────────────────────────────────────────────────────
TMDB_TOKEN = st.secrets["TMDB_TOKEN"]

def get_poster_bytes(title, year=""):
    query = title.split("(")[0].strip()
    url = "https://api.themoviedb.org/3/search/movie"
    headers = {"Authorization": f"Bearer {TMDB_TOKEN}"}
    params = {"query": query, "language": "en-US"}
    if year:
        params["year"] = year
    try:
        res = requests.get(url, headers=headers, params=params, timeout=5)
        data = res.json()
        if data.get("results"):
            poster_path = data["results"][0].get("poster_path")
            if poster_path:
                img_url = f"https://image.tmdb.org/t/p/w200{poster_path}"
                img_res = requests.get(img_url, timeout=5)
                return img_res.content  # returns raw bytes
    except:
        pass
    return None


@st.cache_data
def fetch_all_posters(movies):
    posters = {}
    for item_idx, title in movies:
        try:
            year = title.split("(")[-1].replace(")", "").strip()
        except:
            year = ""
        posters[item_idx] = get_poster_bytes(title, year)
    return posters
   
# ── Helper functions ──────────────────────────────────────────────────────────
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

# ── Movie list ────────────────────────────────────────────────────────────────
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

# Fetch all posters once
posters = fetch_all_posters(MOVIES)

# ── Rating section ────────────────────────────────────────────────────────────
st.divider()
st.subheader("Rate the movies you've seen")

ratings = {}
for item_idx, title in MOVIES:
    col1, col2 = st.columns([1, 3])

    with col1:
        poster_url = posters.get(item_idx)
        if poster_url:
            st.image(poster_url, use_container_width=True)
        else:
            st.write("🎬")

    with col2:
        st.write("")
        st.write("")
        st.markdown(f"### {title}")
        star = st.feedback("stars", key=f"star_{item_idx}")
        if star is not None:
            ratings[item_idx] = star + 1  # st.feedback returns 0-4, we want 1-5

    st.divider()

# ── Submit button ─────────────────────────────────────────────────────────────
if st.button(" Get Recommendations", use_container_width=True):
    if len(ratings) == 0:
        st.warning("Please rate at least one movie first.")
    else:
        with st.spinner("Finding your recommendations..."):
            rated_items = list(ratings.items())
            u_vec = get_new_user_vector(rated_items, model, device)
            recs  = recommend_new_user(u_vec, model, device,
                                       item_categories, movies_df, k=10)

        st.subheader(" Your Top 10 Recommendations")
        for i, rec in enumerate(recs, 1):
            col1, col2 = st.columns([1, 3])

            with col1:
                try:
                    year = rec["title"].split("(")[-1].replace(")", "").strip()
                except:
                    year = ""
                poster_url = get_poster(rec["title"], year)
                if poster_url:
                    st.image(poster_url, user_container_width=100)
                else:
                    st.write("")

            with col2:
                st.write("")
                st.write("")
                st.markdown(f"###{i}. {rec['title']}")
                st.write(f"{rec['genre']}")
                st.write(f"Predicted rating: {rec['score']:.1f}")

            st.divider()
