import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
TMDB_API_KEY = "f2ce123753b4b0f45a81d56187d920fd"
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def fetch_movie_details(movie_id):
    """Fetch full movie details from TMDB including cast, director, certification, box office"""
    movie_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    credits_url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits"
    release_url = f"https://api.themoviedb.org/3/movie/{movie_id}/release_dates"
    
    movie_r = requests.get(movie_url, params={"api_key": TMDB_API_KEY}).json()
    credits_r = requests.get(credits_url, params={"api_key": TMDB_API_KEY}).json()
    release_r = requests.get(release_url, params={"api_key": TMDB_API_KEY}).json()
    
    director = ""
    cast_list = []
    for member in credits_r.get("crew", []):
        if member.get("job") == "Director":
            director = member.get("name")
    for c in credits_r.get("cast", [])[:5]:
        cast_list.append(c.get("name"))
    
    certification = ""
    for rd in release_r.get("results", []):
        if rd.get("iso_3166_1") == "US":
            if rd.get("release_dates"):
                certification = rd["release_dates"][0].get("certification", "")
    
    return {
        "title": movie_r.get("title", ""),
        "release_year": movie_r.get("release_date","")[:4],
        "runtime": movie_r.get("runtime", 0),
        "genres": ", ".join([g['name'] for g in movie_r.get("genres",[])]),
        "overview": movie_r.get("overview",""),
        "vote_average": movie_r.get("vote_average",0),
        "vote_count": movie_r.get("vote_count",0),
        "language": movie_r.get("original_language",""),
        "director": director,
        "cast": ", ".join(cast_list),
        "revenue": movie_r.get("revenue",0),
        "certification": certification,
        "id": movie_r.get("id")
    }

def fetch_movies_by_search(query, num_pages=1):
    movies = []
    for page in range(1,num_pages+1):
        url = "https://api.themoviedb.org/3/search/movie"
        params = {"api_key": TMDB_API_KEY, "query": query, "page": page}
        r = requests.get(url, params=params).json()
        for m in r.get("results",[]):
            movies.append(fetch_movie_details(m['id']))
    return pd.DataFrame(movies)

def fetch_trending_movies(time_window="week"):
    url = f"https://api.themoviedb.org/3/trending/movie/{time_window}"
    r = requests.get(url, params={"api_key": TMDB_API_KEY}).json()
    movies = []
    for m in r.get("results",[]):
        movies.append(fetch_movie_details(m['id']))
    return pd.DataFrame(movies)

def preprocess_texts(df, col='overview'):
    df[col] = df[col].fillna('')
    return df

def compute_cosine_sim(df, col='overview'):
    df[col] = df[col].fillna('')
    if df.empty or df[col].str.strip().eq('').all():
        return np.array([[]])
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df[col])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

# -----------------------------
# RECOMMENDATIONS
# -----------------------------
def get_recommendations(title_query, movies_df, cosine_sim, top_n=5):
    if movies_df.empty or cosine_sim.size == 0:
        return pd.DataFrame()
    if title_query not in movies_df['title'].values:
        st.warning(f"Movie '{title_query}' not found in filtered results.")
        return pd.DataFrame()
    idx = movies_df[movies_df['title']==title_query].index[0]
    if idx >= cosine_sim.shape[0]:
        return pd.DataFrame()
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse=True)
    top_indices = [i for i, s in sim_scores[1:top_n+1] if i < movies_df.shape[0]]
    return movies_df.iloc[top_indices][['title','release_year','vote_average','genres','director','cast']]

def search_by_plot(plot_query, movies_df, top_n=5):
    if movies_df.empty or plot_query.strip()=="":
        return pd.DataFrame()
    plots = movies_df['overview'].tolist()
    if all(p.strip()=="" for p in plots):
        return pd.DataFrame()
    embeddings = EMBEDDING_MODEL.encode(plots)
    query_emb = EMBEDDING_MODEL.encode([plot_query])[0]
    sims = cosine_similarity([query_emb], embeddings)[0]
    top_idx = np.argsort(sims)[::-1][:top_n]
    return movies_df.iloc[top_idx][['title','release_year','vote_average','genres']]

# -----------------------------
# STREAMLIT APP
# -----------------------------
st.set_page_config(page_title="Movie Recommender Pro", layout="wide")
st.title("🎬 Movie Recommendation System")

# Sidebar filters
st.sidebar.header("Filters")
year_min, year_max = st.sidebar.slider("Release Year", 1900, 2030, (2000,2025))
vote_min, vote_max = st.sidebar.slider("Vote Average", 0.0, 10.0, (0.0,10.0))
runtime_min, runtime_max = st.sidebar.slider("Runtime (minutes)", 0, 300, (0,300))
certification = st.sidebar.text_input("Certification (e.g., PG-13)")
language = st.sidebar.text_input("Language code (e.g., en)")
genre_filter = st.sidebar.text_input("Genre (comma-separated)")
director_filter = st.sidebar.text_input("Director")
actor_filter = st.sidebar.text_input("Actor")

# Movie inputs
title_query = st.text_input("Enter a movie title for content-based recommendations")
plot_query = st.text_input("Describe a plot you like for NLP recommendations")
search_query = st.text_input("Search for movies (optional)")

# Load movies
if st.button("Load Movies"):
    if search_query.strip() != "":
        df_movies = fetch_movies_by_search(search_query, num_pages=1)
        df_movies = preprocess_texts(df_movies)
        st.session_state['movies_df'] = df_movies
        st.success(f"Loaded {len(df_movies)} movies")
    else:
        st.info("No search query entered. Loading trending movies instead.")
        df_movies = fetch_trending_movies()
        df_movies = preprocess_texts(df_movies)
        st.session_state['movies_df'] = df_movies
        st.success(f"Loaded {len(df_movies)} trending movies")

# Use cached movies
movies_df = st.session_state.get('movies_df', pd.DataFrame())

# -----------------------------
# FILTERS
# -----------------------------
if not movies_df.empty:
    movies_df['release_year'] = pd.to_numeric(movies_df['release_year'], errors='coerce')

    filtered_df = movies_df[
        (movies_df['release_year'].notna()) &
        (movies_df['release_year'] >= year_min) &
        (movies_df['release_year'] <= year_max) &
        (movies_df['vote_average'] >= vote_min) &
        (movies_df['vote_average'] <= vote_max) &
        (movies_df['runtime'] >= runtime_min) &
        (movies_df['runtime'] <= runtime_max)
    ]
    if certification.strip()!="":
        filtered_df = filtered_df[filtered_df['certification'].str.contains(certification)]
    if language.strip()!="":
        filtered_df = filtered_df[filtered_df['language']==language.strip()]
    if genre_filter.strip()!="":
        genres = [g.strip().lower() for g in genre_filter.split(",")]
        filtered_df = filtered_df[filtered_df['genres'].str.lower().apply(lambda x: any(g in x for g in genres))]
    if director_filter.strip()!="":
        filtered_df = filtered_df[filtered_df['director'].str.contains(director_filter, case=False)]
    if actor_filter.strip()!="":
        filtered_df = filtered_df[filtered_df['cast'].str.contains(actor_filter, case=False)]

    cosine_sim = compute_cosine_sim(filtered_df)
    if cosine_sim.size == 0:
        st.warning("No valid plot descriptions available for content-based recommendations.")
    else:
        # Content-based recommendations
        if title_query:
            st.subheader(f"Content-Based Recommendations for '{title_query}'")
            recs = get_recommendations(title_query, filtered_df, cosine_sim)
            if not recs.empty:
                st.dataframe(recs)
            else:
                st.info("No recommendations found.")

    # Plot similarity
    if plot_query:
        st.subheader("Plot-Based Recommendations")
        plot_recs = search_by_plot(plot_query, filtered_df)
        if not plot_recs.empty:
            st.dataframe(plot_recs)
        else:
            st.info("No plot-based recommendations found.")

    # Watchlist
    if 'watchlist' not in st.session_state:
        st.session_state['watchlist'] = []
    st.subheader("Watchlist")
    selected_movies = st.multiselect("Add movies to watchlist", filtered_df['title'].tolist())
    for m in selected_movies:
        if m not in st.session_state['watchlist']:
            st.session_state['watchlist'].append(m)
    st.write(st.session_state['watchlist'])

    # Visualization
    st.subheader("Ratings Distribution")
    fig, ax = plt.subplots()
    filtered_df['vote_average'].hist(ax=ax, bins=10)
    st.pyplot(fig)

    st.subheader("Genre Distribution")
    genres_series = filtered_df['genres'].str.split(', ').explode()
    fig2, ax2 = plt.subplots()
    genres_series.value_counts().plot(kind='bar', ax=ax2)
    st.pyplot(fig2)

else:
    st.info("Load some movies first using the buttons above.")
