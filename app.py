import ast
from pathlib import Path

import html

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Book Matchmaker",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #eae5cf;
    }
    [data-testid="stSidebar"] * {
        color: #754215 !important;
    }
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stSelectbox label {
        font-size: 0.8rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #a05a20 !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA + MODEL CONFIG
# ─────────────────────────────────────────────
DATASETS = {
    "thrillers.csv": "data/thrillers.csv",
    "books_dataset.csv": "data/books_dataset.csv",
    "200thrillers.csv": "data/200thrillers.csv",
}
SEMANTIC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CACHED_200THRILLERS_MODELS = {
    "all-MiniLM-L6-v2": "all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
}
CACHED_200THRILLERS_DATASET = DATASETS["200thrillers.csv"]
CACHED_200THRILLERS_PATH = Path(CACHED_200THRILLERS_DATASET).resolve()
CACHE_DIR = Path(".cache/book_matchmaker")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
BOOK_SIMILARITY_CACHE_DIR = Path(".cache/book_similarity")
BOOK_SIMILARITY_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# DATA LOADING  — cached so it only runs once
# ─────────────────────────────────────────────
def _parse_genres(value) -> str:
    if pd.isna(value):
        return ""

    if isinstance(value, list):
        return ", ".join(str(item).strip() for item in value if str(item).strip())

    text = str(value).strip()
    if not text:
        return ""

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = []

        if isinstance(parsed, list):
            return ", ".join(
                str(item).strip().strip("'").strip('"')
                for item in parsed
                if str(item).strip()
            )

    return text


def _normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "genres" in df.columns:
        df["genres"] = df["genres"].fillna("").map(_parse_genres)
    elif "genres_list" in df.columns:
        df["genres"] = df["genres_list"].fillna("").map(_parse_genres)
    else:
        df["genres"] = ""

    if "num_pages" in df.columns:
        df["num_pages"] = pd.to_numeric(df["num_pages"], errors="coerce")
    else:
        df["num_pages"] = pd.NA

    if "original_publication_year" in df.columns:
        df["original_publication_year"] = pd.to_numeric(
            df["original_publication_year"], errors="coerce"
        )
    else:
        df["original_publication_year"] = pd.NA

    if "avg_rating" in df.columns:
        df["avg_rating"] = pd.to_numeric(df["avg_rating"], errors="coerce")
    else:
        df["avg_rating"] = pd.NA

    if "ratings_count" in df.columns:
        df["ratings_count"] = pd.to_numeric(df["ratings_count"], errors="coerce")
    else:
        df["ratings_count"] = pd.NA

    if "author" not in df.columns:
        df["author"] = ""
    else:
        df["author"] = df["author"].fillna("").astype(str).str.strip()

    if "image_url" not in df.columns:
        df["image_url"] = ""

    df["original_title"] = df["original_title"].fillna("Unknown Title").astype(str).str.strip()
    df["description"] = df["description"].fillna("").astype(str).str.strip()
    df = df[df["description"] != ""].reset_index(drop=True)
    return df


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _normalize_schema(df)


@st.cache_data
def get_all_genres(df: pd.DataFrame) -> list:
    """Extract and sort unique genres across all books."""
    genres = (
        df['genres']
        .str.split(',')
        .explode()
        .str.strip()
        .replace('', pd.NA)
        .dropna()
        .unique()
        .tolist()
    )
    return sorted(genres)


def _embedding_cache_paths(path: str, model_name: str) -> tuple[Path, Path]:
    source_path = Path(path)
    if source_path.resolve() == CACHED_200THRILLERS_PATH:
        cache_dir = BOOK_SIMILARITY_CACHE_DIR
    else:
        cache_dir = CACHE_DIR
    key = f"{source_path.stem}__{model_name.replace('/', '__')}"
    return cache_dir / f"{key}.npy", cache_dir / f"{key}.json"


@st.cache_resource(show_spinner=False)
def get_sentence_model(model_name: str) -> object:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def build_path_token(path: str) -> str:
    source_path = Path(path)
    return f"{source_path.resolve()}::{source_path.stat().st_mtime_ns}"


@st.cache_data(show_spinner=False)
def get_semantic_index(path: str, model_name: str, source_token: str) -> tuple[pd.DataFrame, np.ndarray]:
    df = load_data(path)
    embeddings_path, meta_path = _embedding_cache_paths(path, model_name)

    if embeddings_path.exists() and meta_path.exists():
        return df, np.load(embeddings_path)

    model = get_sentence_model(model_name)
    embeddings = model.encode(
        df["description"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    np.save(embeddings_path, embeddings)
    meta_path.write_text(
        pd.Series(
            {
                "path": path,
                "model_name": model_name,
                "source_token": source_token,
                "rows": len(df),
            }
        ).to_json(indent=2),
        encoding="utf-8",
    )
    return df, embeddings


def search_books(
    query_text: str,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    model_name: str,
    top_n: int,
    min_similarity: float,
) -> pd.DataFrame:
    query_text = query_text.strip()
    if not query_text:
        raise ValueError("Please enter a description to search.")

    model = get_sentence_model(model_name)
    query_embedding = model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    scores = embeddings @ query_embedding
    ranked_indices = np.argsort(scores)[::-1]
    ranked_indices = [idx for idx in ranked_indices if scores[idx] >= min_similarity]
    top_indices = ranked_indices[:top_n]

    results = df.iloc[top_indices].copy()
    results["similarity"] = scores[top_indices]
    results["rank"] = range(1, len(results) + 1)
    return results.reset_index(drop=True)


# ─────────────────────────────────────────────
# FILTERING LOGIC  — pure function, no UI calls
# ─────────────────────────────────────────────
def apply_filters(
    df: pd.DataFrame,
    selected_genres: list,
    page_range: tuple,
    year_range: tuple,
    min_rating: float,
    min_ratings_count: int,
) -> pd.DataFrame:

    mask = pd.Series(True, index=df.index)

    # Genre filter: book must contain ALL selected genres
    if selected_genres:
        genre_mask = df['genres'].apply(
            lambda g: all(genre.strip() in g for genre in selected_genres)
        )
        mask &= genre_mask

    # Page length filter
    if df['num_pages'].notna().any():
        mask &= (df['num_pages'].fillna(0).between(page_range[0], page_range[1]))

    # Publication year range
    mask &= (
        df['original_publication_year'].fillna(0).between(year_range[0], year_range[1])
    )

    # Minimum average rating
    mask &= (df['avg_rating'].fillna(0) >= min_rating)

    # Minimum popularity (ratings count)
    mask &= (df['ratings_count'].fillna(0) >= min_ratings_count)

    return df[mask].copy()


# ─────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────
def build_book_card_html(row: pd.Series) -> str:
    """Return HTML string for a single book card in the grid."""

    image_url = str(row.get('image_url', ''))
    if pd.notna(row.get('image_url')) and image_url.startswith('http'):
        cover_html = f'<img src="{image_url}" alt="cover" class="book-cover-img">'
    else:
        cover_html = '<div class="cover-fallback">📖</div>'

    title  = html.escape(str(row.get('original_title', 'Unknown Title')))
    author = html.escape(str(row.get('author', 'Unknown Author')))
    description = html.escape(str(row.get('description', '')))
    if len(description) > 420:
        description = f"{description[:417].rstrip()}..."
    rating = f"★ {row['avg_rating']:.2f}" if pd.notna(row.get('avg_rating')) else ''
    similarity = row.get('similarity')
    similarity_html = (
        f'<p class="ov-score">Match {float(similarity) * 100:.0f}%</p>'
        if pd.notna(similarity)
        else ''
    )

    return f"""
    <div class="book-card">
        <div class="cover-wrap">
            {cover_html}
            <div class="overlay">
                <p class="ov-title">{title}</p>
                <p class="ov-author">by {author}</p>
                <p class="ov-rating">{rating}</p>
                {similarity_html}
                <p class="ov-desc">{description}</p>
            </div>
        </div>
    </div>
    """


def render_book_grid(results: pd.DataFrame):
    cards_html = ''.join(
        build_book_card_html(row) for _, row in results.iterrows()
    )
    full_html = f"""
    <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ background: transparent; font-family: sans-serif; }}
    .book-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
        gap: 1.2rem;
        padding: 0.5rem;
    }}
    .book-card {{
        position: relative;
        border-radius: 8px;
        overflow: hidden;
        aspect-ratio: 2 / 3;
        box-shadow: 3px 5px 14px rgba(0,0,0,0.18);
        transition: transform 0.22s ease, box-shadow 0.22s ease;
    }}
    .book-card:hover {{
        transform: translateY(-6px) scale(1.03);
        box-shadow: 6px 12px 28px rgba(0,0,0,0.28);
    }}
    .cover-wrap {{ position: relative; width: 100%; height: 100%; }}
    .book-cover-img {{
        width: 100%; height: 100%;
        object-fit: cover; display: block; border-radius: 8px;
    }}
    .cover-fallback {{
        width: 100%; height: 100%;
        background: linear-gradient(160deg, #1a1a2e 0%, #4a3f8c 100%);
        display: flex; align-items: center;
        justify-content: center; font-size: 2.5rem; border-radius: 8px;
    }}
    .overlay {{
        position: absolute; inset: 0;
        background: linear-gradient(
            to top,
            rgba(10,8,30,0.96) 0%,
            rgba(10,8,30,0.75) 50%,
            rgba(10,8,30,0.0) 100%
        );
        display: flex; flex-direction: column;
        justify-content: flex-end;
        padding: 0.6rem 0.5rem;
        opacity: 0; transition: opacity 0.22s ease; border-radius: 8px;
    }}
    .book-card:hover .overlay {{ opacity: 1; }}
    .book-card.show-overlay .overlay {{ opacity: 1; }}
    .ov-title {{
        font-size: 0.68rem; font-weight: 700; color: #fff;
        margin: 0 0 0.1rem; line-height: 1.3;
        display: -webkit-box; -webkit-line-clamp: 2;
        -webkit-box-orient: vertical; overflow: hidden;
    }}
    .ov-author {{
        font-size: 0.62rem; color: #b0aed0;
        margin: 0 0 0.25rem; white-space: nowrap;
        overflow: hidden; text-overflow: ellipsis;
    }}
    .ov-rating {{
        font-size: 0.65rem; color: #f0c060;
        font-weight: 600; margin: 0;
    }}
    .ov-score {{
        font-size: 0.62rem; color: #8fe3c4;
        font-weight: 700; margin: 0.08rem 0 0;
        letter-spacing: 0.02em;
    }}
        .ov-desc {{
                font-size: 0.6rem;
                color: #e4e2f2;
                line-height: 1.35;
                margin-top: 0.3rem;
                display: -webkit-box;
                -webkit-line-clamp: 6;
                -webkit-box-orient: vertical;
                overflow: hidden;
        }}
    </style>
        <div class="book-grid">{cards_html}</div>
        <script>
            (function () {{
                const cards = document.querySelectorAll('.book-card');
                cards.forEach((card) => {{
                    card.addEventListener('click', () => {{
                        card.classList.toggle('show-overlay');
                    }});
                }});
            }})();
        </script>
    """
    # With aspect-ratio 2/3, card height ≈ column width × 1.5
    # At 5 columns in ~700px container, each col ≈ 130px → card height ≈ 195px
    n_cols = 5
    n_rows = max(1, -(-len(results) // n_cols))  # ceiling division
    estimated_height = n_rows * 215 + 40 # 195px card + 20px gap
    components.html(full_html, height=estimated_height, scrolling=False)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar(df: pd.DataFrame) -> dict:
    """Render all controls and return selected values as a dict."""

    with st.sidebar:
        st.markdown("## 📖 Book Matchmaker")
        st.markdown("*Discover your next favourite book*")
        mode = st.radio(
            "MODE",
            options=["Browse", "Semantic search"],
            horizontal=True,
        )

        dataset_label = st.selectbox(
            "Corpus",
            options=list(DATASETS.keys()),
            index=list(DATASETS.keys()).index("200thrillers.csv"),
        )

        active_df = load_data(DATASETS[dataset_label])

        st.divider()

        semantic_model_label = SEMANTIC_MODEL_NAME
        semantic_model_name = SEMANTIC_MODEL_NAME

        if mode == "Semantic search":
            if dataset_label == "200thrillers.csv":
                semantic_model_label = st.selectbox(
                    "Embedding model",
                    options=list(CACHED_200THRILLERS_MODELS.keys()),
                    index=0,
                )
                semantic_model_name = CACHED_200THRILLERS_MODELS[semantic_model_label]
            else:
                semantic_model_label = SEMANTIC_MODEL_NAME
                semantic_model_name = SEMANTIC_MODEL_NAME

            # top_n = st.slider("Top matches", min_value=3, max_value=50, value=8)
            # min_similarity = st.slider(
            #     "Minimum similarity",
            #     min_value=0.0,
            #     max_value=1.0,
            #     value=0.2,
            #     step=0.01,
            #     format="%.2f",
            # )
            
            top_n = 5
            min_similarity = 0.2

        all_genres = get_all_genres(active_df)
        selected_genres = st.multiselect(
            "Genres",
            options=all_genres,
            placeholder="Select all genres",
        )

        st.markdown("---")

        valid_pages = active_df['num_pages'].dropna()
        page_min = int(valid_pages.min()) if not valid_pages.empty else 1
        page_max = int(valid_pages.max()) if not valid_pages.empty else 2000
        page_range = st.slider(
            "Page Range",
            min_value=page_min,
            max_value=page_max,
            value=(page_min, page_max),
        )

        st.markdown("---")

        valid_years = active_df['original_publication_year'].dropna()
        year_min = int(valid_years.min()) if not valid_years.empty else 0
        year_max = int(valid_years.max()) if not valid_years.empty else 2025
        year_range = st.slider(
            "Publication Year",
            min_value=year_min,
            max_value=year_max,
            value=(year_min, year_max),
        )

        st.markdown("---")

        valid_ratings = active_df['avg_rating'].dropna()
        rating_min = float(valid_ratings.min()) if not valid_ratings.empty else 0.0
        rating_max = float(valid_ratings.max()) if not valid_ratings.empty else 5.0
        min_rating = st.slider(
            "Minimum Rating",
            min_value=rating_min,
            max_value=rating_max,
            value=rating_min,
            step=0.1,
            format="%.1f",
        )

        st.markdown("---")

        min_ratings_count = st.select_slider(
            "Minimum popularity",
            options=[0, 100, 500, 1000, 5000, 10000, 50000],
            value=0,
            format_func=lambda x: f"{x:,} ratings" if x > 0 else "Any",
        )

        st.markdown("---")

        sort_by = st.selectbox(
            "Sort results by",
            options=["avg_rating", "ratings_count", "original_publication_year", "num_pages"],
            format_func=lambda x: {
                "avg_rating": "⭐ Rating",
                "ratings_count": "🔥 Popularity",
                "original_publication_year": "📅 Year",
                "num_pages": "📄 Length",
            }[x],
        )

        # Prepare return dict
        result_dict = {
            "mode": mode,
            "dataset_label": dataset_label,
            "data_path": DATASETS[dataset_label],
            "semantic_model_label": semantic_model_label,
            "semantic_model_name": semantic_model_name,
            "selected_genres": selected_genres,
            "page_range": page_range,
            "year_range": year_range,
            "min_rating": min_rating,
            "min_ratings_count": min_ratings_count,
            "sort_by": sort_by,
        }

        # Add semantic search specific params
        if mode == "Semantic search":
            result_dict["top_n"] = top_n
            result_dict["min_similarity"] = min_similarity

        return result_dict


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # ── Default data ───────────────────────────
    default_path = DATASETS["200thrillers.csv"]
    df = load_data(default_path)

    # ── Pagination state ───────────────────────
    if "page" not in st.session_state:
        st.session_state.page = 1

    if "last_filters" not in st.session_state:
        st.session_state.last_filters = ""

    if "last_mode" not in st.session_state:
        st.session_state.last_mode = "Browse"

    # ── Sidebar filters ────────────────────────
    filters = render_sidebar(df)

    data_path = filters["data_path"]
    df = load_data(data_path)

    if st.session_state.last_mode != filters["mode"]:
        st.session_state.page = 1
        st.session_state.last_mode = filters["mode"]

    if filters["mode"] == "Semantic search":
        st.markdown("## Semantic Search")
        st.markdown(
            "Write a short description of the book you want to read."
        )

        query_key = f"semantic_query::{filters['dataset_label']}::{filters['semantic_model_label']}"
        if query_key not in st.session_state:
            st.session_state[query_key] = ""

        with st.form("semantic_search_form"):
            query_text = st.text_area(
                "Describe the book you want to read",
                key=query_key,
                height=160,
                placeholder="Example: A dark psychological thriller with a missing person, hidden secrets, and a tense investigation.",
                label_visibility="hidden",
            )
            submitted = st.form_submit_button("Find matches", use_container_width=True)

        if not submitted:
            st.info("Enter a description and press Find matches to see the top semantic matches.")
            return

        if not query_text.strip():
            st.warning("Please enter some text before searching.")
            return

        source_token = build_path_token(data_path)
        df, embeddings = get_semantic_index(
            data_path,
            filters["semantic_model_name"],
            source_token,
        )
        
        # Get top N semantic matches
        results = search_books(
            query_text=query_text,
            df=df,
            embeddings=embeddings,
            model_name=filters["semantic_model_name"],
            top_n=filters["top_n"],
            min_similarity=filters["min_similarity"],
        )

        if results.empty:
            st.info("No books match your description.")
            return

        # Apply filters to semantic results
        results = apply_filters(
            results,
            selected_genres=filters["selected_genres"],
            page_range=filters["page_range"],
            year_range=filters["year_range"],
            min_rating=filters["min_rating"],
            min_ratings_count=filters["min_ratings_count"],
        )

        st.markdown(
            f'<p class="result-count">{len(results):,} book{"s" if len(results) != 1 else ""} matched your description and filters</p>',
            unsafe_allow_html=True,
        )

        if results.empty:
            st.info("No books match both the semantic search and your selected filters. Try adjusting your filters.")
            return

        # Semantic mode should rank by similarity first
        results = results.sort_values("similarity", ascending=False)

        render_book_grid(results)
        return

    # ── Apply filters ──────────────────────────
    results = apply_filters(
        df,
        selected_genres=filters["selected_genres"],
        page_range=filters["page_range"],
        year_range=filters["year_range"],
        min_rating=filters["min_rating"],
        min_ratings_count=filters["min_ratings_count"],
    )

    # ── Sort ───────────────────────────────────
    results = results.sort_values(filters["sort_by"], ascending=False)

    # Auto-reset page when filters change
    filter_fingerprint = str(filters)
    if st.session_state.last_filters != filter_fingerprint:
        st.session_state.page = 1
        st.session_state.last_filters = filter_fingerprint

    # ── Main area ──────────────────────────────
    st.markdown("## Discover Your Next Read")
    st.markdown(
        f'<p class="result-count">{len(results):,} book{"s" if len(results) != 1 else ""} match your filters</p>',
        unsafe_allow_html=True,
    )

    if results.empty:
        st.info("No books match your current filters. Try relaxing some constraints.")
        return

    # ── Pagination logic ───────────────────────
    PAGE_SIZE = 20
    total_pages = max(1, -(-len(results) // PAGE_SIZE))
    current_page = st.session_state.page
    start = (current_page - 1) * PAGE_SIZE
    page_results = results.iloc[start : start + PAGE_SIZE]

    # ── Render grid ────────────────────────────
    render_book_grid(page_results)

    # ── Navigation bar ─────────────────────────
    nav_html = f"""
    <div style="
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
        padding: 1rem 0 0.5rem;
        font-family: sans-serif;
    ">
        <span style="font-size: 0.85rem; color: #888;">
            Page {current_page} of {total_pages}
            &nbsp;·&nbsp;
            {start + 1}–{min(start + PAGE_SIZE, len(results))} of {len(results):,} books
        </span>
    </div>
    """
    components.html(nav_html, height=50)

    # ── Prev / Next buttons ────────────────────
    col_prev, col_spacer, col_next = st.columns([1, 6, 1])

    with col_prev:
        if st.button("← Prev", disabled=(current_page <= 1), use_container_width=True):
            st.session_state.page -= 1
            st.rerun()

    with col_next:
        if st.button("Next →", disabled=(current_page >= total_pages), use_container_width=True):
            st.session_state.page += 1
            st.rerun()


if __name__ == "__main__":
    main()
