import streamlit as st
import pandas as pd
import html
import base64
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
# DATA LOADING  — cached so it only runs once
# ─────────────────────────────────────────────
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Genres: strip whitespace from each genre value
    df['genres'] = df['genres'].fillna('')

    # Pages: coerce to numeric, drop unresolvable
    df['num_pages'] = pd.to_numeric(df['num_pages'], errors='coerce')

    # Publication year: coerce to int where possible
    df['original_publication_year'] = pd.to_numeric(
        df['original_publication_year'], errors='coerce'
    )

    # Ratings
    df['avg_rating'] = pd.to_numeric(df['avg_rating'], errors='coerce')
    df['ratings_count'] = pd.to_numeric(df['ratings_count'], errors='coerce')

    return df


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
    rating = f"★ {row['avg_rating']:.2f}" if pd.notna(row.get('avg_rating')) else ''

    return f"""
    <div class="book-card">
        <div class="cover-wrap">
            {cover_html}
            <div class="overlay">
                <p class="ov-title">{title}</p>
                <p class="ov-author">by {author}</p>
                <p class="ov-rating">{rating}</p>
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
    </style>
    <div class="book-grid">{cards_html}</div>
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
    """Render all filter controls and return selected values as a dict."""

    with st.sidebar:
        st.markdown("## 📖 Next Read")
        st.markdown("*Discover your next favourite book*")
        st.divider()

        all_genres = get_all_genres(df)
        selected_genres = st.multiselect(
            "Genres",
            options=all_genres,
            placeholder="Select all genres",
        )

        st.markdown("---")

        valid_pages = df['num_pages'].dropna()
        page_min = int(valid_pages.min()) if not valid_pages.empty else 1
        page_max = int(valid_pages.max()) if not valid_pages.empty else 2000
        page_range = st.slider(
            "Page Range",
            min_value=page_min,
            max_value=page_max,
            value=(page_min, page_max),
        )

        st.markdown("---")

        valid_years = df['original_publication_year'].dropna()
        year_min = int(valid_years.min()) if not valid_years.empty else 0
        year_max = int(valid_years.max()) if not valid_years.empty else 2025
        year_range = st.slider(
            "Publication Year",
            min_value=year_min,
            max_value=year_max,
            value=(year_min, year_max),
        )

        st.markdown("---")

        valid_ratings = df['avg_rating'].dropna()
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

    return {
        "selected_genres": selected_genres,
        "page_range": page_range,
        "year_range": year_range,
        "min_rating": min_rating,
        "min_ratings_count": min_ratings_count,
        "sort_by": sort_by,
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # ── Load data ──────────────────────────────
    DATA_PATH = "data/thrillers.csv"
    df = load_data(DATA_PATH)

    # ── Pagination state ───────────────────────
    if "page" not in st.session_state:
        st.session_state.page = 1

    if "last_filters" not in st.session_state:
        st.session_state.last_filters = ""

    # ── Sidebar filters ────────────────────────
    filters = render_sidebar(df)

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
