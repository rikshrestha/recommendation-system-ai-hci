# 🎬 IMDb Content-Based Movie Recommender System

This project implements a **content-based movie recommendation system** using the [IMDb 5000 Movie Dataset](https://www.kaggle.com/datasets/carolzhangdc/imdb-5000-movie-dataset).  
It was developed as part of the *AI in Human–Computer Interaction* course to demonstrate how intelligent systems can enhance user interaction through explainable and user-centric design.

---

## 📚 Overview

The system uses **TF–IDF vectorization** and **cosine similarity** to recommend movies similar to a selected title.  
Unlike collaborative filtering, this model relies only on movie metadata (genres, plot keywords, cast, and director) — making it ideal for cold-start scenarios and transparent recommendations.

A **Streamlit web app** provides an interactive interface for exploring recommendations with explanations.

---

## 🧠 Features

- **Content-Based Filtering:** Suggests movies based on metadata similarity.  
- **Explainable Results:** Each recommendation includes human-readable reasons (e.g., “Shares genres: Action, Sci-Fi”).  
- **Streamlit UI:** Clean, intuitive interface designed using human-centered principles.  
- **Search & Filters:** Search movie titles, control number of results, filter by IMDb score.  
- **Cached Performance:** Uses Streamlit caching to avoid recomputing TF–IDF on reruns.

---

## 🏗️ Project Structure

```
recommendation-system-ai-hci/
│
├── data/
│   └── movie_metadata.csv              # IMDb dataset (download from Kaggle)
│
├── recommenders/
│   ├── __init__.py
│   ├── base.py                         # Abstract recommender base class
│   └── content_based.py                # IMDb content-based recommender implementation
│
├── app.py                              # Streamlit web interface
├── requirements.txt                    # Dependencies
├── test_recommender.py                 # Console tester (optional)
├── .gitignore
└── README.md
```

---

## 🧩 Dataset

**Source:** [IMDb 5000 Movie Dataset (Kaggle)](https://www.kaggle.com/datasets/carolzhangdc/imdb-5000-movie-dataset)  
**File Used:** `movie_metadata.csv`

**Important Columns:**
- `movie_title` – Movie title  
- `genres` – Pipe-separated genres (e.g., Action|Adventure|Sci-Fi)  
- `plot_keywords` – Key themes or story concepts  
- `director_name` – Director of the movie  
- `actor_1_name`, `actor_2_name`, `actor_3_name` – Main cast  
- `language`, `country`, `content_rating` – Metadata  
- `duration`, `title_year`, `imdb_score` – Descriptive numeric fields

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/recommendation-system-ai-hci.git
cd recommendation-system-ai-hci
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv recommendation-system-venv
# Windows
recommendation-system-venv\Scripts\activate
# macOS/Linux
source recommendation-system-venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the Dataset
Download `movie_metadata.csv` from Kaggle and place it inside the `data/` folder.

---

## 🚀 Run the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

Then open your browser at [http://localhost:8501](http://localhost:8501).

---

## 🧮 How It Works

1. Loads `movie_metadata.csv`  
2. Creates text features combining:  
   ```
   genres + plot_keywords + director_name + actor_1_name + actor_2_name + actor_3_name
   ```
3. Vectorizes text with **TF–IDF** (Term Frequency–Inverse Document Frequency)  
4. Calculates **cosine similarity** between all movies  
5. Returns top-N similar movies for any selected title  
6. Displays results in an explainable format with metadata and genre-based reasoning

---

## 🧠 Example

**Input:** “Avatar”  
**Output:**
```
1. John Carter — Shares genres: Action, Adventure, Sci-Fi  
2. Pirates of the Caribbean: At World’s End — Shares genres: Action, Adventure, Fantasy  
3. The Dark Knight Rises — Shares genres: Action, Thriller  
4. Spectre — Similar themes/keywords based on description  
```

---

## 🧪 Quick Backend Test

Run a console test (optional):

```bash
python test_recommender.py
```

---

## 🧩 Configuration

Modify key parameters in `app.py` to adjust app behavior:

```python
SETTINGS = {
    "DATA_PATH": "data/movie_metadata.csv",
    "DEFAULT_TOP_N": 5,
    "MAX_RECOMMENDATIONS": 20,
    "PAGE_TITLE": "IMDb Content-Based Movie Recommender",
    "LAYOUT": "wide",
    "MIN_IMDB_DEFAULT": 0.0,
}
```

---

## 🧭 Future Improvements

- Add collaborative or hybrid filtering  
- Integrate with TMDb or OMDb APIs for posters and more metadata  
- Add user preference learning  
- Include sentiment analysis for richer contextual recommendations

---

## 🧾 References

- Kaggle: *IMDb 5000 Movie Dataset*  
  https://www.kaggle.com/datasets/carolzhangdc/imdb-5000-movie-dataset  
- Scikit-learn documentation: TF–IDF and cosine similarity  
- Streamlit documentation: [https://docs.streamlit.io](https://docs.streamlit.io)  
- University of the Cumberlands – *AI in Human–Computer Interaction*

---

© 2025 — Developed by [Your Name]  
*Graduate Project – AI in Human–Computer Interaction, University of the Cumberlands*
