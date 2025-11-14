# 🎬 IMDb Content-Based Movie Recommender (AI/HCI Project)

**Course:** AI in Human–Computer Interaction  
**Institution:** University of the Cumberlands  
**Author:** Rikrish Shrestha  
**Instructor:** Professor Jennifer Merritt  
**Date:** November 2025

---

## 📘 Overview

This project implements a **content-based movie recommendation system** using the IMDb `movie_metadata.csv` dataset.  
The system demonstrates how machine learning and human–computer interaction (HCI) principles can be combined to create a user-friendly recommendation experience.

The recommender analyzes movie **genres**, **plot keywords**, **director names**, and **main cast members** to find similar movies using **TF–IDF vectorization** and **cosine similarity**.

The application includes a graphical interface built with **Streamlit**, allowing interactive filtering and dynamic exploration of movie relationships.

---

## 🧠 Features

- Content-based filtering  
- TF–IDF + cosine similarity  
- Streamlit front-end  
- IMDb score filtering  
- Logging system  
- Cached model for speed  

---

## 🗂️ Project Structure

```
recommendation-system-ai-hci/
├── app.py
├── README.md
├── requirements.txt
├── test_recommender.py
├── data/
│   └── movie_metadata.csv
├── recommenders/
│   ├── __init__.py
│   ├── base.py
│   └── content_based.py
└── logs/
    ├── app.log
    └── content_based.log
```

---

## ⚙️ Setup and Usage

### 1. Create and activate virtual environment

**Windows**
```
python -m venv recommendation-system-venv
recommendation-system-venv\Scripts\activate
```

**macOS/Linux**
```
python3 -m venv recommendation-system-venv
source recommendation-system-venv/bin/activate
```

---

### 2. Install dependencies

```
pip install -r requirements.txt
```

---

### 3. Run Streamlit UI

```
streamlit run app.py
```

---

### 4. Optional: Test script

```
python test_recommender.py
```

---

## 🧾 Technologies Used

- Python  
- Streamlit  
- Pandas  
- Scikit-learn  
- NumPy  

---

## 📈 Future Improvements

- Hybrid recommender  
- User personalization  
- Clustering visualizations  

---

## 🧾 References

- scikit-learn: https://scikit-learn.org  
- Streamlit: https://streamlit.io  
- Kaggle IMDb dataset  

---

© 2025 University of the Cumberlands
