# 🎬 Mazag Movies - AI Recommendation System

A modern, full-stack movie recommendation system that combines **Rule-Based Filtering** (Node.js) with **AI-Powered Recommendations** (Python/KNN).

## 🚀 Features
- **Hybrid Search:** Filter by Name, Genre, Year, and Rating instantly.
- **AI Engine:** Uses K-Nearest Neighbors (KNN) to suggest similar movies based on features.
- **Modern UI:** Glassmorphism design with responsive grid layout.
- **Performance:** Optimized to handle large datasets without freezing.

## 🛠️ Tech Stack
- **Frontend:** HTML5, CSS3 (Glassmorphism), Vanilla JS.
- **Backend:** Node.js, Express.js.
- **AI/ML:** Python, Pandas, Scikit-learn (KNN Model).
- **Data:** JSON & CSV processing.

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/mazag-movies.git](https://github.com/YOUR_USERNAME/mazag-movies.git)
   cd mazag-movies
Setup Backend (Node.js):

Bash
cd backend
npm install
Setup AI Engine (Python): Install required libraries:

Bash
pip install pandas scikit-learn numpy
▶️ Usage
Start the Server:

Bash
cd backend
node server.js
Open Browser: Go to http://localhost:3000

📂 Project Structure
├── backend/         # Express Server & API
├── frontend/        # UI & Client Logic
├── inference/       # Python Scripts for ML
├── model/           # Trained .pkl Models
├── data/            # Movie Datasets
└── images/          # Movie Posters