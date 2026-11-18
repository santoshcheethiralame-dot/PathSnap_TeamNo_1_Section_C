# PathSnap – GPS Map Matching & Route Reconstruction

**Team No. 1 | Section C | PES University**

PathSnap is a machine learning-based system that reconstructs realistic vehicle routes from noisy GPS observations using road-network information, machine learning, and Hidden Markov Models.

## Project Overview

GPS signals can be inaccurate due to buildings, signal interference and location errors. PathSnap identifies the most likely road segments travelled by a vehicle and reconstructs its route.

### How It Works

1. **GPS Processing:** Processes GPS coordinates, speed, timestamps and heading.
2. **Candidate Generation:** Identifies approximately 11 nearby road candidates.
3. **Feature Extraction:** Extracts 26 features for each candidate.
4. **Machine Learning:** Uses HistGradientBoosting to predict the correct road.
5. **Route Reconstruction:** Uses HMM and Viterbi to generate a connected route.
6. **Visualisation:** Displays the reconstructed route on an interactive map.

## Machine Learning Models

We experimented with five models:

- Logistic Regression
- Gaussian Naive Bayes
- Random Forest
- HistGradientBoosting
- MLP (64, 32)

HistGradientBoosting was selected based on accuracy, generalisation and training efficiency. Optuna was used for hyperparameter optimisation with 40 trials.

## Performance Results

| Method | Top-1 Accuracy |
|---|---:|
| Rule-Based Baseline | 86.0% |
| HistGradientBoosting | 90.6% |
| Classifier + HMM | **93.3%** |

Additional results:
- Route Continuity: 90.8%
- Route Overlap: 95.8%
- All 10 student-testing trips improved with HMM.

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| ML | Scikit-learn |
| Optimisation | Optuna |
| Backend | FastAPI |
| Frontend | HTML, CSS, JavaScript |
| Maps | Folium, Leaflet |
| Testing | Pytest |

## Installation

```bash
git clone https://github.com/santoshcheethiralame-dot/PathSnap_TeamNo_1_Section_C.git
cd PathSnap_TeamNo_1_Section_C

python -m venv .venv
```

Activate the environment and install dependencies:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run the Application

Start the backend:

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Start the frontend in another terminal:

```bash
python -m http.server 8080 --directory ui
```

Open `http://localhost:8080` in your browser and connect to the backend.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check server status |
| GET | `/trips` | List available trips |
| POST | `/predict` | Reconstruct a selected trip |

API documentation: `http://localhost:8000/docs`

## Repository Structure

```text
PathSnap/
├── api/          # FastAPI backend
├── data/         # GPS datasets
├── docs/         # Project documentation
├── models/       # Trained ML models
├── report/       # Final report and figures
├── results/      # Experimental results
├── scripts/      # Training and evaluation scripts
├── src/pathsnap/ # Core ML pipeline
├── tests/        # Unit tests
├── ui/           # Frontend application
└── README.md
```

## Limitations and Future Scope

- Improve road candidate generation.
- Handle stopped vehicles and inaccurate heading information.
- Improve route prediction on parallel roads.
- Test on larger and more diverse GPS datasets.

## Academic Integrity

AI assistance was used during development and documentation. The team is responsible for understanding, validating and explaining the implementation and experimental results.

---

**PathSnap – From Noisy GPS Points to Realistic Road Routes.**
