# kaggle-airline-satisfaction

Machine learning project for the Kaggle tabular competition: **Predicting Airline Passenger Satisfaction**.

## Project Structure

```text
├── data/              # Raw and processed datasets (gitignored)
│   ├── raw/
│   └── processed/
├── notebooks/         # Exploratory data analysis and experimental notebooks
├── src/               # Reusable Python source code
│   ├── __init__.py
│   ├── data.py        # Data loading and preprocessing pipelines
│   ├── features.py    # Feature engineering routines
│   ├── models.py      # Model training, hyperparameter tuning, and inference
│   └── utils.py       # Metrics, logging, and evaluation utilities
├── submissions/       # Generated Kaggle submission CSV files
├── requirements.txt   # Core Python dependencies
├── .gitignore
└── README.md
```

## Setup & Environment

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
