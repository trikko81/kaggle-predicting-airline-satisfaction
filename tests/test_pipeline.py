import unittest
import numpy as np
import pandas as pd
from pathlib import Path

class TestPipelineModules(unittest.TestCase):
    def setUp(self):
        # Create minimal synthetic dataframe matching competition schema
        np.random.seed(42)
        n = 100
        self.dummy_train = pd.DataFrame({
            'id': range(n),
            'Age': np.random.randint(18, 75, n),
            'Flight Distance': np.random.choice([100, 250, 500, 1000], n),
            'Inflight wifi service': np.random.randint(0, 6, n),
            'Departure/Arrival time convenient': np.random.randint(0, 6, n),
            'Ease of Online booking': np.random.randint(0, 6, n),
            'Gate location': np.random.randint(0, 6, n),
            'Food and drink': np.random.randint(0, 6, n),
            'Online boarding': np.random.randint(0, 6, n),
            'Seat comfort': np.random.randint(0, 6, n),
            'Inflight entertainment': np.random.randint(0, 6, n),
            'On-board service': np.random.randint(0, 6, n),
            'Leg room service': np.random.randint(0, 6, n),
            'Baggage handling': np.random.randint(1, 6, n),
            'Checkin service': np.random.randint(0, 6, n),
            'Cleanliness': np.random.randint(0, 6, n),
            'Departure Delay in Minutes': np.random.exponential(15, n),
            'Arrival Delay in Minutes': np.random.exponential(15, n),
            'Gender': np.random.choice(['Male', 'Female'], n),
            'Customer Type': np.random.choice(['Loyal Customer', 'disloyal Customer'], n),
            'Type of Travel': np.random.choice(['Personal Travel', 'Business travel'], n),
            'Class': np.random.choice(['Eco', 'Eco Plus', 'Business'], n),
            'satisfaction': np.random.choice(['satisfied', 'neutral or dissatisfied'], n)
        })
        self.dummy_test = self.dummy_train.drop(columns=['satisfaction']).copy()

    def test_folds_and_features(self):
        from src.folds import generate_stratified_folds
        from src.features import build_static_features

        folds = generate_stratified_folds(self.dummy_train, n_splits=5, seed=42)
        self.assertEqual(len(folds), len(self.dummy_train))
        self.assertEqual(len(np.unique(folds)), 5)

        X_train, X_test = build_static_features(self.dummy_train, self.dummy_test)
        self.assertEqual(len(X_train), len(self.dummy_train))
        self.assertEqual(len(X_test), len(self.dummy_test))
        # 63 static features (or 58 if clean prior is excluded without orig data)
        self.assertGreaterEqual(X_train.shape[1], 55)

if __name__ == '__main__':
    unittest.main()
