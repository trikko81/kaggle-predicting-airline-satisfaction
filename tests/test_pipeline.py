import unittest
import numpy as np
import pandas as pd
from src.folds import generate_stratified_folds
from src.features import build_static_features
from src.blend import rank_average, optimize_blend_weights

class TestPipelineModules(unittest.TestCase):
    def setUp(self):
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
        folds = generate_stratified_folds(self.dummy_train, n_splits=5, seed=42)
        self.assertEqual(len(folds), len(self.dummy_train))
        self.assertEqual(len(np.unique(folds)), 5)

        X_train, X_test = build_static_features(self.dummy_train, self.dummy_test)
        self.assertEqual(len(X_train), len(self.dummy_train))
        self.assertEqual(len(X_test), len(self.dummy_test))
        self.assertEqual(X_train.shape[1], 63)

    def test_blend_functions(self):
        y_true = np.random.randint(0, 2, 200)
        p1 = np.clip(y_true * 0.8 + np.random.normal(0, 0.2, 200), 0, 1)
        p2 = np.clip(y_true * 0.7 + np.random.normal(0, 0.3, 200), 0, 1)

        rank_blend = rank_average([p1, p2])
        self.assertEqual(len(rank_blend), 200)
        self.assertTrue((rank_blend >= 0).all() and (rank_blend <= 1).all())

        weights, best_auc = optimize_blend_weights(y_true, {'m1': p1, 'm2': p2})
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=4)
        self.assertGreater(best_auc, 0.5)

if __name__ == '__main__':
    unittest.main()
