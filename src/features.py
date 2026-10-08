import numpy as np
import pandas as pd
from typing import Tuple, Optional

FD = "Flight Distance"
CATS = ["Gender", "Customer Type", "Type of Travel", "Class"]
NUMS = ["Age", FD, "Departure Delay in Minutes", "Arrival Delay in Minutes"]
RATINGS = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service",
    "Baggage handling", "Checkin service", "Cleanliness"
]
CORE_FEATS = NUMS + RATINGS + CATS

def build_static_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    orig_df: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Constructs the verified 63 static sovereign features across train and test sets."""
    ntr = len(train_df)
    full_df = pd.concat([train_df.drop(columns=["satisfaction", "id"], errors="ignore"),
                         test_df.drop(columns=["id"], errors="ignore")],
                        ignore_index=True)

    # 1. Missing Value Imputation
    full_df["Arrival Delay in Minutes"] = full_df["Arrival Delay in Minutes"].fillna(
        full_df["Departure Delay in Minutes"]
    ).fillna(0.0)

    # 2. Delay Dynamics & Survey Engagement
    full_df["Delay_Total"] = (full_df["Departure Delay in Minutes"] + full_df["Arrival Delay in Minutes"]).astype(np.float32)
    full_df["Delay_Diff"] = (full_df["Arrival Delay in Minutes"] - full_df["Departure Delay in Minutes"]).astype(np.float32)
    full_df["Delay_Ratio"] = ((full_df["Arrival Delay in Minutes"] + 1.0) / (full_df["Departure Delay in Minutes"] + 1.0)).astype(np.float32)
    full_df["Delay_Per_Mile"] = (full_df["Delay_Total"] / (full_df[FD] + 1.0)).astype(np.float32)
    full_df["Zero_Ratings_Count"] = (full_df[RATINGS] == 0).sum(axis=1).astype(np.float32)

    # 3. Context & Service Interaction Features
    full_df["Digital_Service_Index"] = ((full_df["Inflight wifi service"] + 1.0) * (full_df["Online boarding"] + 1.0)).astype(np.float32)
    full_df["Digital_Satisfaction_Gap"] = (full_df["Inflight wifi service"] - full_df["Online boarding"]).astype(np.float32)
    full_df["Physical_Service_Index"] = ((full_df["Seat comfort"] + full_df["Cleanliness"] + full_df["Food and drink"]) / 3.0).astype(np.float32)
    full_df["Cabin_Comfort_Cluster"] = ((full_df["Seat comfort"] + full_df["Leg room service"] + full_df["Cleanliness"]) / 3.0).astype(np.float32)
    
    class_map = {"Eco": 0, "Eco Plus": 1, "Business": 2}
    class_numeric = full_df["Class"].map(class_map).fillna(0).astype(np.float32)
    full_df["Service_Expectation_Gap"] = (full_df["Physical_Service_Index"] - (class_numeric * 1.5)).astype(np.float32)
    full_df["Wifi_Boarding_Ratio"] = ((full_df["Inflight wifi service"] + 0.5) / (full_df["Online boarding"] + 0.5)).astype(np.float32)
    
    loyalty_num = (full_df["Customer Type"] == "Loyal Customer").astype(np.float32)
    biz_num = (full_df["Type of Travel"] == "Business travel").astype(np.float32)
    full_df["Loyal_Business_Segment"] = (loyalty_num * 2.0 + biz_num).astype(np.float32)

    # 4. Frequency Encodings
    cnt_feats = pd.DataFrame(index=full_df.index)
    for col in NUMS:
        cnt_feats[f"cnt_{col}"] = full_df[col].map(train_df[col].value_counts()).fillna(0).astype(np.float32)

    # 5. Route Profile Means Grouped by Flight Distance
    coded_full = full_df[CORE_FEATS].copy()
    for c in CATS:
        coded_full[c] = coded_full[c].astype("category").cat.codes
    route_group = coded_full.groupby(FD)
    route_profiles = pd.DataFrame(index=full_df.index)
    for c in CORE_FEATS:
        if c != FD:
            route_profiles[f"route_mean_{c}"] = route_group[c].transform("mean").astype(np.float32)
    route_profiles["route_size"] = route_group["Age"].transform("size").astype(np.float32)

    # 6. External Priors (or fallback)
    orig_priors = pd.DataFrame(index=full_df.index)
    orig_priors["orig_route_satisfaction_prior"] = 0.5
    orig_priors["og_xgb_prob"] = 0.5
    orig_priors["og_xgb_logit"] = 0.0
    orig_priors["og_lgb_prob"] = 0.5
    orig_priors["og_lgb_logit"] = 0.0

    core_encoded = full_df[CORE_FEATS].copy()
    for c in CATS:
        core_encoded[c] = core_encoded[c].astype("category")

    context_cols = [
        "Delay_Total", "Delay_Diff", "Delay_Ratio", "Delay_Per_Mile",
        "Zero_Ratings_Count", "Digital_Service_Index", "Digital_Satisfaction_Gap",
        "Physical_Service_Index", "Cabin_Comfort_Cluster", "Service_Expectation_Gap",
        "Wifi_Boarding_Ratio", "Loyal_Business_Segment"
    ]

    static_x = pd.concat([
        core_encoded,
        full_df[context_cols],
        cnt_feats,
        route_profiles,
        orig_priors
    ], axis=1)

    x_train = static_x.iloc[:ntr].copy().reset_index(drop=True)
    x_test = static_x.iloc[ntr:].copy().reset_index(drop=True)
    return x_train, x_test
