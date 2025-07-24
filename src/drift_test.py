import pandas as pd
from data_drift import check_data_drift

ref_df = pd.read_csv("data/reference_dataset.csv")
cur_df = pd.read_csv("data/current_dataset.csv")

# Call the function
if check_data_drift(ref_df, cur_df):
    print("Drift detected — trigger alert or retraining")
else:
    print("No drift — model looks healthy")
