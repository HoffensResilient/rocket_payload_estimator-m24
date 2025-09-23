# Just to visualise what the data looks like.

import pandas as pd
import os


BASE_DIR = os.path.dirname(os.path.dirname(__file__))  
DATA_DIR = os.path.join(BASE_DIR, "data")

REAL_FILE = os.path.join(DATA_DIR, "Altitude_dataset.xlsx")

# loading the data
real_df = pd.read_excel(REAL_FILE)

print(real_df.shape)
print(real_df.head(15))
print("\nColumns:", real_df.columns.tolist())
