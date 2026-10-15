import pandas as pd

# Put the name of the 85k row file you downloaded here
large_file = "data/raw/Tong_hop_thu_tuc.csv"

# Read it and slice exactly 500 rows
df = pd.read_csv(large_file)
small_df = df.head(500)

# Save it as a new file for your test
small_df.to_csv("data/raw/small_test_data.csv", index=False)
print("Done! Created small_test_data.csv with 500 rows.")
