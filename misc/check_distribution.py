import pandas as pd

eng = pd.read_csv("eot_data/eot_data/english/labels.csv")
hin = pd.read_csv("eot_data/eot_data/hindi/labels.csv")

print(eng.head())
print(eng["label"].value_counts())

print(hin["label"].value_counts())