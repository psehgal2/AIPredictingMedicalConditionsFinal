import numpy as np
import pandas as pd
import csv

pred = []
with open('predictions.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        pred.append(float(row[0]))

df = pd.DataFrame()

df['Pneumonia'] = pred

df.to_csv('../data/pneuomia_predictions')
