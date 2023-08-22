import pandas as pd
import os
import tqdm

PATH = '/groups/CS156b/data/student_labels/train2023.csv'

df = pd.read_csv(PATH)

for idx, row in tqdm.tqdm(df.iterrows(), total=len(df)):
    if not os.path.exists('/groups/CS156b/data/' +  row['Path']):
        print('/groups/CS156b/data/' +  row['Path'] + "doesn't exist")