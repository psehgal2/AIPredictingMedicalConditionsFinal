NUM_IMAGES = 50

import pandas as pd
import sys, os
from PIL import Image, ImageStat
import util

print(os.listdir())
df = util.get_image_df()
df = df.drop('Unnamed: 0.1', axis=1)
df = df.drop('Unnamed: 0', axis=1)
df['NoFinding'] = df['No Finding']
df.drop('No Finding',axis=1)
print(os.listdir())
print()
os.chdir('/')
print(os.listdir())
print()
for idx, row in df.iterrows():
    if idx > NUM_IMAGES:
        break
    path = util.DATA_DIRECTORY + row['Path']
    with Image.open(path) as im:
        im.save(util.TEAM_DIRECTORY + f'/visualizations/images/{idx}.png')
