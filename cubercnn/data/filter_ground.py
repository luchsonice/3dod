# Basically a hotfix script to avoid having to run the ground segemntation script again
# this will filter out empty ground maps and add the indices to the no_ground_idx.csv file
# It removes ground maps with very little ground, because we assume that it has found something wrong
import os
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd

files = os.listdir('datasets/ground_maps')
no_ground = []
for file in tqdm(files):
    mask = np.load(f'datasets/ground_maps/{file}')['mask']
    ground_map = torch.as_tensor(mask)[::5,::5]
    nnz = torch.count_nonzero(ground_map).item()
    # 100 is determined from looking at the pictures
    if nnz < 100:
        print(nnz)
        print('indices', file[:-4])
        no_ground.append(int(file[:-4]))
        os.remove(f'datasets/ground_maps/{file}')

df = pd.DataFrame(no_ground, columns=['img_id'])
df2 = pd.read_csv('datasets/no_ground_idx.csv')
df = pd.concat([df, df2])
df.to_csv('datasets/no_ground_idx.csv', index=False)