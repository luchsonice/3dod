# Basically a hotfix script to avoid having to run the ground segemntation script again
# this will filter out empty ground maps and add the indices to the no_ground_idx.csv file
import os
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd

files = os.listdir('datasets/ground_maps')
no_ground = []
for file in tqdm(files):
    mask = np.load(f'datasets/ground_maps/{file}')['mask']
    ground_map = torch.as_tensor(mask).unsqueeze(0)
    nnz = torch.count_nonzero(ground_map, dim=(-2, -1))
    indices = torch.nonzero(nnz <= 1000).flatten()
    if len(indices) > 0:
        print('indices', file[:-4])
        no_ground.append(int(file[:-4]))
        os.remove(f'datasets/ground_maps/{file}')

df = pd.DataFrame(no_ground, columns=['img_id'])
df2 = pd.read_csv('datasets/no_ground_idx.csv')
df = pd.concat([df, df2])
df.to_csv('datasets/no_ground_idx.csv', index=False)