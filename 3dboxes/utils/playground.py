from spaces import Box, Bube, Cube
from conversions import bube_to_box

import torch







cube = Cube(torch.tensor([0,0,0,1,1,1,torch.eye(3).flatten()]))
print(cube.get_all_corners())