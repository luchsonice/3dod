from cubercnn import util
from ProposalNetwork.utils.spaces import Cubes
import torch
from ProposalNetwork.utils.utils import iou_3d

center = torch.tensor([0,0,0])
dim = torch.tensor([1,1,1])
unit_rotation = util.euler2mat([0,0,0]).flatten()
grid_rotation = util.euler2mat([0,2.5,0]).flatten()

unit_cube = torch.cat([center,dim,torch.tensor(unit_rotation)])
grid_cube = torch.cat([center,dim,torch.tensor(grid_rotation)])
cubes = Cubes(torch.stack((unit_cube,grid_cube)).unsqueeze(1))

print('Difference in IoU due to rotation grid:',1-iou_3d(cubes[0],cubes[1]).item())