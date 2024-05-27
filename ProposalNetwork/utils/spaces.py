import numpy as np
import torch
from cubercnn import util

'''
coordinate system is assumed to have origin in the upper left
(0,0) _________________(N,0)
|  
|    
| 
|
|
(0,M)
'''
"""
class Cube:
    '''
    3D box in the format [c1, c2, c3, w, h, l, R]

    Args:
        c1: The x coordinate of the center of the box.
        c2: The y coordinate of the center of the box.
        c3: The z coordinate of the center of the box.
        w: The width of the box in meters.
        h: The height of the box in meters.
        l: The length of the box in meters.
        R: The 3D rotation matrix of the box.
    ```

                      _____________________ 
                    /|                    /|
                   / |                   / |
                  /  |                  /  |
                 /___|_________________/   |
                |    |                 |   | h
                |    |                 |   |
                |    |                 |   |
                |    |   (c1,c2,c3)    |   |
                |    |_________________|___|
                |   /                  |   /
                |  /                   |  /
                | /                    | / l
                |/_____________________|/
                            w             
    ```
    '''
    def __init__(self,tensor: torch.Tensor, R: torch.Tensor, score=None, label=None) -> None:
        self.tensor = tensor
        self.center = tensor[:3]
        self.dimensions = tensor[3:6]
        self.rotation = R

        # score and label are meant as auxiliary information
        self.score = score
        self.label = label

    def get_cube(self):
        color = [c/255.0 for c in util.get_color()]
        return util.mesh_cuboid(torch.cat((self.center,self.dimensions)), self.rotation, color=color)
    
    def get_all_corners(self):
        '''wrap ``util.get_cuboid_verts_faces``
        
        Returns:
            verts: the 3D vertices of the cuboid in camera space'''
        verts, _ = util.get_cuboid_verts_faces(torch.cat((self.center,self.dimensions)), self.rotation)
        return verts
    
    def get_bube_corners(self,K) -> torch.Tensor:
        cube_corners = self.get_all_corners()
        cube_corners = torch.mm(K, cube_corners.t()).t()
        return cube_corners[:,:2]/cube_corners[:,2].unsqueeze(1)
    
    def get_volume(self) -> float:
        return self.dimensions.prod().item()

    
    def __repr__(self) -> str:
        return f'Cube({self.center}, {self.dimensions}, {self.rotation})'
    
    def to_device(self, device):
        '''
        Move all tensors of the instantiated class to the specified device.

        Args:
            device: The device to move the tensors to (e.g., 'cuda', 'cpu').
        '''
        self.tensor = self.tensor.to(device)
        self.center = self.center.to(device)
        self.dimensions = self.dimensions.to(device)
        self.rotation = self.rotation.to(device)
        return self
"""

class Cubes:
    '''
    3D boxes in the format [[c1, c2, c3, w, h, l, R]]

    inspired by `detectron2.structures.Boxes`

    Args:
        tensor: torch.tensor(
            c1: The x coordinates of the center of the boxes.
            c2: The y coordinates of the center of the boxes.
            c3: The z coordinates of the center of the boxes.
            w: The width of the boxes in meters.
            h: The height of the boxes in meters.
            l: The length of the boxes in meters.
            R: The flattened 3D rotation matrix of the boxes (i.e. the rows are next to each other).
            )
            of shape (N, 15).
    ```

                      _____________________ 
                    /|                    /|
                   / |                   / |
                  /  |                  /  |
                 /___|_________________/   |
                |    |                 |   | h
                |    |                 |   |
                |    |                 |   |
                |    |   (c1,c2,c3)    |   |
                |    |_________________|___|
                |   /                  |   /
                |  /                   |  /
                | /                    | / l
                |/_____________________|/
                            w             
    ```
    '''
    def __init__(self,tensor: torch.Tensor, scores=None, labels=None) -> None:

        # score and label are meant as auxiliary information
        if scores is not None:
            assert scores.ndim == 2, f"scores.shape must be (n_instances, n_proposals), but was {scores.shape}" 
        self.scores = scores
        self.labels = labels

        if not isinstance(tensor, torch.Tensor):
            if not isinstance(tensor, np.ndarray):
                tensor = np.asarray(tensor)
            tensor = torch.as_tensor(tensor, dtype=torch.float32, device=torch.device("cpu"))
        else:
            tensor = tensor.to(torch.float32)
        if tensor.numel() == 0:
            tensor = tensor.reshape((-1, 15)).to(dtype=torch.float32)
        self.tensor = tensor
        if self.tensor.dim() == 1:
            self.tensor = self.tensor.unsqueeze(0)
        if self.tensor.dim() == 2:
            self.tensor = self.tensor.unsqueeze(0)

    @property
    def centers(self):
        return self.tensor[:, :, :3]
    
    @property
    def dimensions(self):
        return self.tensor[:, :, 3:6]
    
    @property
    def rotations(self):
        shape = self.tensor.shape
        return self.tensor[:, :, 6:].reshape(shape[0],shape[1], 3, 3)
    
    @property
    def device(self):
        return self.tensor.device
    
    @property
    def num_instances(self):
        return self.tensor.shape[0]

    def clone(self) -> "Cubes":
        """
        Clone the Cubes.

        Returns:
            Cubes
        """
        return Cubes(self.tensor.clone())
    

    def get_cubes(self):
        color = [c/255.0 for c in util.get_color()]
        return util.mesh_cuboid(torch.cat((self.centers.squeeze(0),self.dimensions.squeeze(0)),dim=1), self.rotations.squeeze(0), color=color)
        
    
    def get_all_corners(self):
        '''wrap ``util.get_cuboid_verts_faces``
        
        Returns:
            verts: the 3D vertices of the cuboid in camera space'''

        verts_list = []
        for i in range(self.num_instances):
            verts_next_instance, _ = util.get_cuboid_verts_faces(self.tensor[i, :, :6], self.rotations[i])
            verts_list.append(verts_next_instance)
        verts = torch.stack(verts_list, dim=0)

        return verts
    
    def get_cuboids_verts_faces(self):
        '''wrap ``util.get_cuboid_verts_faces``
        
        Returns:
            verts: the 3D vertices of the cuboid in camera space
            faces: the faces of the cuboid in camera space'''

        verts_list = []
        faces_list = []
        for i in range(self.num_instances):
            verts_next_instance, faces = util.get_cuboid_verts_faces(self.tensor[i, :, :6], self.rotations[i])
            verts_list.append(verts_next_instance)
            faces_list.append(faces)
        verts = torch.stack(verts_list, dim=0)
        faces = torch.stack(faces_list, dim=0)

        return verts, faces
    
    def get_bube_corners(self, K, clamp:tuple) -> torch.Tensor:
        '''This assumes that all the cubes have the same camera intrinsic matrix K

        clamp is a typically the image shape (width, height) to truncate the boxes to image frame, this avoids huge projected boxes
        Returns:
            num_instances x N x 8 x 2'''
        cube_corners = self.get_all_corners() # num_instances x N x 8 x 3
        num_prop = cube_corners.shape[1]
        cube_corners = cube_corners.reshape(self.num_instances * num_prop, 8, 3)
        K_repeated = K.repeat(self.num_instances * num_prop,1,1)
        cube_corners = torch.matmul(K_repeated, cube_corners.transpose(2,1))
        cube_corners = cube_corners[:, :2, :]/cube_corners[:, 2, :].unsqueeze(-2)
        cube_corners = cube_corners.transpose(2,1)
        cube_corners = cube_corners.reshape(self.num_instances,num_prop, 8, 2)

        # we must clamp and then stack, otherwise the gradient is fucked
        if clamp is not None:
            x = torch.clamp(cube_corners[..., 0], 0, clamp[0]-1)
            y = torch.clamp(cube_corners[..., 1], 0, clamp[1]-1)
        cube_corners = torch.stack((x, y), dim=-1)

        return cube_corners # num_instances x num_proposals x 8 x 2
    
    def get_volumes(self) -> float:
        return self.get_dimensions().prod(1).item()
    
    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        return f'Cubes({self.tensor})'
    
    def to(self, device: torch.device):
        # Cubes are assumed float32 and does not support to(dtype)
        if isinstance(self.scores, torch.Tensor):
            self.scores = self.scores.to(device=device)
        if isinstance(self.labels, torch.Tensor):
            self.labels = self.labels.to(device=device)
        return Cubes(self.tensor.to(device=device), self.scores, self.labels)
    
    def __getitem__(self, item) -> "Cubes":
        """
        Args:
            item: int, slice, or a BoolTensor

        Returns:
            Cubes: Create a new :class:`Cubes` by indexing.

        The following usage are allowed:

        1. `new_cubes = cubes[3]`: return a `Cubes` which contains only one box.
        2. `new_cubes = cubes[2:10]`: return a slice of cubes.
        3. `new_cubes = cubes[vector]`, where vector is a torch.BoolTensor
           with `length = len(cubes)`. Nonzero elements in the vector will be selected.

        Note that the returned Cubes might share storage with this Cubes,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            prev_n_prop = self.tensor.shape[1]
            return Cubes(self.tensor[item].view(1, prev_n_prop, -1))
        elif isinstance(item, tuple):
            return Cubes(self.tensor[item[0],item[1]].view(1, 1, -1))
        b = self.tensor[item]
        assert b.dim() == 2, "Indexing on Cubes with {} failed to return a matrix!".format(item)
        return Cubes(b)
    

    @classmethod
    def cat(cls, cubes_list: list["Cubes"]) -> "Cubes":
        """
        Concatenates a list of Cubes into a single Cubes

        Arguments:
            cubes_list (list[Cubes])

        Returns:
            Cubes: the concatenated Cubes
        """
        assert isinstance(cubes_list, (list, tuple))
        if len(cubes_list) == 0:
            return cls(torch.empty(0))
        assert all([isinstance(box, Cubes) for box in cubes_list])

        # use torch.cat (v.s. layers.cat) so the returned cubes never share storage with input
        cat_cubes = cls(torch.cat([b.tensor for b in cubes_list], dim=0))
        return cat_cubes
    
    @torch.jit.unused
    def __iter__(self):
        """
        Yield a cube as a Tensor of shape (15,) at a time.
        """
        yield from self.tensor