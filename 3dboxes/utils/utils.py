import torch

def meters_to_pixels(meters: torch.Tensor, focal_length: torch.Tensor, image_width: torch.Tensor, image_height: torch.Tensor) -> torch.Tensor:
    '''
    Converts meters to pixels.

    Args:
        meters: A tensor of shape (N, 2) where N is the number of boxes and the last dimension is the x and y coordinates in meters.
        focal_length: The focal length of the camera.
        image_width: The width of the image.
        image_height: The height of the image.

    Returns:
        A tensor of shape (N, 2) where N is the number of boxes and the last dimension is the x and y coordinates in pixels.
    '''
    return (meters / meters[:, 0].unsqueeze(1)) * focal_length.unsqueeze(0) + torch.tensor([image_width, image_height]).unsqueeze(0) / 2

def 3D_rotation_to_2D_rotation(rotation: torch.Tensor, focal_length: torch.Tensor) -> torch.Tensor:
    '''
    Converts a 3D rotation to a 2D rotation.

    Args:
        rotation: A tensor of shape (N, 3) where N is the number of boxes and the last dimension is the 3D rotation.
        focal_length: The focal length of the camera.

    Returns:
        A tensor of shape (N, 2, 2) where N is the number of boxes and the last two dimensions are the 2D rotation.
    '''
    return rotation[:, :2] * focal_length.unsqueeze(1)