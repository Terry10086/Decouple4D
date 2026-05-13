import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
import torch.nn.functional as F


class CameraOptModule(torch.nn.Module):
    """Camera pose optimization module."""

    def __init__(self, n: int):
        super().__init__()
        # Delta positions (3D) + Delta rotations (6D)
        self.embeds = torch.nn.Embedding(n, 9)
        # Identity rotation in 6D representation
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def zero_init(self):
        torch.nn.init.zeros_(self.embeds.weight)

    def random_init(self, std: float):
        torch.nn.init.normal_(self.embeds.weight, std=std)

    def forward(self, camtoworlds: Tensor, embed_ids: Tensor) -> Tensor:
        """Adjust camera pose based on deltas.

        Args:
            camtoworlds: (..., 4, 4)
            embed_ids: (...,)

        Returns:
            updated camtoworlds: (..., 4, 4)
        """

        # assert camtoworlds.shape[:-2] == embed_ids.shape

        
        pose_deltas = self.embeds(embed_ids)  # (..., 9)
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]
        rot = rotation_6d_to_matrix(
            drot + self.identity
        )  # (..., 3, 3)
        transform = torch.eye(4, device=pose_deltas.device)
        transform[:3, :3] = rot
        transform[:3, 3] = dx
        return torch.matmul(camtoworlds, transform)
    




# def get_positional_encodings(
#     height: int, width: int, num_frequencies: int, device: str = "cuda"
# ) -> torch.Tensor:
#     """Generates positional encodings for a given image size and frequency range.

#     Args:
#       height: height of the image
#       width: width of the image
#       num_frequencies: number of frequencies
#       device: device to use

#     Returns:

#     """
#     # Generate grid of (x, y) coordinates
#     y, x = torch.meshgrid(

#         torch.arange(height, device=device),
#         torch.arange(width, device=device),
#         indexing="ij",
#     )

#     # Normalize coordinates to the range [0, 1]
#     y = y / (height - 1)
#     x = x / (width - 1)

#     # Create frequency range [1, 2, 4, ..., 2^(num_frequencies-1)]
#     frequencies = (
#         torch.pow(2, torch.arange(num_frequencies, device=device)).float() * torch.pi
#     )

#     # Compute sine and cosine of the frequencies multiplied by the coordinates
#     y_encodings = torch.cat(
#         [torch.sin(frequencies * y[..., None]), torch.cos(frequencies * y[..., None])],
#         dim=-1,
#     )
#     x_encodings = torch.cat(
#         [torch.sin(frequencies * x[..., None]), torch.cos(frequencies * x[..., None])],
#         dim=-1,
#     )

#     # Combine the encodings
#     pos_encodings = torch.cat([y_encodings, x_encodings], dim=-1)

#     return pos_encodings


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


# def normalized_quat_to_rotmat(quat: Tensor) -> Tensor:
#     assert quat.shape[-1] == 4, quat.shape
#     w, x, y, z = torch.unbind(quat, dim=-1)
#     mat = torch.stack(
#         [
#             1 - 2 * (y**2 + z**2),
#             2 * (x * y - w * z),
#             2 * (x * z + w * y),
#             2 * (x * y + w * z),
#             1 - 2 * (x**2 + z**2),
#             2 * (y * z - w * x),
#             2 * (x * z - w * y),
#             2 * (y * z + w * x),
#             1 - 2 * (x**2 + y**2),
#         ],
#         dim=-1,
#     )
#     return mat.reshape(quat.shape[:-1] + (3, 3))


# def knn(x: Tensor, K: int = 4) -> Tensor:
#     x_np = x.cpu().numpy()
#     model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
#     distances, _ = model.kneighbors(x_np)
#     return torch.from_numpy(distances).to(x)


# def rgb_to_sh(rgb: Tensor) -> Tensor:
#     C0 = 0.28209479177387814
#     return (rgb - 0.5) / C0


# def set_random_seed(seed: int):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
