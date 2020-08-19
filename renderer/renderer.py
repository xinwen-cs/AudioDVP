import numpy as np

import torch
import torch.nn as nn

from .rasterizer import Rasterizer


class Renderer(nn.Module):
    def __init__(self, image_width=256, image_height=256):
        super(Renderer, self).__init__()

        self.device = torch.device('cuda')
        self.image_width = image_width
        self.image_height = image_height

        self.rasterizer = Rasterizer()
        self.shader = SphericalHarmonics()

    def forward(self, clip_vertices, triangles, normals, diffuse_colors, gamma):
        diffuse_colors_sh = self.shader(normals, diffuse_colors, gamma)
        render_image, alpha_mask = self.rasterizer(clip_vertices, diffuse_colors_sh, triangles)

        return render_image, alpha_mask


class SphericalHarmonics(nn.Module):
    """Compute vertex color using face texture and Spherical Harmonics lighting approximation

    Following https://github.com/microsoft/Deep3DFaceReconstruction/blob/80066b54507e98652e7efb2d4213146632380a05/reconstruct_mesh.py

    Args:
        face_texture: a [1, N, 3] tensor
        norm: a [1, N, 3] tensor
        gamma: a [1, 27] tensor

    Returns:
        face_color: a [1,N,3] tensor, RGB order, range from 0-255
    """
    def __init__(self):
        super(SphericalHarmonics, self).__init__()
        self.device = torch.device('cuda')
        self.init_light = torch.tensor([[[0.7000, 0.7000, 0.7000],
                                         [0.0000, 0.0000, 0.0000],
                                         [0.0000, 0.0000, 0.0000],
                                         [0.0000, 0.0000, 0.0000],
                                         [0.0000, 0.0000, 0.0000],
                                         [0.0000, 0.0000, 0.0000],
                                         [0.0000, 0.0000, 0.0000],
                                         [0.0000, 0.0000, 0.0000],
                                         [0.0000, 0.0000, 0.0000]]], device=self.device)

        self.a0 = np.pi
        self.a1 = 2 * np.pi / np.sqrt(3.0)
        self.a2 = 2 * np.pi / np.sqrt(8.0)
        self.c0 = 1 / np.sqrt(4 * np.pi)
        self.c1 = np.sqrt(3.0) / np.sqrt(4 * np.pi)
        self.c2 = 3 * np.sqrt(5.0) / np.sqrt(12 * np.pi)

    def forward(self, normals, diffuse_colors, gamma):
        batch_size, num_vertex, _ = diffuse_colors.shape

        gamma = gamma.reshape(-1, 9, 3) + self.init_light

        nx, ny, nz = normals[:, :, 0], normals[:, :, 1], normals[:, :, 2]

        Y0 = torch.ones(batch_size, num_vertex, device=self.device) * self.a0 * self.c0
        Y1 = -self.a1 * self.c1 * ny
        Y2 = self.a1 * self.c1 * nz
        Y3 = -self.a1 * self.c1 * nx
        Y4 = self.a2 * self.c2 * nx * ny
        Y5 = -self.a2 * self.c2 * ny * nz
        Y6 = self.a2 * self.c2 * 0.5 / np.sqrt(3.0) * (3 * nz.pow(2)-1)
        Y7 = -self.a2 * self.c2 * nx * nz
        Y8 = self.a2 * self.c2 * 0.5 * (nx.pow(2) - ny.pow(2))

        Y = torch.stack([Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8], 2)

        lighting = Y.bmm(gamma)
        face_color = diffuse_colors * lighting

        return face_color
