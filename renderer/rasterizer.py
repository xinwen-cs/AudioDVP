"""
Following https://github.com/google/tf_mesh_renderer/blob/master/mesh_renderer/rasterize_triangles.py
"""

import torch
import torch.nn as nn

from .kernels import rasterize_triangles_cpp


class RasterizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vertices, triangles, image_width, image_height):
        px_triangle_ids = torch.zeros(image_height, image_width, dtype=torch.int32, device=vertices.device)
        px_barycentric_coords = torch.zeros(image_height, image_width, 3, dtype=torch.float32, device=vertices.device, requires_grad=True)
        z_buffer = torch.ones(image_height, image_width, dtype=torch.float32, device=vertices.device)

        px_triangle_ids, px_barycentric_coords, z_buffer \
            = rasterize_triangles_cpp.forward(vertices, triangles, image_width, image_height,px_triangle_ids, px_barycentric_coords, z_buffer)

        ctx.save_for_backward(vertices, triangles, px_triangle_ids, px_barycentric_coords, torch.tensor(image_width), torch.tensor(image_height))

        return px_triangle_ids, px_barycentric_coords, z_buffer

    @staticmethod
    def backward(ctx, df_dpx_triangle_ids, df_dpx_barycentric_coords, df_dz_buffer):
        del df_dpx_triangle_ids, df_dz_buffer

        vertices, triangles, px_triangle_ids, px_barycentric_coords, image_width, image_height = ctx.saved_tensors

        df_dvertices = torch.zeros(vertices.shape[0], 4, dtype=torch.float32, device=vertices.device)

        df_dvertices = rasterize_triangles_cpp.backward(
            vertices, triangles, px_triangle_ids, px_barycentric_coords, df_dpx_barycentric_coords, image_width, image_height, df_dvertices)

        return df_dvertices, None, None, None


class Rasterizer(nn.Module):
    def __init__(self, image_width=256, image_height=256):
        super(Rasterizer, self).__init__()

        self.image_width = image_width
        self.image_height = image_height

    def forward(self, clip_vertices, diffuse_colors, triangles):
        batch_size, num_vertex, _ = clip_vertices.shape

        per_image_barycentric_coordinates = []
        per_image_vertex_ids = []

        for b in range(batch_size):
            px_triangle_ids, px_barycentric_coords, _ = RasterizeFunction.apply(clip_vertices[b, :, :], triangles, self.image_width, self.image_height)

            per_image_barycentric_coordinates.append(torch.reshape(px_barycentric_coords, [-1, 3]))
            vertex_ids = triangles[px_triangle_ids.long()]
            reindexed_ids = vertex_ids + b * clip_vertices.shape[1]
            per_image_vertex_ids.append(reindexed_ids)

        barycentric_coordinates = torch.reshape(torch.stack(per_image_barycentric_coordinates, 0), [-1, 3, 1])
        vertex_ids = torch.reshape(torch.stack(per_image_vertex_ids, 0), [-1, 3])

        # Indexes with each pixel's clip-space triangle's extrema ids to get the relevant properties for deferred shading.
        flattened_vertex_diffuse_colors = torch.reshape(diffuse_colors, [batch_size * num_vertex, -1])
        corner_diffuse_colors = flattened_vertex_diffuse_colors[vertex_ids.long()]

        weighted_vertex_diffuse_colors = torch.mul(corner_diffuse_colors, barycentric_coordinates)

        summed_diffuse_colors = torch.clamp(torch.sum(weighted_vertex_diffuse_colors, dim=1), 0, 1)
        images = torch.reshape(summed_diffuse_colors, [batch_size, self.image_height, self.image_width, 3]).permute(0, 3, 1, 2)

        # Barycentric coordinates should approximately sum to one where there is rendered geometry, but be exactly zero where there is not.
        alphas = torch.clamp(torch.sum(2.0 * barycentric_coordinates, dim=1), 0.0, 1.0)
        alphas = torch.reshape(alphas, [batch_size, 1, self.image_height, self.image_width])

        return images, alphas
