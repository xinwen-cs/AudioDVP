"""
Follwing https://github.com/google/tf_mesh_renderer/blob/master/mesh_renderer/camera_utils.py
"""

import math
import torch
import torch.nn as nn


class Camera(nn.Module):
    def __init__(self):

        super(Camera, self).__init__()

        self.device = torch.device('cuda')

        self.camera_position = torch.tensor([[0.0, 0.0, 8.0]], device=self.device)
        self.camera_lookat = torch.tensor([[0.0, 0.0, 0.0]], device=self.device)
        self.camera_up = torch.tensor([[0.0, -1.0, 0.0]], device=self.device)

        self.fov_y = 25.0
        self.near_clip = 0.01
        self.far_clip = 100.0

        self.camera_matrices = look_at(self.camera_position, self.camera_lookat, self.camera_up)
        self.perspective_transforms = perspective(
            torch.tensor([1.0], device=self.device),
            torch.tensor([self.fov_y], device=self.device),
            torch.tensor([self.near_clip], device=self.device),
            torch.tensor([self.far_clip], device=self.device)
        )
        self.clip_space_transforms = torch.matmul(self.perspective_transforms, self.camera_matrices)

    def forward(self, vertices):
        clip_vertices = transform_homogeneous(self.clip_space_transforms, vertices)

        return clip_vertices


def look_at(eye, center, world_up):
    """Compute camera viewing matrices.

    Functionality mimes gluLookAt (external/GL/glu/include/GLU/glu.h).

    Args:
        eye: 2D float32 tensor with shape [batch_size, 3] containing the XYZ
            world space position of the camera.
        center: 2D float32 tensor with shape [batch_size, 3] containing a
            position along the center of the camera's gaze line.
        world_up: 2D float32 tensor with shape [batch_size, 3] specifying the
            world's up direction; the output camera will have no tilt with
            respect to this direction.

    Returns:
        A [batch_size, 4, 4] float tensor containing a right-handed camera
        extrinsics matrix that maps points from world space to points in eye
        space.
    """
    batch_size = center.shape[0]
    forward = center - eye

    forward = torch.nn.functional.normalize(forward, dim=1, p=2)

    to_side = torch.cross(forward, world_up)
    to_side = torch.nn.functional.normalize(to_side, dim=1, p=2)

    cam_up = torch.cross(to_side, forward)

    w_column = torch.tensor(batch_size * [[0., 0., 0., 1.]], device=eye.device)
    w_column = torch.reshape(w_column, [batch_size, 4, 1])

    view_rotation = torch.stack([to_side, cam_up, -forward, torch.zeros_like(to_side)], dim=1)  # [batch_size, 4, 3] matrix
    view_rotation = torch.cat([view_rotation, w_column], dim=2)  # [batch_size, 4, 4]

    identity_batch = torch.unsqueeze(torch.eye(3, device=center.device), 0,).repeat([batch_size, 1, 1])
    view_translation = torch.cat([identity_batch, torch.unsqueeze(-eye, 2)], 2)
    view_translation = torch.cat([view_translation, torch.reshape(w_column, [batch_size, 1, 4])], 1)
    camera_matrices = torch.matmul(view_rotation, view_translation)

    return camera_matrices


def perspective(aspect_ratio, fov_y, near_clip, far_clip):
    """Computes perspective transformation matrices.

    Functionality mimes gluPerspective (external/GL/glu/include/GLU/glu.h).
    See:
    https://unspecified.wordpress.com/2012/06/21/calculating-the-gluperspective-matrix-and-other-opengl-matrix-maths/

    Args:
        aspect_ratio: float value specifying the image aspect ratio
            (width/height).
        fov_y: 1D float32 Tensor with shape [batch_size] specifying output
            vertical field of views in degrees.
        near_clip: 1D float32 Tensor with shape [batch_size] specifying near
            clipping plane distance.
        far_clip: 1D float32 Tensor with shape [batch_size] specifying far
            clipping plane distance.

    Returns:
        A [batch_size, 4, 4] float tensor that maps from right-handed points in
        eye space to left-handed points in clip space.
    """
    # The multiplication of fov_y by pi/360.0 simultaneously converts to radians
    # and adds the half-angle factor of .5.
    focal_lengths_y = 1.0 / torch.tan(fov_y * (math.pi / 360.0))
    depth_range = far_clip - near_clip
    p_22 = -(far_clip + near_clip) / depth_range
    p_23 = -2.0 * (far_clip * near_clip / depth_range)

    zeros = torch.zeros_like(p_23, dtype=torch.float32)
    perspective_transform = torch.cat(
        [
            focal_lengths_y / aspect_ratio, zeros, zeros, zeros,
            zeros, focal_lengths_y, zeros, zeros,
            zeros, zeros, p_22, p_23,
            zeros, zeros, -torch.ones_like(p_23, dtype=torch.float32), zeros
        ], dim=0)

    perspective_transform = torch.reshape(perspective_transform, [4, 4, -1])
    return perspective_transform.permute(2, 0, 1)


def transform_homogeneous(matrices, vertices):
    """Applies batched 4x4 homogeneous matrix transforms to 3D vertices.

    The vertices are input and output as row-major, but are interpreted as
    column vectors multiplied on the right-hand side of the matrices. More
    explicitly, this function computes (MV^T)^T.
    Vertices are assumed to be xyz, and are extended to xyzw with w=1.

    Args:
        matrices: a [batch_size, 4, 4] tensor of matrices.
        vertices: a [batch_size, N, 3] tensor of xyz vertices.

    Returns:
        a [batch_size, N , 4] tensor of xyzw vertices.
    """
    homogeneous_coord = torch.ones([vertices.shape[0], vertices.shape[1], 1], device=vertices.device)
    vertices_homogeneous = torch.cat([vertices, homogeneous_coord], 2)

    return torch.matmul(vertices_homogeneous, matrices.permute(0, 2, 1))
