import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio

from .camera import Camera
from .renderer import Renderer


class FaceModel(nn.Module):
    def __init__(self, data_path, batch_size, image_width=256, image_height=256):
        super(FaceModel, self).__init__()
        self.mat_data = sio.loadmat(data_path)
        self.batch_size = batch_size
        self.device = torch.device('cuda')

        self.image_width = image_width
        self.image_height = image_height

        self.load_data()

        self.camera = Camera()
        self.renderer = Renderer()

    def load_data(self):
        self.triangles = torch.from_numpy(self.mat_data['triangles']).to(self.device)
        self.triangles64 = torch.from_numpy(self.mat_data['triangles']).long().to(self.device)
        self.mouth_triangles = torch.from_numpy(self.mat_data['mouth_triangles']).to(self.device)
        self.point_buf = torch.from_numpy(self.mat_data['point_buf']).to(self.device)  # adjacent vertex

        self.geo_mean = torch.from_numpy(self.mat_data['geo_mean']).unsqueeze(0).expand(self.batch_size, -1, -1).to(self.device)
        self.tex_mean = torch.from_numpy(self.mat_data['tex_mean']).unsqueeze(0).expand(self.batch_size, -1, -1).to(self.device)
        self.id_base = torch.from_numpy(self.mat_data['id_base']).unsqueeze(0).to(self.device)
        self.exp_base = torch.from_numpy(self.mat_data['exp_base']).unsqueeze(0).expand(self.batch_size, -1, -1).to(self.device)
        self.tex_base = torch.from_numpy(self.mat_data['tex_base']).unsqueeze(0).to(self.device)

        self.landmark_index = torch.tensor([
            27440, 27208, 27608, 27816, 35472, 34766, 34312, 34022, 33838, 33654,
            33375, 32939, 32244, 16264, 16467, 16888, 16644, 31716, 31056, 30662,
            30454, 30288, 29549, 29382, 29177, 28787, 28111,  8161,  8177,  8187,
            8192,  9883,  9163,  8204,  7243,  6515, 14066, 12383, 11353, 10455,
            11492, 12653,  5828,  4920,  3886,  2215,  3640,  4801, 10795, 10395,
            8935,  8215,  7495,  6025,  5522,  6915,  7636,  8236,  8836,  9555,
            10537,  9064,  8223,  7384,  5909,  7629,  8229,  8829
        ], device=self.device)

    def build_face_model(self, alpha, delta, beta):
        tex = self.tex_mean + self.tex_base.bmm(beta)
        tex = tex.reshape(self.batch_size, -1, 3)

        geo = self.geo_mean + self.id_base.bmm(alpha) + self.exp_base.bmm(delta)
        geo = geo.reshape(self.batch_size, -1, 3)

        return geo, tex

    def calc_norm(self, geo):
        v1 = geo[:, self.triangles64[:, 0], :]
        v2 = geo[:, self.triangles64[:, 1], :]
        v3 = geo[:, self.triangles64[:, 2], :]

        e1 = v1 - v2
        e2 = v2 - v3

        face_norm = e1.cross(e2)  # compute normal for each face
        empty = torch.zeros(self.batch_size, 1, 3, device=self.device)
        face_norm = torch.cat((face_norm, empty), 1)  # concat face_normal with a zero vector at the end

        vertex_norm = face_norm[:, self.point_buf, :].sum(2)  # compute vertex normal using one-ring neighborhood
        vertex_norm = F.normalize(vertex_norm, dim=2)

        return vertex_norm

    def transform_to_world_space(self, geo, norm, rotation, translation):
        model2world = euler_matrices(rotation).permute(0, 2, 1)  # R^(-1)
        geo = torch.matmul(geo - translation.permute(0, 2, 1), model2world)  # R^(-1)(V-t)
        norm = torch.matmul(norm, model2world)

        return geo, norm

    def transform_to_clip_space(self, geo):
        clip_vertices = self.camera(geo)

        return clip_vertices

    def transform_to_screen_space(self, clip_vertices):
        screen_vertices = ((clip_vertices[:, :, :2] / clip_vertices[:, :, 3:]) + 1) * self.image_height * 0.5

        return screen_vertices

    def forward(self, alpha, delta, beta, rotation, translation, gamma, lower=False):
        geo, tex = self.build_face_model(alpha, delta, beta)
        norm = self.calc_norm(geo)
        geo, norm = self.transform_to_world_space(geo, norm, rotation, translation)
        clip_vertices = self.transform_to_clip_space(geo)
        screen_vertices = self.transform_to_screen_space(clip_vertices)

        landmarks = screen_vertices[:, self.landmark_index]

        if not lower:
            render_image, alpha_mask = self.renderer(clip_vertices, self.triangles, norm, tex, gamma)
        else:
            render_image, alpha_mask = self.renderer(clip_vertices, self.mouth_triangles, norm, tex, gamma)

        return render_image, alpha_mask, landmarks


def euler_matrices(angles):
    """Compute a XYZ Tait-Bryan (improper Euler angle) rotation.

    Follwing https://github.com/google/tf_mesh_renderer/blob/master/mesh_renderer/camera_utils.py

    Return 3x3 matrices for convenient multiplication with other transformations.
        following tf_mesh_renderer
    Args:
      angles: a [batch_size, 3] tensor containing X, Y, and Z angles in radians.

    Returns:
      a [batch_size, 3, 3] tensor of matrices.
    """
    s = torch.sin(angles)
    c = torch.cos(angles)

    c0, c1, c2 = (c[:, 0], c[:, 1], c[:, 2])
    s0, s1, s2 = (s[:, 0], s[:, 1], s[:, 2])

    flattened = torch.cat([
            c2*c1, c2*s1*s0 - c0*s2, s2*s0 + c2*c0*s1,
            c1*s2, c2*c0 + s2*s1*s0, c0*s2*s1 - c2*s0,
            -s1, c1*s0, c1*c0,
    ])

    return torch.reshape(flattened, [3, 3, -1]).permute(2, 0, 1)


if __name__ == '__main__':
    from torchvision import utils

    alpha = torch.zeros(1, 80, 1).cuda()
    delta = torch.zeros(1, 64, 1).cuda()
    beta = torch.zeros(1, 80, 1).cuda()
    gamma = torch.zeros(1, 27, 1).cuda()
    angle = torch.zeros(1, 3, 1).cuda()
    translation = torch.zeros(1, 3, 1).cuda()

    face_model = FaceModel(data_path='data/data.mat', batch_size=1)

    render_image, _, _ = face_model(alpha, delta, beta, angle, translation, gamma)
    utils.save_image(render_image, 'render.png')
