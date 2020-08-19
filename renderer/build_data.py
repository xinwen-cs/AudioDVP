"""
Following https://github.com/microsoft/Deep3DFaceReconstruction/blob/master/load_data.py to extract 3DMM data
"""

import numpy as np
import scipy.io as sio


def load_exp_base():
    num_vertex = 53215

    with open('data/Exp_Pca.bin', 'rb') as exp_bin:
        exp_dim = np.fromfile(exp_bin, dtype=np.int32, count=1)  # [79]
        _ = np.fromfile(exp_bin, dtype=np.float32, count=3 * num_vertex)  # expMU not used

        exp_pc = np.fromfile(exp_bin, dtype=np.float32, count=3 * exp_dim[0] * num_vertex)
        exp_pc = np.reshape(exp_pc, [exp_dim[0], -1])
        exp_pc = np.transpose(exp_pc)  # 53215*3, 79

    exp_ev = np.loadtxt('data/std_exp.txt')  # 79,

    return exp_pc, exp_ev


def load_trim_index():
    """
    remove ear neck and inner mouth
    """
    exp_index = sio.loadmat('data/BFM_front_idx.mat')['idx'].astype(np.int32) - 1  # 35709
    geo_index = sio.loadmat('data/BFM_exp_idx.mat')['trimIndex'].astype(np.int32) - 1  # 53215
    geo_index = geo_index[exp_index]

    return exp_index, geo_index[:, :, 0]


def load_bfm():
    mat_data = sio.loadmat('data/01_MorphableModel.mat')

    geo_mu = mat_data['shapeMU']
    geo_pc = mat_data['shapePC']
    geo_ev = mat_data['shapeEV']

    tex_mu = mat_data['texMU']
    tex_pc = mat_data['texPC']
    tex_ev = mat_data['texEV']

    return geo_mu, geo_pc, geo_ev, tex_mu, tex_pc, tex_ev


def load_triangles():
    other_info = sio.loadmat('data/facemodel_info.mat')
    triangles = other_info['tri'].astype(np.int32) - 1

    key_points = other_info['keypoints'][0].astype(np.int64) - 1
    point_buf = other_info['point_buf'].astype(np.int32) - 1  # adjacent face index for each vertex
    point_buf = point_buf.astype(np.int64)

    return triangles, key_points, point_buf


def get_mouth_triangles(geo_mean, triangles):
    geo = geo_mean.copy().reshape(-1, 3)
    geo = geo - geo.mean(axis=0, keepdims=True)
    geo = geo / geo.max()

    mouth_vertex_list = []

    threshold = 0.0

    for i in range(geo.shape[0]):  # 35709
        if geo[i, 1] < threshold:  # y coordinate
            mouth_vertex_list.append(i)

    mouth_triangle_list = []

    for tri in triangles:
        for ver in tri:
            if ver.item() in mouth_vertex_list:
                mouth_triangle_list.append(np.array(tri))
                break

    mouth_triangles = np.stack(mouth_triangle_list, axis=0)

    return mouth_triangles


if __name__ == '__main__':
    geo_mu, geo_pc, geo_ev, tex_mu, tex_pc, tex_ev = load_bfm()

    exp_pc, exp_ev = load_exp_base()

    index_exp, index_geo = load_trim_index()

    triangles, key_points, point_buf = load_triangles()

    id_base = geo_pc * np.reshape(geo_ev, [-1, 199])
    id_base = id_base[:, :80]  # use only first 80 basis
    id_base = np.reshape(id_base, [-1, 3, 80])
    id_base = id_base[index_geo, :, :]
    id_base = np.reshape(id_base, [-1, 80]) / 92156.7

    exp_base = exp_pc * np.reshape(exp_ev, [-1, 79])
    exp_base = exp_base[:, :64]  # use only first 64 basis
    exp_base = np.reshape(exp_base, [-1, 3, 64])
    exp_base = exp_base[index_exp, :, :]
    exp_base = np.reshape(exp_base, [-1, 64]).astype(np.float32) / 92156.7

    tex_base = tex_pc * np.reshape(tex_ev, [-1, 199])
    tex_base = tex_base[:, :80]  # use only first 80 basis
    tex_base = np.reshape(tex_base, [-1, 3, 80])
    tex_base = tex_base[index_geo, :, :]
    tex_base = np.reshape(tex_base, [-1, 80]) / 255.0

    geo_mean = np.reshape(geo_mu, [-1, 3])
    geo_mean = geo_mean[index_geo, :]
    geo_mean = np.reshape(geo_mean, [-1, 1]) / 92156.7

    tex_mean = np.reshape(tex_mu, [-1, 3])
    tex_mean = tex_mean[index_geo, :]
    tex_mean = np.reshape(tex_mean, [-1, 1]) / 255.0

    mouth_triangles = get_mouth_triangles(geo_mean, triangles)

    sio.savemat('data/data.mat', {
                        'triangles': triangles, 'key_points': key_points, 'point_buf': point_buf, 'mouth_triangles': mouth_triangles,
                        'geo_mean': geo_mean, 'tex_mean': tex_mean,
                        'id_base': id_base, 'exp_base': exp_base, 'tex_base': tex_base
                    }
                )
