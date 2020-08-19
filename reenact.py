import os
from tqdm import tqdm
from torchvision import utils

from renderer.face_model import FaceModel
from options.options import Options
from utils.util import create_dir, load_coef


if __name__ == '__main__':
    opt = Options().parse_args()

    create_dir(os.path.join(opt.src_dir, 'reenact'))

    alpha_list = load_coef(os.path.join(opt.tgt_dir, 'alpha'), opt.test_num)
    beta_list = load_coef(os.path.join(opt.tgt_dir, 'beta'), opt.test_num)
    delta_list = load_coef(os.path.join(opt.src_dir, 'reenact_delta'), opt.test_num)
    gamma_list = load_coef(os.path.join(opt.tgt_dir, 'gamma'), opt.test_num)
    angle_list = load_coef(os.path.join(opt.tgt_dir, 'rotation'), opt.test_num)
    translation_list = load_coef(os.path.join(opt.tgt_dir, 'translation'), opt.test_num)

    face_model = FaceModel(data_path=opt.matlab_data_path, batch_size=1)

    for i in tqdm(range(len(delta_list))):
        alpha = alpha_list[i + opt.offset].unsqueeze(0).cuda()
        beta = beta_list[i + opt.offset].unsqueeze(0).cuda()
        delta = delta_list[i].unsqueeze(0).cuda()
        gamma = gamma_list[i + opt.offset].unsqueeze(0).cuda()
        rotation = angle_list[i + opt.offset].unsqueeze(0).cuda()
        translation = translation_list[i + opt.offset].unsqueeze(0).cuda()

        render, _, _ = face_model(alpha, delta, beta, rotation, translation, gamma, lower=True)
        utils.save_image(render, os.path.join(opt.src_dir, 'reenact', '%05d.png' % (i+1)))

        if i >= opt.test_num:
            break
