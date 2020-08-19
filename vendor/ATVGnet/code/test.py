import os
import glob
from collections import OrderedDict
import argparse
import numpy as np
import torch
import librosa
import python_speech_features
from tqdm import tqdm

from models import AT_net


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--at_model", type=str, default="vendor/ATVGnet/model/atnet_lstm_18.pth")
    parser.add_argument('-i','--in_file', type=str, default=None)

    return parser.parse_args()

config = parse_args()

def multi2single(model_path, id):
    checkpoint = torch.load(model_path)
    state_dict = checkpoint
    if id ==1:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        return new_state_dict
    else:
        return state_dict

def test():
    encoder = AT_net().cuda()
    state_dict = multi2single(config.at_model, 1)
    encoder.load_state_dict(state_dict)
    encoder.eval()

    input_audio = glob.glob(os.path.join(config.in_file, 'audio/*'))[0]

    speech, sr = librosa.load(input_audio, sr=16000, mono=True)
    speech = np.insert(speech, 0, np.zeros(1920))
    speech = np.append(speech, np.zeros(1920))
    mfcc = python_speech_features.mfcc(speech,16000,winstep=0.01)

    ind = 3

    with torch.no_grad():
        input_mfcc = []
        while ind <= int(mfcc.shape[0]/4) - 4:
            t_mfcc =mfcc[( ind - 3) * 4: (ind + 4) * 4, 1:]
            t_mfcc = torch.FloatTensor(t_mfcc).cuda()
            input_mfcc.append(t_mfcc)
            ind += 1

        input_mfcc = torch.stack(input_mfcc,dim = 0)
        input_mfcc = input_mfcc.unsqueeze(0)
        features = encoder(input_mfcc)

        features_lengh = (len(features) // 25) * 25

        for i in tqdm(range(features_lengh)):
            torch.save(features[i][0].detach().cpu(), os.path.join(config.in_file, 'feature/%05d.pt' % (i+1)))

        print('feature length:',features_lengh)

test()
