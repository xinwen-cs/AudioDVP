import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def _apply(layer, activation, normalizer, channel_out=None):
    if normalizer:
        layer.append(normalizer(channel_out))
    if activation:
        layer.append(activation())
    return layer


def conv2d(channel_in, channel_out,
           ksize=3, stride=1, padding=1,
           activation=nn.ReLU,
           normalizer=nn.BatchNorm2d):
    layer = list()
    bias = True if not normalizer else False

    layer.append(nn.Conv2d(channel_in, channel_out,
                     ksize, stride, padding,
                     bias=bias))
    _apply(layer, activation, normalizer, channel_out)

    return nn.Sequential(*layer)


class AT_net(nn.Module):
    def __init__(self):
        super(AT_net, self).__init__()
        self.lmark_encoder = nn.Sequential(
            nn.Linear(6,256),
            nn.ReLU(True),
            nn.Linear(256,512),
            nn.ReLU(True),
        )

        self.audio_eocder = nn.Sequential(
            conv2d(1,64,3,1,1),
            conv2d(64,128,3,1,1),
            nn.MaxPool2d(3, stride=(1,2)),
            conv2d(128,256,3,1,1),
            conv2d(256,256,3,1,1),
            conv2d(256,512,3,1,1),
            nn.MaxPool2d(3, stride=(2,2))
        )

        self.audio_eocder_fc = nn.Sequential(
            nn.Linear(1024 *12,2048),
            nn.ReLU(True),
            nn.Linear(2048,256),
            nn.ReLU(True),
        )

        self.lstm = nn.LSTM(256*3,256,3,batch_first = True)
        self.lstm_fc = nn.Sequential(
            nn.Linear(256,6),
        )

    def forward(self, audio):
        features = []
        for step_t in range(audio.size(1)):
            current_audio = audio[ : ,step_t , :, :].unsqueeze(1)
            current_feature = self.audio_eocder(current_audio) # [1, 512, 12, 2]
            current_feature = current_feature.view(current_feature.size(0), -1)
            current_feature = self.audio_eocder_fc(current_feature)
            features.append(current_feature)

        return features
