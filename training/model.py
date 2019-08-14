import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict


class SimpleCNN(nn.Module):
    """
    Baseline CNN for Judge
    """
    def __init__(self, input_size, num_class):
        super(SimpleCNN, self).__init__()
        self.convolution = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv2d(in_channels=input_size[0], out_channels=32, kernel_size=3)),
            ('bn1_1', nn.BatchNorm2d(num_features=32)),
            ('relu1_1', nn.ReLU()),
            ('conv1_2', nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)),
            ('bn1_2', nn.BatchNorm2d(num_features=32)),
            ('relu1_2', nn.ReLU()),
            ('maxpool1', nn.MaxPool2d(kernel_size=2)),

            ('conv2_1', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)),
            ('bn2_1', nn.BatchNorm2d(num_features=64)),
            ('relu2_1', nn.ReLU()),
            ('conv2_2', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)),
            ('bn2_2', nn.BatchNorm2d(num_features=64)),
            ('relu2_2', nn.ReLU()),
            ('maxpool2', nn.MaxPool2d(kernel_size=2))
        ]))

        feature_size = self.determine_feature_size(input_size)

        self.dense = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(in_features=feature_size, out_features=512)),
            ('bn_d', nn.BatchNorm1d(num_features=512)),
            ('relu_d', nn.ReLU()),
            ('dropout', nn.Dropout(p=0.2)),
            ('out', nn.Linear(512, num_class))
        ]))

    def determine_feature_size(self, input_size):
        fake_input = Variable(torch.randn(input_size)[None, :, :, :])
        fake_out = self.convolution(fake_input)
        return fake_out.view(-1).shape[0]

    def forward(self, x):
        feat = self.convolution(x)
        feat = feat.view(x.size(0), -1)
        return self.dense(feat)


class ConvEncoder(nn.Module):
    """
    A convolutional encoder, splitting apart encoder-decoder for shared encoder scheme
    """
    def __init__(self, input_size):
        super(ConvEncoder, self).__init__()
        self.convolution = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv2d(in_channels=input_size[0], out_channels=32, kernel_size=3)),
            ('bn1_1', nn.BatchNorm2d(num_features=32)),
            ('relu1_1', nn.ReLU()),
            ('conv1_2', nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)),
            ('bn1_2', nn.BatchNorm2d(num_features=32)),
            ('relu1_2', nn.ReLU()),
            ('maxpool1', nn.MaxPool2d(kernel_size=2)),

            ('conv2_1', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)),
            ('bn2_1', nn.BatchNorm2d(num_features=64)),
            ('relu2_1', nn.ReLU()),
            ('conv2_2', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)),
            ('bn2_2', nn.BatchNorm2d(num_features=64)),
            ('relu2_2', nn.ReLU()),
            ('maxpool2', nn.MaxPool2d(kernel_size=2))
        ]))

    def forward(self, x):
        return self.convolution(x)


class ConvDecoder(nn.Module):
    """
    Decoding counterpart to ConvEncoder
    """

    def __init__(self, input_size):
        super(ConvDecoder, self).__init__()
        self.deconvolution = nn.Sequential(OrderedDict([
            ('deconv1', nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1)),
            ('bn1', nn.BatchNorm2d(num_features=32)),
            ('relu1', nn.ReLU()),
            ('deconv2', nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2)),
            ('bn2', nn.BatchNorm2d(num_features=16)),
            ('relu2', nn.ReLU()),
            ('deconv3', nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=2, stride=2)),
            ('bn3', nn.BatchNorm2d(num_features=8)),
            ('relu3', nn.ReLU()),
            ('deconv4', nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=5, stride=1)),
            ('bn4', nn.BatchNorm2d(num_features=4)),
            ('relu4', nn.ReLU()),

            ('conv1', nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, padding=1)),
            ('conv_bn1', nn.BatchNorm2d(num_features=2)),
            ('conv2', nn.Conv2d(in_channels=2, out_channels=input_size[0], kernel_size=1))
        ]))

    def forward(self, x):
        return self.deconvolution(x)


class ConvDeconv(nn.Module):
    """
    Hyper-simple convolutional encoder-decoder to replace hourglass
    """
    def __init__(self, input_size, capacity=None):
        super(ConvDeconv, self).__init__()
        if capacity is None:
            self.convolution = ConvEncoder(input_size)
            self.deconvolution = ConvDecoder(input_size)
        else:
            self.convolution = VariableConvEncoder(input_size, capacity)
            self.deconvolution = VariableConvDecoder(input_size)

    def forward(self, x):
        compress = self.convolution(x)
        decompress = self.deconvolution(compress)
        return decompress


class VariableConvEncoder(nn.Module):
    """
    A convolutional encoder, splitting apart encoder-decoder for shared encoder scheme
    """
    def __init__(self, input_size, capacity, residual=True):
        super(VariableConvEncoder, self).__init__()
        self.residual = residual
        self.capacity = capacity
        self.blocks_1 = nn.ModuleList()
        for i in range(capacity[0]):
            if i == 0:
                inp_size = input_size[0]
            else:
                inp_size = 32
            self.blocks_1.append(nn.Sequential(OrderedDict([
                ('conv1_{}'.format(i), nn.Conv2d(in_channels=inp_size, out_channels=32, kernel_size=3, padding=1)),
                ('bn1_{}'.format(i), nn.BatchNorm2d(num_features=32)),
                ('relu1_{}'.format(i), nn.ReLU())])))
        self.blocks_2 = nn.ModuleList()
        for i in range(capacity[1]):
            if i == 0:
                inp_size = 32
            else:
                inp_size = 64
            self.blocks_2.append(nn.Sequential(OrderedDict([
                ('conv2_{}'.format(i), nn.Conv2d(in_channels=inp_size, out_channels=64, kernel_size=3, padding=1)),
                ('bn2_{}'.format(i), nn.BatchNorm2d(num_features=64)),
                ('relu2_{}'.format(i), nn.ReLU())])))

        # reduce from 7x7 to 4x4
        self.final_conv = nn.Sequential(OrderedDict([
            ('conv3_1', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2)),
            ('bn3_1', nn.BatchNorm2d(num_features=64)),
            ('relu3_1', nn.ReLU()),
            ('conv3_2', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2)),
            ('bn3_2', nn.BatchNorm2d(num_features=64)),
            ('relu3_2', nn.ReLU()),
            ('conv3_3', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2)),
            ('bn3_3', nn.BatchNorm2d(num_features=64)),
            ('relu3_3', nn.ReLU())
        ]))

    def forward(self, x):
        for i in range(self.capacity[0]):
            xp = self.blocks_1[i](x)
            if i == 0:
                x = xp
            else:
                x = xp + x
        x = F.max_pool2d(x, kernel_size=2)
        for i in range(self.capacity[1]):
            xp = self.blocks_2[i](x)
            if i == 0:
                x = xp
            else:
                x = xp + x
        x = F.max_pool2d(x, kernel_size=2)
        x = self.final_conv(x)
        return x


class VariableConvDecoder(nn.Module):
    """
    Decoding counterpart to ConvEncoder
    """
    def __init__(self, input_size):
        super(VariableConvDecoder, self).__init__()
        self.deconvolution = nn.Sequential(OrderedDict([
            ('deconv1', nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1)),
            ('bn1', nn.BatchNorm2d(num_features=32)),
            ('relu1', nn.ReLU()),
            ('deconv2', nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2)),
            ('bn2', nn.BatchNorm2d(num_features=16)),
            ('relu2', nn.ReLU()),
            ('deconv3', nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=2, stride=2)),
            ('bn3', nn.BatchNorm2d(num_features=8)),
            ('relu3', nn.ReLU()),
            ('deconv4', nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=5, stride=1)),
            ('bn4', nn.BatchNorm2d(num_features=4)),
            ('relu4', nn.ReLU()),

            ('conv1', nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, padding=1)),
            ('conv_bn1', nn.BatchNorm2d(num_features=2)),
            ('conv2', nn.Conv2d(in_channels=2, out_channels=input_size[0], kernel_size=1))
        ]))

    def forward(self, x):
        return self.deconvolution(x)


class VariableCNN(nn.Module):
    """
    Baseline CNN
    """
    def __init__(self, input_size, num_class, capacity):
        super(VariableCNN, self).__init__()
        self.convolution = VariableConvEncoder(input_size, capacity)

        feature_size = self.determine_feature_size(input_size)

        self.dense = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(in_features=feature_size, out_features=512)),
            ('bn_d', nn.BatchNorm1d(num_features=512)),
            ('relu_d', nn.ReLU()),
            ('dropout', nn.Dropout(p=0.2)),
            ('out', nn.Linear(512, num_class))
        ]))

    def determine_feature_size(self, input_size):
        fake_input = Variable(torch.randn(input_size)[None, :, :, :])
        fake_out = self.convolution(fake_input)
        return fake_out.view(-1).shape[0]

    def forward(self, x):
        feat = self.convolution(x)
        feat = feat.view(x.size(0), -1)
        return self.dense(feat)


class AdvocacyNet(nn.Module):
    """
    Serves as the model for each approach. By setting num_advocates=1 you get Attention Net, by setting
    num_advocates=num_class you get Multi-Attention Net, Advocate Net, or Honest Advocate Net.
    ts is used for MIMIC experiments
    """
    def __init__(self, input_size, num_class, num_advocates, shared_encoder=False,
                 ts=False, advocate_capacity=None, judge_capacity=None):
        super(AdvocacyNet, self).__init__()
        self.ts = ts
        if self.ts:
            assert advocate_capacity is None
            assert judge_capacity is None
            encoder_decoder = TSConvDeconv
            encoder = TSConvEncoder
            decoder = TSConvDecoder
            judge = MIMICNet
        else:
            encoder_decoder = ConvDeconv
            if advocate_capacity is None:
                encoder = ConvEncoder
                decoder = ConvDecoder
            else:
                encoder = VariableConvEncoder
                decoder = VariableConvDecoder
            if judge_capacity is None:
                judge = SimpleCNN
            else:
                judge = VariableCNN
        self.num_class = num_class
        # note, not setting to num_class to allow for attention net
        self.num_advocates = num_advocates
        self.shared_encoder = shared_encoder
        self.advocates = torch.nn.ModuleList()
        if self.shared_encoder:
            if advocate_capacity is None:
                self.encoder = encoder(input_size)
            else:
                self.encoder = encoder(input_size, advocate_capacity)
            for i in range(self.num_advocates):
                self.advocates.add_module(name=str(i), module=nn.Sequential(self.encoder, decoder(input_size)))
        else:
            for i in range(self.num_advocates):
                self.advocates.add_module(name=str(i), module=encoder_decoder(input_size=input_size,
                                                                              capacity=advocate_capacity))
        if judge_capacity is None:
            self.decision_module = judge(input_size=(self.num_advocates*input_size[0], input_size[1], input_size[2]),
                                         num_class=self.num_class)
        else:
            self.decision_module = judge(input_size=(self.num_advocates * input_size[0], input_size[1], input_size[2]),
                                         num_class=self.num_class, capacity=judge_capacity)
        self.relu = torch.nn.ReLU()
        self.sm = torch.nn.Softmax(dim=1)

    def process_attention(self, attn):
        return self.relu(attn)

    def get_attention(self, x):
        attention = []
        for i in range(self.num_advocates):
            attn = self.advocates[i](x)
            attn = self.process_attention(attn)
            attention.append(attn)
        attention = torch.cat(attention, dim=1)
        return attention

    def attention_func(self, attention, x):
        if self.ts:
            attn_x = attention * x.repeat(1, self.num_advocates, 1)
        else:
            attn_x = attention * x.repeat(1, self.num_advocates, 1, 1)
        return attn_x

    def forward(self, x):
        attention = self.get_attention(x)
        attn_x = self.attention_func(attention, x)
        out = self.decision_module(attn_x)
        return out, attention


class MIMICNet(nn.Module):
    """
    Judge CNN for MIMIC
    """
    def __init__(self, input_size, num_class):
        super(MIMICNet, self).__init__()
        self.convolution = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv1d(in_channels=input_size[0], out_channels=64, kernel_size=3)),
            ('bn1_1', nn.BatchNorm1d(num_features=64)),
            ('relu1_1', nn.ReLU()),
            ('maxpool1', nn.MaxPool1d(kernel_size=2)),

            ('conv2_1', nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)),
            ('bn2_1', nn.BatchNorm1d(num_features=64)),
            ('relu2_1', nn.ReLU()),
            ('maxpool2', nn.MaxPool1d(kernel_size=2))
        ]))

        feature_size = self.determine_feature_size((1, input_size[0], 48))

        self.dense = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(in_features=feature_size, out_features=64)),
            ('bn_d', nn.BatchNorm1d(num_features=64)),
            ('relu_d', nn.ReLU()),
            ('dropout', nn.Dropout(p=0.2)),
            ('out', nn.Linear(64, num_class))
        ]))

    def determine_feature_size(self, input_size):
        fake_input = Variable(torch.randn(input_size))
        fake_out = self.convolution(fake_input)
        return fake_out.view(-1).shape[0]

    def forward(self, x):
        feat = self.convolution(x)
        feat = feat.view(x.size(0), -1)
        return self.dense(feat)


class TSConvEncoder(nn.Module):
    """
    A convolutional encoder, splitting apart encoder-decoder for shared encoder scheme
    """
    def __init__(self, input_size):
        super(TSConvEncoder, self).__init__()
        self.convolution = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv1d(in_channels=76, out_channels=32, kernel_size=3)),
            ('bn1_1', nn.BatchNorm1d(num_features=32)),
            ('relu1_1', nn.ReLU()),
            ('conv1_2', nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3)),
            ('bn1_2', nn.BatchNorm1d(num_features=32)),
            ('relu1_2', nn.ReLU()),
            ('maxpool1', nn.MaxPool1d(kernel_size=2)),

            ('conv2_1', nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)),
            ('bn2_1', nn.BatchNorm1d(num_features=64)),
            ('relu2_1', nn.ReLU()),
            ('conv2_2', nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)),
            ('bn2_2', nn.BatchNorm1d(num_features=64)),
            ('relu2_2', nn.ReLU()),
            ('maxpool2', nn.MaxPool1d(kernel_size=2))
        ]))

    def forward(self, x):
        return self.convolution(x)


class TSConvDecoder(nn.Module):
    """
    Decoding counterpart to ConvEncoder
    """
    def __init__(self, input_size):
        super(TSConvDecoder, self).__init__()
        self.deconvolution = nn.Sequential(OrderedDict([
            ('deconv1', nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3, stride=1)),
            ('bn1', nn.BatchNorm1d(num_features=32)),
            ('relu1', nn.ReLU()),
            ('deconv2', nn.ConvTranspose1d(in_channels=32, out_channels=32, kernel_size=2, stride=2)),
            ('bn2', nn.BatchNorm1d(num_features=32)),
            ('relu2', nn.ReLU()),
            ('deconv3', nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=2, stride=2)),
            ('bn3', nn.BatchNorm1d(num_features=64)),
            ('relu3', nn.ReLU()),
            ('deconv4', nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size=5, stride=1)),
            ('bn4', nn.BatchNorm1d(num_features=64)),
            ('relu4', nn.ReLU()),

            ('conv1', nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)),
            ('conv_bn1', nn.BatchNorm1d(num_features=64)),
            ('conv2', nn.Conv1d(in_channels=64, out_channels=76, kernel_size=1))
        ]))

    def forward(self, x):
        return self.deconvolution(x)


class TSConvDeconv(nn.Module):
    """
    Hyper-simple convolutional encoder-decoder to replace hourglass
    """
    def __init__(self, input_size):
        super(TSConvDeconv, self).__init__()
        self.convolution = TSConvEncoder(input_size)
        self.deconvolution = TSConvDecoder(input_size)

    def forward(self, x):
        compress = self.convolution(x)
        decompress = self.deconvolution(compress)
        return decompress
