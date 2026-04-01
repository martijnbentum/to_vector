import numpy as np
import torch
from torch import nn
from unittest import mock


class DummyModel:
    def __init__(self, name_or_path='repo/model'):
        self.name_or_path = name_or_path
        self.base_model_prefix = 'dummy'
        self.moves = []

    def to(self, device):
        self.moves.append(device)
        return self

    def parameters(self):
        return iter([np.zeros(1)])


class DummyParameter:
    def __init__(self, device_type='cpu'):
        self.device = mock.Mock(type=device_type)


class DeviceModel(DummyModel):
    def __init__(self, device_type='cpu'):
        super().__init__()
        self._parameter = DummyParameter(device_type)

    def parameters(self):
        return iter([self._parameter])


class FakeSpidrModule:
    def __init__(self, outputs):
        self.outputs = outputs

    def get_intermediate_outputs(self, features):
        return self.outputs


class FakeSpidrModel(DeviceModel):
    __module__ = 'spidr.tests'

    def __init__(self):
        super().__init__(device_type='cpu')
        self.base_model_prefix = 'spidr'
        self.feature_extractor = mock.Mock(
            return_value=torch.tensor([[[1.0, 2.0]]]))
        self.feature_projection = mock.Mock(
            side_effect=lambda value: value + 1.0)
        self.student = FakeSpidrModule([
            torch.tensor([[[3.0, 4.0]]]),
            torch.tensor([[[5.0, 6.0]]]),
        ])
        self.teacher = FakeSpidrModule([
            torch.tensor([[[7.0, 8.0]]]),
        ])
        self.get_codebooks = mock.Mock(
            return_value=[torch.tensor([[[9.0, 10.0]]])])


class FakeHuggingFaceModel(DeviceModel):
    def __init__(self, outputs):
        super().__init__(device_type='cpu')
        self.outputs = outputs

    def __call__(self, **kwargs):
        return self.outputs


class FakeSpidrFeatureExtractor(nn.Module):
    def forward(self, waveform):
        return waveform.unsqueeze(-1).repeat(1, 1, 4)


class ZeroPositionalEmbedding(nn.Module):
    def forward(self, x, attention_mask=None):
        return torch.zeros_like(x)


class FakeSpidrAttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.Module()
        self.attention.embed_dim = 4
        self.attention.num_heads = 2
        self.attention.dropout = 0.0
        self.attention.qkv = nn.Linear(4, 12, bias=False)
        self.attention.proj = nn.Linear(4, 4, bias=False)
        self.dropout = nn.Identity()
        self.layer_norm = nn.Identity()
        self.final_layer_norm = nn.Identity()
        self.feed_forward = nn.Identity()
        self.layer_norm_first = False
        with torch.no_grad():
            eye = torch.eye(4)
            self.attention.qkv.weight.copy_(torch.cat([eye, eye, eye], dim=0))
            self.attention.proj.weight.copy_(eye)


class FakeSpidrStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_conv_embed = ZeroPositionalEmbedding()
        self.layer_norm = nn.Identity()
        self.layer_norm_first = False
        self.dropout = nn.Identity()
        self.layers = nn.ModuleList([FakeSpidrAttentionLayer()])


class FakeSpidrAttentionModel(DeviceModel):
    __module__ = 'spidr.tests'

    def __init__(self):
        super().__init__(device_type='cpu')
        self.base_model_prefix = 'spidr'
        self.feature_extractor = FakeSpidrFeatureExtractor()
        self.feature_projection = nn.Identity()
        self.student = FakeSpidrStudent()
