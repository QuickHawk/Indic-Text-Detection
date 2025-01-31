from torch import sigmoid, reciprocal, exp
from torch.nn import Module, Conv2d
from torch.nn.functional import interpolate

from transformers import AutoConfig, UperNetForSemanticSegmentation


class TrDBNet(Module):
    '''
    TrDBNet is a Transformer-based Differential Binarization Network for semantic segmentation.
    '''

    def __init__(self, pretrained=False, k=50, backbone='swin', variant='base', *args, **kwargs):

        if variant not in {None, 'tiny', 'small', 'base', 'large'}:
            raise ValueError(
                "variant must be one of 'tiny', 'small', 'base', or 'large'")

        if backbone not in {'swin', 'convnext'}:
            raise ValueError("backbone must be one of 'swin' or 'convnext'")

        super().__init__(*args, **kwargs)

        config = AutoConfig.from_pretrained(
            f"openmmlab/upernet-{backbone}-{variant}")

        self.k = k
        self.pretrained = pretrained

        if self.pretrained:
            self.model = UperNetForSemanticSegmentation.from_pretrained(
                f"openmmlab/upernet-{backbone}-{variant}", config=config)
        else:
            self.model = UperNetForSemanticSegmentation(config)

        self.conv1 = Conv2d(150, 1, 1)
        self.conv2 = Conv2d(150, 1, 1)

    def forward(self, x):

        outputs = self.model(x)

        outputs = outputs.logits

        outputs = interpolate(
            outputs, size=x.shape[2:], mode='bilinear', align_corners=False)

        probability_map = sigmoid(self.conv1(outputs))
        threshold_map = sigmoid(self.conv2(outputs))

        binary_map = reciprocal(
            1 + exp(-self.k * (probability_map-threshold_map)))

        return probability_map, threshold_map, binary_map
