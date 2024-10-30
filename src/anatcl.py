"""
Author: Carlo Alberto Barbano <carlo.barbano@unito.it>
Date: 29/10/24
"""
import torch
import torch.nn as nn
import models


WEIGHTS_URLS = {
    'anatcl-g3': {
        'resnet18': {
            'local': [
                "https://gitlab.di.unito.it/barbano/anatcl-pretrained/-/raw/main/resnet18-anatcl-l3/fold0.pth",
                "https://gitlab.di.unito.it/barbano/anatcl-pretrained/-/raw/main/resnet18-anatcl-l3/fold1.pth",
                "https://gitlab.di.unito.it/barbano/anatcl-pretrained/-/raw/main/resnet18-anatcl-l3/fold2.pth",
                "https://gitlab.di.unito.it/barbano/anatcl-pretrained/-/raw/main/resnet18-anatcl-l3/fold3.pth",
                "https://gitlab.di.unito.it/barbano/anatcl-pretrained/-/raw/main/resnet18-anatcl-l3/fold4.pth",
            ],
            'global': [
                "https://gitlab.di.unito.it/barbano/anatcl-pretrained/-/raw/main/resnet18-anatcl-g3/fold0.pth",
                "https://gitlab.di.unito.it/barbano/anatcl-pretrained/-/raw/main/resnet18-anatcl-g3/fold1.pth",
                "https://gitlab.di.unito.it/barbano/anatcl-pretrained/-/raw/main/resnet18-anatcl-g3/fold2.pth",
                "https://gitlab.di.unito.it/barbano/anatcl-pretrained/-/raw/main/resnet18-anatcl-g3/fold3.pth",
                "https://gitlab.di.unito.it/barbano/anatcl-pretrained/-/raw/main/resnet18-anatcl-g3/fold4.pth",
            ]
        }
    }
}


class AnatCL(nn.Module):
    def __init__(self, model="resnet18", descriptor="global", fold=0, use_head=False,
                 pretrained=True):
        super().__init__()

        self.model = model
        self.descriptor = descriptor
        self.fold = fold
        self.use_head = use_head

        self.backbone = models.SupConResNet(model, feat_dim=128, use_head=use_head)

        # Download weights from url
        weights_url = WEIGHTS_URLS['anatcl-g3'][model][descriptor][fold]
        if pretrained:
            print("Downloading weights from", weights_url)
            checkpoint = torch.hub.load_state_dict_from_url(weights_url, map_location="cpu")
            self.backbone.load_state_dict(checkpoint['model'])

    def forward(self, x):
        return self.backbone(x)