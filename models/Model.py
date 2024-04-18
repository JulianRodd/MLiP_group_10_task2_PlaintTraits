from torch import nn
import timm 


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = timm.create_model(
                config.BACKBONE,
                num_classes=config.N_TARGETS,
                pretrained=True)
        
    def forward(self, inputs):
        return self.backbone(inputs)