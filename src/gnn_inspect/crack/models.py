import timm
import torch

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

ARCHS = {
    "resnet18": "resnet18.a1_in1k",           # CNN, 11.2M params
    "deit_tiny": "deit_tiny_patch16_224.fb_in1k",  # ViT-Tiny/16, 5.5M params
}


def build(arch: str, pretrained: bool = True) -> torch.nn.Module:
    return timm.create_model(ARCHS[arch], pretrained=pretrained, num_classes=2)


def prep(x_uint8: torch.Tensor) -> torch.Tensor:
    """(B,224,224,3) uint8 -> normalized float (B,3,224,224)."""
    x = x_uint8.permute(0, 3, 1, 2).float() / 255.0
    return (x - MEAN.to(x.device)) / STD.to(x.device)
