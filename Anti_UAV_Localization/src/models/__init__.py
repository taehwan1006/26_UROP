import torch.nn as nn

from .thin_dy_unet import ThinDyUNet
from .thin_unet import ThinUNet

MODELS = {
    "thindyunet": ThinDyUNet,
    "thinunet": ThinUNet,
}


def build_model(model_cfg: dict) -> nn.Module:
    """config의 model 블록으로부터 모델을 만든다.

    `name`으로 ThinDyUNet / ThinUNet(ablation)을 고른다. ThinUNet은 regular conv라
    n_kernels를 받지 않으므로 여기서 걸러낸다.
    """
    name = model_cfg.get("name", "ThinDyUNet")
    key = name.lower().replace("_", "").replace("-", "")
    if key not in MODELS:
        raise ValueError(f"Unknown model: {name} (available: {sorted(MODELS)})")

    kwargs = {
        "in_channels": model_cfg.get("in_channels", 3),
        "n_classes": model_cfg.get("n_classes", 1),
        "base_ch": model_cfg.get("base_ch", 64),
    }
    if key == "thindyunet":
        kwargs["n_kernels"] = model_cfg.get("n_kernels", 3)

    return MODELS[key](**kwargs)
