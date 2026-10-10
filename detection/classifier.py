from pathlib import Path
import torch
from detection.xception_net import Xception

_WEIGHTS_PATH = Path(__file__).parent.parent / "models" / "xception_best.pth"


def load_model() -> Xception:
    model = Xception()
    state_dict = torch.load(_WEIGHTS_PATH, map_location="cpu")
    state_dict = {k.removeprefix("backbone."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model