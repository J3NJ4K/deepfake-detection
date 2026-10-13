from pathlib import Path
import torch
from detection.xception_net import Xception

_WEIGHTS_PATH = Path(__file__).parent.parent / "models" / "xception_best.pth"


def load_model() -> Xception:
    """Build Xception and load the pretrained DeepfakeBench weights (FaceForensics++ c23)."""
    model = Xception()
    # DeepfakeBench saved the network as "backbone", so removes that prefix from the keys
    state_dict = torch.load(_WEIGHTS_PATH, map_location="cpu")
    state_dict = {k.removeprefix("backbone."): v for k, v in
                  state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def predict(model: Xception, tensor: torch.Tensor) -> float:
    """Return the probability (0-1) that the face is manipulated."""
    # softmax turns the two outputs into probabilities, class 1 = manipulated
    logits = model(tensor)
    return torch.softmax(logits, dim=1)[0, 1].item()
