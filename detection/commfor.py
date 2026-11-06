import cv2
import numpy as np
import timm
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

_REPO = "OwensLab/commfor-model-384"

resize_size = 440
input_size = 384
imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_commfor() -> torch.nn.Module:
    model = timm.create_model("vit_small_patch16_384", pretrained=False, num_classes=1)
    state = load_file(hf_hub_download(_REPO, "model.safetensors"))
    # the original wraps the timm model as self.vit, so the keys start with "vit."
    model.load_state_dict({k.removeprefix("vit."): v for k, v in state.items()})
    model.eval()
    return model


def to_commfor_tensor(image: np.ndarray) -> torch.Tensor:
    # shortest side -> 440, then center crop 384
    h, w = image.shape[:2]
    scale = resize_size / min(h, w)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    image = cv2.resize(image, (max(resize_size, round(w * scale)), max(resize_size, round(h * scale))),
                       interpolation=interp)
    h, w = image.shape[:2]
    top, left = (h - input_size) // 2, (w - input_size) // 2
    image = image[top:top + input_size, left:left + input_size]

    x = (image.astype(np.float32) / 255.0 - imagenet_mean) / imagenet_std
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)


@torch.no_grad()
def predict_commfor(model: torch.nn.Module, tensor: torch.Tensor) -> float:
    # sigmoid gives the probability that the image is AI-generated
    return torch.sigmoid(model(tensor))[0, 0].item()