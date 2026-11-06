import cv2
import numpy as np
import timm
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

_REPO = "OwensLab/commfor-model-384"

RESIZE_SIZE = 440  # same eval preprocessing as the Community Forensics repo: resize 440, center crop 384
INPUT_SIZE = 384
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_commfor() -> torch.nn.Module:
    # ViT-S/16 at 384 px with a single-logit head, like ViTClassifier in the original repo (~87 MB download)
    model = timm.create_model("vit_small_patch16_384", pretrained=False, num_classes=1)
    state = load_file(hf_hub_download(_REPO, "model.safetensors"))
    # the original wraps the timm model as self.vit, so the keys start with "vit."
    model.load_state_dict({k.removeprefix("vit."): v for k, v in state.items()})
    model.eval()
    return model


def to_commfor_tensor(image: np.ndarray) -> torch.Tensor:
    # shortest side -> 440, then center crop 384
    h, w = image.shape[:2]
    scale = RESIZE_SIZE / min(h, w)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    image = cv2.resize(image, (max(RESIZE_SIZE, round(w * scale)), max(RESIZE_SIZE, round(h * scale))),
                       interpolation=interp)
    h, w = image.shape[:2]
    top, left = (h - INPUT_SIZE) // 2, (w - INPUT_SIZE) // 2
    image = image[top:top + INPUT_SIZE, left:left + INPUT_SIZE]

    x = (image.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)


@torch.no_grad()
def predict_commfor(model: torch.nn.Module, tensor: torch.Tensor) -> float:
    # label 1 = fake, so sigmoid gives the probability that the image is AI-generated
    return torch.sigmoid(model(tensor))[0, 0].item()