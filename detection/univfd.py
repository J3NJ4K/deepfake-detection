from pathlib import Path
import cv2
import numpy as np
import torch
from transformers import CLIPVisionModelWithProjection

_WEIGHTS_PATH = Path(__file__).parent.parent / "models" / "univfd_fc.pth"
_CLIP_NAME = "openai/clip-vit-large-patch14"

input_size = 224  # CLIP ViT-L/14 input resolution
# CLIP's own normalization
clip_mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
clip_std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


class UnivFD(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.clip = CLIPVisionModelWithProjection.from_pretrained(_CLIP_NAME)
        self.fc = torch.nn.Linear(768, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.clip(pixel_values=x).image_embeds)


def load_univfd() -> UnivFD:
    # downloads CLIP (~1.7 GB) the first time, then uses the Hugging Face cache
    model = UnivFD()
    model.fc.load_state_dict(torch.load(_WEIGHTS_PATH, map_location="cpu"))
    model.eval()
    return model


def to_clip_tensor(image: np.ndarray) -> torch.Tensor:
    # shortest side -> 224, then center crop, so the image isn't streched
    h, w = image.shape[:2]
    scale = input_size / min(h, w)
    image = cv2.resize(image, (max(input_size, round(w * scale)), max(input_size, round(h * scale))),
                       interpolation=cv2.INTER_CUBIC)
    h, w = image.shape[:2]
    top, left = (h - input_size) // 2, (w - input_size) // 2
    image = image[top:top + input_size, left:left + input_size]

    x = (image.astype(np.float32) / 255.0 - clip_mean) / clip_std
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)


@torch.no_grad()
def predict_generated(model: UnivFD, tensor: torch.Tensor) -> float:
    # sigmoid gives the probability that the image is AI-generated
    return torch.sigmoid(model(tensor))[0, 0].item()