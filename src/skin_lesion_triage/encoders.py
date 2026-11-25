"""Frozen foundation-model encoders that follow the `Encoder` contract.

Each encoder loads its model once, when it is created, and then turns a list of
images into one embedding row per image. The models are used as-is ("frozen"):
no training happens here.

PyTorch and the model libraries are imported *inside* the encoders, not at the
top of this file. That way this module (and the `ENCODERS` registry) can be
imported without them, which keeps local runs and CI free of PyTorch. Install
them with the `embeddings` dependency group (e.g. in Colab).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

DINOV2_MODEL = "facebook/dinov2-base"
BIOMEDCLIP_MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def _pick_device(device: str | None) -> str:
    """Use the requested device, or the GPU if there is one, else the CPU."""
    import torch

    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class DinoV2Encoder:
    """DINOv2 (Meta): a general-purpose vision model trained on 142M natural images.

    The embedding is the model's summary token (CLS), 768 numbers for the base model.
    """

    def __init__(self, model_name: str = DINOV2_MODEL, device: str | None = None) -> None:
        from transformers import AutoImageProcessor, AutoModel

        self.device = _pick_device(device)
        # The processor resizes, crops to 224x224 and normalises colours exactly as
        # the model saw images during its own training.
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        # `.eval()` switches off training-only behaviour, so outputs are deterministic.
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def __call__(self, images: list[Image.Image]) -> np.ndarray:
        import torch

        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        # `inference_mode` skips the bookkeeping needed for training: faster, less memory.
        with torch.inference_mode():
            outputs = self.model(**inputs)
        return outputs.pooler_output.float().cpu().numpy()


class BiomedClipEncoder:
    """BiomedCLIP (Microsoft): trained on 15M figure-caption pairs from biomedical papers.

    The embedding is the image half of the CLIP model, 512 numbers.
    """

    def __init__(self, model_name: str = BIOMEDCLIP_MODEL, device: str | None = None) -> None:
        import open_clip

        self.device = _pick_device(device)
        self.model, self.preprocess = open_clip.create_model_from_pretrained(model_name)
        self.model = self.model.to(self.device).eval()

    def __call__(self, images: list[Image.Image]) -> np.ndarray:
        import torch

        # Unlike the DINOv2 processor, `preprocess` handles one image at a time,
        # so we preprocess each one and stack them into a batch.
        batch = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        with torch.inference_mode():
            features = self.model.encode_image(batch)
        return features.float().cpu().numpy()


# Name -> encoder class. To compare models, change only the name.
ENCODERS = {
    "dinov2": DinoV2Encoder,
    "biomedclip": BiomedClipEncoder,
}


def get_encoder(name: str, device: str | None = None) -> DinoV2Encoder | BiomedClipEncoder:
    """Create the encoder registered under `name` (this downloads the model the first time)."""
    if name not in ENCODERS:
        raise ValueError(f"unknown encoder {name!r}; choose one of {sorted(ENCODERS)}")
    return ENCODERS[name](device=device)
