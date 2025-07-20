import io

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from utils.data import FacadImage


class CLIPMixin:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        """
        CLIP類似度マトリックス作成クラス

        Args:
            model_name: 使用するCLIPモデル名
        """
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.model.to(self.device)

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """
        テキストをエンコード

        Args:
            texts: テキストのリスト

        Returns:
            テキストの特徴量テンソル
        """
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            text_features: torch.Tensor = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(
                p=2, dim=-1, keepdim=True
            )

        return text_features

    def encode_images(self, images: list[FacadImage]) -> torch.Tensor:
        """
        画像をエンコード

        Args:
            images: FacadImageのリスト

        Returns:
            画像の特徴量テンソル
        """
        pil_images = []
        for img in images:
            assert img.image is not None, "None image provided to encode_images"
            pil_images.append(Image.open(io.BytesIO(img.image)))

        inputs = self.processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features: torch.Tensor = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(
                p=2, dim=-1, keepdim=True
            )

        return image_features
