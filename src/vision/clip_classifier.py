from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from config import settings
from src.vision.plant_labels import PLANT_CLASSES, PlantClass

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    plant: PlantClass
    score: float

    @property
    def label(self) -> str:
        return self.plant.label

    @property
    def common(self) -> str:
        return self.plant.common


class PlantClassifier:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.clip_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Ładowanie modelu CLIP '%s' na urządzeniu %s", self.model_name, self.device)

        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(self.model_name)

        self.prompts = [
            settings.clip_prompt_template.format(label=p.label) for p in PLANT_CLASSES
        ]

    @torch.no_grad()
    def classify(self, image: Image.Image, top_k: int | None = None) -> list[Prediction]:
        top_k = top_k or settings.clip_top_k
        image = image.convert("RGB")

        inputs = self.processor(
            text=self.prompts,
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        outputs = self.model(**inputs)
        logits = outputs.logits_per_image.softmax(dim=-1).squeeze(0)

        scores = logits.cpu().tolist()
        ranked = sorted(
            (Prediction(plant=PLANT_CLASSES[i], score=float(s)) for i, s in enumerate(scores)),
            key=lambda p: p.score,
            reverse=True,
        )
        return ranked[:top_k]

    def best(self, image: Image.Image) -> Prediction:
        return self.classify(image, top_k=1)[0]


@lru_cache(maxsize=1)
def get_classifier() -> PlantClassifier:
    return PlantClassifier()
