from __future__ import annotations

import logging
from dataclasses import dataclass, field

from langchain_core.tools import StructuredTool, Tool
from PIL import Image

from src.llm.chatbot import answer_with_context
from src.retrieval.rag import RAGEngine
from src.vision.clip_classifier import get_classifier
from src.vision.plant_labels import find_by_label

logger = logging.getLogger(__name__)


@dataclass
class SessionContext:
    image: Image.Image | None = None
    identified_label: str | None = None
    identified_common: str | None = None
    identified_confidence: float | None = None
    rag: RAGEngine = field(default_factory=RAGEngine)

    def reset_plant(self) -> None:
        self.identified_label = None
        self.identified_common = None
        self.identified_confidence = None
        self.rag = RAGEngine()


def build_tools(ctx: SessionContext) -> list[Tool]:
    def identify_plant(_: str = "") -> str:
        if ctx.image is None:
            return ("Brak przesłanego zdjęcia. Poproś użytkownika o przesłanie "
                    "fotografii rośliny.")
        classifier = get_classifier()
        predictions = classifier.classify(ctx.image)
        best = predictions[0]
        ctx.identified_label = best.label
        ctx.identified_common = best.common
        ctx.identified_confidence = best.score

        top3 = ", ".join(f"{p.common} ({p.score:.0%})" for p in predictions[:3])
        return (
            f"Rozpoznano roślinę: {best.common} (pewność {best.score:.0%}). "
            f"Najlepsze dopasowania: {top3}."
        )

    def gather_care_information(plant_name: str = "") -> str:
        label = ctx.identified_label
        query: str

        if plant_name:
            pc = find_by_label(plant_name) or find_by_label(plant_name.lower())
            query = (pc.query if pc else f"how to take care of {plant_name} houseplant")
            display = pc.common if pc else plant_name
        elif label:
            pc = find_by_label(label)
            query = pc.query if pc else f"how to take care of {label}"
            display = pc.common if pc else label
        else:
            return ("Nie wiadomo, o jaką roślinę chodzi. Najpierw rozpoznaj roślinę "
                    "(`identify_plant`) lub podaj jej nazwę.")

        count = ctx.rag.build_knowledge(search_query=query, plant_name=display)
        if count == 0:
            return f"Nie udało się znaleźć informacji o pielęgnacji rośliny: {display}."
        return (
            f"Zebrano i zaindeksowano informacje o pielęgnacji rośliny '{display}' "
            f"({count} fragmentów z {len(ctx.rag.sources)} źródeł). "
            f"Można teraz odpowiadać na pytania o pielęgnację."
        )

    def answer_care_question(question: str) -> str:
        if not ctx.rag.is_ready:
            return ("Baza wiedzy nie została jeszcze zbudowana. Najpierw użyj "
                    "narzędzia `gather_care_information`.")
        context = ctx.rag.retrieve_context(question)
        return answer_with_context(question, context)

    return [
        StructuredTool.from_function(
            func=identify_plant,
            name="identify_plant",
            description=(
                "Rozpoznaje gatunek rośliny domowej na podstawie przesłanego przez "
                "użytkownika zdjęcia (model CLIP). Używaj, gdy jest zdjęcie, a gatunek "
                "nie jest jeszcze znany. Nie wymaga argumentów."
            ),
        ),
        StructuredTool.from_function(
            func=gather_care_information,
            name="gather_care_information",
            description=(
                "Wyszukuje w internecie i indeksuje informacje o pielęgnacji rośliny. "
                "Argument `plant_name` jest opcjonalny — jeśli pominięty, użyje "
                "wcześniej rozpoznanego gatunku. Używaj po rozpoznaniu rośliny lub gdy "
                "użytkownik poda nazwę rośliny."
            ),
        ),
        StructuredTool.from_function(
            func=answer_care_question,
            name="answer_care_question",
            description=(
                "Odpowiada na konkretne pytanie o pielęgnację rośliny w oparciu o "
                "zebraną wiedzę (RAG). Argument `question` to pytanie użytkownika. "
                "Wymaga wcześniejszego użycia `gather_care_information`."
            ),
        ),
    ]
