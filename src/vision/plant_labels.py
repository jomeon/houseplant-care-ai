from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlantClass:
    label: str
    common: str
    query: str


PLANT_CLASSES: list[PlantClass] = [
    PlantClass("monstera deliciosa", "Monstera dziurawa (Monstera deliciosa)", "Monstera deliciosa care"),
    PlantClass("snake plant sansevieria", "Sansewieria / Wężownica (Sansevieria)", "Sansevieria snake plant care"),
    PlantClass("golden pothos", "Epipremnum / Pothos (Epipremnum aureum)", "Pothos Epipremnum aureum care"),
    PlantClass("peace lily spathiphyllum", "Skrzydłokwiat (Spathiphyllum)", "Peace lily Spathiphyllum care"),
    PlantClass("dracaena", "Dracena (Dracaena)", "Dracaena houseplant care"),
    PlantClass("fiddle leaf fig ficus lyrata", "Figowiec lirolistny (Ficus lyrata)", "Fiddle leaf fig care"),
    PlantClass("rubber plant ficus elastica", "Figowiec sprężysty (Ficus elastica)", "Rubber plant Ficus elastica care"),
    PlantClass("zz plant zamioculcas", "Zamiokulkas (Zamioculcas zamiifolia)", "ZZ plant Zamioculcas care"),
    PlantClass("aloe vera", "Aloes (Aloe vera)", "Aloe vera plant care"),
    PlantClass("spider plant chlorophytum", "Zielistka (Chlorophytum comosum)", "Spider plant care"),
    PlantClass("boston fern", "Paproć bostońska (Nephrolepis)", "Boston fern care"),
    PlantClass("english ivy hedera", "Bluszcz pospolity (Hedera helix)", "English ivy Hedera care"),
    PlantClass("philodendron", "Filodendron (Philodendron)", "Philodendron houseplant care"),
    PlantClass("calathea", "Kalatea (Calathea)", "Calathea plant care"),
    PlantClass("orchid phalaenopsis", "Storczyk (Phalaenopsis)", "Phalaenopsis orchid care"),
    PlantClass("succulent echeveria", "Sukulent Echeveria", "Echeveria succulent care"),
    PlantClass("cactus", "Kaktus", "indoor cactus care"),
    PlantClass("jade plant crassula", "Grubosz / Drzewko szczęścia (Crassula ovata)", "Jade plant Crassula care"),
    PlantClass("anthurium", "Anturium (Anthurium)", "Anthurium plant care"),
    PlantClass("begonia", "Begonia (Begonia)", "Begonia houseplant care"),
    PlantClass("african violet saintpaulia", "Fiołek afrykański (Saintpaulia)", "African violet care"),
    PlantClass("croton codiaeum", "Kroton (Codiaeum variegatum)", "Croton plant care"),
    PlantClass("areca palm", "Palma areka (Dypsis lutescens)", "Areca palm care"),
    PlantClass("chinese money plant pilea", "Pilea peperomioides", "Pilea peperomioides care"),
    PlantClass("string of pearls senecio", "Senecio rowleyanus (sznur pereł)", "String of pearls plant care"),
    PlantClass("aglaonema chinese evergreen", "Aglaonema (Aglaonema)", "Aglaonema Chinese evergreen care"),
    PlantClass("dieffenbachia", "Difenbachia (Dieffenbachia)", "Dieffenbachia plant care"),
    PlantClass("hoya wax plant", "Hoja / Woskownica (Hoya)", "Hoya wax plant care"),
    PlantClass("schefflera umbrella plant", "Szeflera (Schefflera)", "Schefflera umbrella plant care"),
    PlantClass("peperomia", "Pieprzowiec (Peperomia)", "Peperomia plant care"),
]


def all_labels() -> list[str]:
    return [p.label for p in PLANT_CLASSES]


def find_by_label(label: str) -> PlantClass | None:
    for p in PLANT_CLASSES:
        if p.label == label:
            return p
    return None
