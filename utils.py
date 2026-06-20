"""
NLP Mythology Knowledge Graph - Utilities Module
================================================

Funciones auxiliares para el procesamiento de textos mitológicos,
extracción de entidades y construcción del grafo de conocimiento.

Author: César Adrián Delgado Díaz
License: MIT
"""

import re
import json
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

import networkx as nx

# =============================================================================
# ENUMERACIONES Y CLASES DE DATOS
# =============================================================================

class EntityType(Enum):
    """Tipos de entidades mitológicas."""
    DEITY = "DEITY"           # Dioses
    HERO = "HERO"             # Héroes
    CREATURE = "CREATURE"     # Criaturas míticas
    LOCATION = "LOCATION"     # Lugares míticos
    ARTIFACT = "ARTIFACT"     # Objetos mágicos
    EVENT = "EVENT"           # Eventos mitológicos
    TITAN = "TITAN"           # Titanes
    MORTAL = "MORTAL"         # Mortales importantes


class RelationType(Enum):
    """Tipos de relaciones entre entidades."""
    FATHER_OF = "FATHER_OF"
    MOTHER_OF = "MOTHER_OF"
    CHILD_OF = "CHILD_OF"
    SIBLING_OF = "SIBLING_OF"
    SPOUSE_OF = "SPOUSE_OF"
    RIVAL_OF = "RIVAL_OF"
    ALLY_OF = "ALLY_OF"
    KILLED = "KILLED"
    KILLED_BY = "KILLED_BY"
    CREATED = "CREATED"
    RULES = "RULES"
    LIVES_IN = "LIVES_IN"
    OWNS = "OWNS"
    PARTICIPATED_IN = "PARTICIPATED_IN"
    TRANSFORMED_INTO = "TRANSFORMED_INTO"


class Mythology(Enum):
    """Mitologías soportadas."""
    GREEK = "Greek"
    ROMAN = "Roman"
    NORSE = "Norse"
    EGYPTIAN = "Egyptian"
    UNKNOWN = "Unknown"


@dataclass
class MythEntity:
    """Representa una entidad mitológica."""
    name: str
    entity_type: EntityType
    mythology: Mythology = Mythology.UNKNOWN
    aliases: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    
    def to_dict(self) -> Dict:
        """Convierte la entidad a diccionario."""
        return {
            "name": self.name,
            "entity_type": self.entity_type.value,
            "mythology": self.mythology.value,
            "aliases": self.aliases,
            "attributes": self.attributes,
            "description": self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "MythEntity":
        """Crea una entidad desde un diccionario."""
        return cls(
            name=data["name"],
            entity_type=EntityType(data["entity_type"]),
            mythology=Mythology(data.get("mythology", "Unknown")),
            aliases=data.get("aliases", []),
            attributes=data.get("attributes", {}),
            description=data.get("description", "")
        )


@dataclass
class MythRelation:
    """Representa una relación entre entidades mitológicas."""
    source: str
    target: str
    relation_type: RelationType
    confidence: float = 1.0
    source_text: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convierte la relación a diccionario."""
        return {
            "source": self.source,
            "target": self.target,
            "relation_type": self.relation_type.value,
            "confidence": self.confidence,
            "source_text": self.source_text,
            "attributes": self.attributes
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "MythRelation":
        """Crea una relación desde un diccionario."""
        return cls(
            source=data["source"],
            target=data["target"],
            relation_type=RelationType(data["relation_type"]),
            confidence=data.get("confidence", 1.0),
            source_text=data.get("source_text", ""),
            attributes=data.get("attributes", {})
        )


# =============================================================================
# PATRONES PARA EXTRACCIÓN
# =============================================================================

# Patrones de relaciones familiares
FAMILY_PATTERNS = {
    "father": [
        r"(\w+),?\s+(?:the\s+)?father\s+of\s+(\w+)",
        r"(\w+)\s+was\s+(?:the\s+)?father\s+(?:of|to)\s+(\w+)",
        r"(\w+)\s+fathered\s+(\w+)",
        r"(\w+),?\s+son\s+of\s+(\w+)",
    ],
    "mother": [
        r"(\w+),?\s+(?:the\s+)?mother\s+of\s+(\w+)",
        r"(\w+)\s+was\s+(?:the\s+)?mother\s+(?:of|to)\s+(\w+)",
        r"(\w+),?\s+(?:daughter|son)\s+of\s+\w+\s+and\s+(\w+)",
    ],
    "sibling": [
        r"(\w+)\s+and\s+(\w+)\s+were\s+(?:brothers|sisters|siblings)",
        r"(\w+),?\s+(?:brother|sister)\s+of\s+(\w+)",
    ],
    "spouse": [
        r"(\w+)\s+(?:married|wed)\s+(\w+)",
        r"(\w+),?\s+(?:wife|husband)\s+of\s+(\w+)",
        r"(\w+)\s+and\s+(\w+)\s+were\s+(?:married|wed)",
    ]
}

# Patrones de conflicto/alianza
CONFLICT_PATTERNS = {
    "killed": [
        r"(\w+)\s+(?:killed|slew|slayed|defeated)\s+(\w+)",
        r"(\w+)\s+was\s+(?:killed|slain)\s+by\s+(\w+)",
    ],
    "rival": [
        r"(\w+)\s+(?:rival|enemy|foe)\s+(?:of|to)\s+(\w+)",
        r"(\w+)\s+and\s+(\w+)\s+were\s+(?:rivals|enemies)",
    ],
    "ally": [
        r"(\w+)\s+(?:allied|joined)\s+(?:with)?\s*(\w+)",
        r"(\w+)\s+and\s+(\w+)\s+(?:fought\s+together|allied)",
    ]
}

# Diccionario de entidades conocidas por mitología
KNOWN_ENTITIES = {
    Mythology.GREEK: {
        EntityType.DEITY: [
            "Zeus", "Hera", "Poseidon", "Hades", "Athena", "Apollo", 
            "Artemis", "Ares", "Aphrodite", "Hephaestus", "Hermes",
            "Dionysus", "Demeter", "Hestia", "Persephone", "Eros"
        ],
        EntityType.TITAN: [
            "Cronus", "Rhea", "Oceanus", "Tethys", "Hyperion", "Theia",
            "Coeus", "Phoebe", "Crius", "Mnemosyne", "Themis", "Iapetus",
            "Atlas", "Prometheus", "Epimetheus"
        ],
        EntityType.HERO: [
            "Heracles", "Perseus", "Theseus", "Achilles", "Odysseus",
            "Jason", "Orpheus", "Bellerophon", "Aeneas", "Hector"
        ],
        EntityType.CREATURE: [
            "Medusa", "Hydra", "Cerberus", "Minotaur", "Cyclops",
            "Chimera", "Sphinx", "Pegasus", "Phoenix", "Centaur"
        ],
        EntityType.LOCATION: [
            "Olympus", "Tartarus", "Elysium", "Underworld", "Styx",
            "Delphi", "Troy", "Athens", "Thebes", "Crete"
        ],
        EntityType.ARTIFACT: [
            "Aegis", "Trident", "Thunderbolt", "Helm of Darkness",
            "Golden Fleece", "Pandora's Box", "Caduceus"
        ]
    },
    Mythology.NORSE: {
        EntityType.DEITY: [
            "Odin", "Thor", "Loki", "Freya", "Frigg", "Tyr",
            "Balder", "Heimdall", "Njord", "Frey"
        ],
        EntityType.CREATURE: [
            "Fenrir", "Jormungandr", "Sleipnir", "Huginn", "Muninn",
            "Nidhogg", "Ratatoskr"
        ],
        EntityType.LOCATION: [
            "Asgard", "Midgard", "Niflheim", "Muspelheim", "Valhalla",
            "Bifrost", "Yggdrasil"
        ]
    },
    Mythology.EGYPTIAN: {
        EntityType.DEITY: [
            "Ra", "Osiris", "Isis", "Horus", "Set", "Anubis",
            "Thoth", "Bastet", "Sekhmet", "Hathor", "Nephthys"
        ],
        EntityType.LOCATION: [
            "Duat", "Aaru", "Nile"
        ]
    },
    Mythology.ROMAN: {
        EntityType.DEITY: [
            "Jupiter", "Juno", "Neptune", "Pluto", "Minerva",
            "Mars", "Venus", "Mercury", "Diana", "Apollo"
        ]
    }
}

# Mapeo Griego-Romano
GREEK_ROMAN_MAP = {
    "Zeus": "Jupiter",
    "Hera": "Juno",
    "Poseidon": "Neptune",
    "Hades": "Pluto",
    "Athena": "Minerva",
    "Ares": "Mars",
    "Aphrodite": "Venus",
    "Hermes": "Mercury",
    "Artemis": "Diana",
    "Hephaestus": "Vulcan",
    "Demeter": "Ceres",
    "Dionysus": "Bacchus"
}


# =============================================================================
# FUNCIONES DE PROCESAMIENTO DE TEXTO
# =============================================================================

def clean_text(text: str) -> str:
    """
    Limpia y normaliza el texto para procesamiento.
    
    Args:
        text: Texto a limpiar
        
    Returns:
        Texto limpio y normalizado
    """
    # Eliminar caracteres especiales pero mantener puntuación básica
    text = re.sub(r'[^\w\s.,;:!?\'"-]', '', text)
    # Normalizar espacios
    text = re.sub(r'\s+', ' ', text)
    # Eliminar espacios al inicio/final
    text = text.strip()
    return text


def split_into_sentences(text: str) -> List[str]:
    """
    Divide el texto en oraciones.
    
    Args:
        text: Texto a dividir
        
    Returns:
        Lista de oraciones
    """
    # Patrón simple para división de oraciones
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def normalize_entity_name(name: str) -> str:
    """
    Normaliza el nombre de una entidad.
    
    Args:
        name: Nombre a normalizar
        
    Returns:
        Nombre normalizado
    """
    # Capitalizar primera letra de cada palabra
    name = name.strip().title()
    # Eliminar caracteres no alfabéticos
    name = re.sub(r'[^a-zA-Z\s]', '', name)
    return name.strip()


def detect_mythology(text: str, entities: List[str] = None) -> Mythology:
    """
    Detecta la mitología basándose en el texto o entidades.
    
    Args:
        text: Texto a analizar
        entities: Lista opcional de entidades encontradas
        
    Returns:
        Mitología detectada
    """
    text_lower = text.lower()
    entities = entities or []
    
    # Contadores por mitología
    scores = {myth: 0 for myth in Mythology}
    
    # Palabras clave por mitología
    keywords = {
        Mythology.GREEK: ["olympus", "greece", "greek", "hellenic", "athens", "sparta"],
        Mythology.ROMAN: ["rome", "roman", "latin", "jupiter", "capitoline"],
        Mythology.NORSE: ["asgard", "norse", "viking", "valhalla", "ragnarok"],
        Mythology.EGYPTIAN: ["egypt", "egyptian", "pharaoh", "nile", "pyramid"]
    }
    
    # Buscar palabras clave
    for myth, words in keywords.items():
        for word in words:
            if word in text_lower:
                scores[myth] += 1
    
    # Buscar entidades conocidas
    for entity in entities:
        for myth, entity_dict in KNOWN_ENTITIES.items():
            for entity_list in entity_dict.values():
                if entity in entity_list:
                    scores[myth] += 2
    
    # Retornar mitología con mayor puntaje
    best_myth = max(scores, key=scores.get)
    return best_myth if scores[best_myth] > 0 else Mythology.UNKNOWN


# =============================================================================
# FUNCIONES DE EXTRACCIÓN DE RELACIONES
# =============================================================================

def extract_relations_with_patterns(
    text: str,
    entities: Set[str]
) -> List[MythRelation]:
    """
    Extrae relaciones usando patrones regex.
    
    Args:
        text: Texto fuente
        entities: Conjunto de entidades conocidas
        
    Returns:
        Lista de relaciones extraídas
    """
    relations = []
    text_clean = clean_text(text)
    
    # Procesar patrones familiares
    for rel_type, patterns in FAMILY_PATTERNS.items():
        for pattern in patterns:
            matches = re.finditer(pattern, text_clean, re.IGNORECASE)
            for match in matches:
                source = normalize_entity_name(match.group(1))
                target = normalize_entity_name(match.group(2))
                
                # Verificar que ambas entidades son conocidas o están en el texto
                if source and target:
                    relation_type = _map_pattern_to_relation(rel_type, source, target)
                    relations.append(MythRelation(
                        source=source,
                        target=target,
                        relation_type=relation_type,
                        confidence=0.8,
                        source_text=match.group(0)
                    ))
    
    # Procesar patrones de conflicto
    for rel_type, patterns in CONFLICT_PATTERNS.items():
        for pattern in patterns:
            matches = re.finditer(pattern, text_clean, re.IGNORECASE)
            for match in matches:
                source = normalize_entity_name(match.group(1))
                target = normalize_entity_name(match.group(2))
                
                if source and target:
                    relation_type = _map_pattern_to_relation(rel_type, source, target)
                    relations.append(MythRelation(
                        source=source,
                        target=target,
                        relation_type=relation_type,
                        confidence=0.75,
                        source_text=match.group(0)
                    ))
    
    return relations


def _map_pattern_to_relation(pattern_type: str, source: str, target: str) -> RelationType:
    """Mapea tipo de patrón a tipo de relación."""
    mapping = {
        "father": RelationType.FATHER_OF,
        "mother": RelationType.MOTHER_OF,
        "sibling": RelationType.SIBLING_OF,
        "spouse": RelationType.SPOUSE_OF,
        "killed": RelationType.KILLED,
        "rival": RelationType.RIVAL_OF,
        "ally": RelationType.ALLY_OF
    }
    return mapping.get(pattern_type, RelationType.ALLY_OF)


# =============================================================================
# FUNCIONES PARA GRAFO
# =============================================================================

def build_networkx_graph(
    entities: List[MythEntity],
    relations: List[MythRelation]
) -> nx.DiGraph:
    """
    Construye un grafo NetworkX desde entidades y relaciones.
    
    Args:
        entities: Lista de entidades
        relations: Lista de relaciones
        
    Returns:
        Grafo dirigido de NetworkX
    """
    G = nx.DiGraph()
    
    # Agregar nodos (entidades)
    for entity in entities:
        G.add_node(
            entity.name,
            entity_type=entity.entity_type.value,
            mythology=entity.mythology.value,
            aliases=entity.aliases,
            attributes=entity.attributes,
            description=entity.description
        )
    
    # Agregar aristas (relaciones)
    for relation in relations:
        G.add_edge(
            relation.source,
            relation.target,
            relation_type=relation.relation_type.value,
            confidence=relation.confidence,
            source_text=relation.source_text
        )
    
    return G


def get_node_color(entity_type: str) -> str:
    """
    Retorna el color para un tipo de entidad.
    
    Args:
        entity_type: Tipo de entidad
        
    Returns:
        Color en formato hex
    """
    colors = {
        "DEITY": "#FFD700",      # Gold
        "HERO": "#4169E1",       # Royal Blue
        "CREATURE": "#DC143C",   # Crimson
        "LOCATION": "#228B22",   # Forest Green
        "ARTIFACT": "#9932CC",   # Dark Orchid
        "EVENT": "#FF8C00",      # Dark Orange
        "TITAN": "#8B4513",      # Saddle Brown
        "MORTAL": "#808080"      # Gray
    }
    return colors.get(entity_type, "#CCCCCC")


def get_edge_color(relation_type: str) -> str:
    """
    Retorna el color para un tipo de relación.
    
    Args:
        relation_type: Tipo de relación
        
    Returns:
        Color en formato hex
    """
    colors = {
        "FATHER_OF": "#4682B4",
        "MOTHER_OF": "#FF69B4",
        "CHILD_OF": "#87CEEB",
        "SIBLING_OF": "#98FB98",
        "SPOUSE_OF": "#FF1493",
        "RIVAL_OF": "#FF4500",
        "ALLY_OF": "#32CD32",
        "KILLED": "#8B0000",
        "KILLED_BY": "#8B0000",
        "RULES": "#FFD700",
        "LIVES_IN": "#228B22",
        "OWNS": "#9932CC"
    }
    return colors.get(relation_type, "#888888")


def calculate_graph_metrics(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Calcula métricas del grafo.
    
    Args:
        G: Grafo de NetworkX
        
    Returns:
        Diccionario con métricas
    """
    metrics = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": nx.density(G),
        "is_connected": nx.is_weakly_connected(G) if G.number_of_nodes() > 0 else False
    }
    
    if G.number_of_nodes() > 0:
        # Centralidad
        in_degree = dict(G.in_degree())
        out_degree = dict(G.out_degree())
        
        metrics["most_connections_in"] = max(in_degree, key=in_degree.get) if in_degree else None
        metrics["most_connections_out"] = max(out_degree, key=out_degree.get) if out_degree else None
        
        # Componentes
        if G.number_of_edges() > 0:
            metrics["num_components"] = nx.number_weakly_connected_components(G)
    
    return metrics


# =============================================================================
# FUNCIONES DE ENTRADA/SALIDA
# =============================================================================

def save_graph_to_json(
    entities: List[MythEntity],
    relations: List[MythRelation],
    filepath: str
) -> None:
    """
    Guarda el grafo en formato JSON.
    
    Args:
        entities: Lista de entidades
        relations: Lista de relaciones
        filepath: Ruta del archivo
    """
    data = {
        "entities": [e.to_dict() for e in entities],
        "relations": [r.to_dict() for r in relations]
    }
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_graph_from_json(filepath: str) -> Tuple[List[MythEntity], List[MythRelation]]:
    """
    Carga el grafo desde formato JSON.
    
    Args:
        filepath: Ruta del archivo
        
    Returns:
        Tupla de (entidades, relaciones)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    entities = [MythEntity.from_dict(e) for e in data.get("entities", [])]
    relations = [MythRelation.from_dict(r) for r in data.get("relations", [])]
    
    return entities, relations


def export_to_cypher(
    entities: List[MythEntity],
    relations: List[MythRelation],
    filepath: str
) -> None:
    """
    Exporta el grafo a comandos Cypher para Neo4j.
    
    Args:
        entities: Lista de entidades
        relations: Lista de relaciones
        filepath: Ruta del archivo
    """
    lines = ["// NLP Mythology Knowledge Graph - Cypher Export\n"]
    
    # Crear nodos
    lines.append("// === NODES ===\n")
    for entity in entities:
        props = {
            "name": entity.name,
            "mythology": entity.mythology.value,
            "description": entity.description
        }
        props.update(entity.attributes)
        props_str = ", ".join([f'{k}: "{v}"' for k, v in props.items() if v])
        lines.append(f'CREATE (:{entity.entity_type.value} {{{props_str}}});')
    
    # Crear relaciones
    lines.append("\n// === RELATIONSHIPS ===\n")
    for rel in relations:
        lines.append(
            f'MATCH (a {{name: "{rel.source}"}}), (b {{name: "{rel.target}"}}) '
            f'CREATE (a)-[:{rel.relation_type.value} {{confidence: {rel.confidence}}}]->(b);'
        )
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def get_entity_by_name(
    name: str,
    entities: List[MythEntity],
    include_aliases: bool = True
) -> Optional[MythEntity]:
    """
    Busca una entidad por nombre o alias.
    
    Args:
        name: Nombre a buscar
        entities: Lista de entidades
        include_aliases: Incluir búsqueda por aliases
        
    Returns:
        Entidad encontrada o None
    """
    name_norm = normalize_entity_name(name)
    
    for entity in entities:
        if normalize_entity_name(entity.name) == name_norm:
            return entity
        if include_aliases:
            for alias in entity.aliases:
                if normalize_entity_name(alias) == name_norm:
                    return entity
    
    return None


def identify_entity_type(
    name: str,
    context: str = ""
) -> Optional[EntityType]:
    """
    Intenta identificar el tipo de una entidad.
    
    Args:
        name: Nombre de la entidad
        context: Contexto textual
        
    Returns:
        Tipo de entidad o None
    """
    name_norm = normalize_entity_name(name)
    
    # Buscar en entidades conocidas
    for myth, entities_by_type in KNOWN_ENTITIES.items():
        for entity_type, entity_list in entities_by_type.items():
            if name_norm in entity_list:
                return entity_type
    
    # Usar contexto para inferir
    context_lower = context.lower()
    
    type_keywords = {
        EntityType.DEITY: ["god", "goddess", "deity", "divine"],
        EntityType.HERO: ["hero", "warrior", "champion", "prince"],
        EntityType.CREATURE: ["monster", "beast", "creature", "serpent"],
        EntityType.LOCATION: ["mountain", "river", "city", "realm", "land"],
        EntityType.ARTIFACT: ["sword", "shield", "weapon", "staff", "ring"]
    }
    
    for entity_type, keywords in type_keywords.items():
        for keyword in keywords:
            if keyword in context_lower:
                return entity_type
    
    return None


def create_sample_data() -> Tuple[List[MythEntity], List[MythRelation]]:
    """
    Crea datos de ejemplo para testing.
    
    Returns:
        Tupla de (entidades, relaciones) de ejemplo
    """
    entities = [
        MythEntity(
            name="Zeus",
            entity_type=EntityType.DEITY,
            mythology=Mythology.GREEK,
            aliases=["Jupiter", "King of Gods"],
            attributes={"domain": "Sky, Thunder", "symbol": "Thunderbolt"},
            description="King of the Olympian gods"
        ),
        MythEntity(
            name="Poseidon",
            entity_type=EntityType.DEITY,
            mythology=Mythology.GREEK,
            aliases=["Neptune"],
            attributes={"domain": "Sea", "symbol": "Trident"},
            description="God of the sea"
        ),
        MythEntity(
            name="Hades",
            entity_type=EntityType.DEITY,
            mythology=Mythology.GREEK,
            aliases=["Pluto"],
            attributes={"domain": "Underworld", "symbol": "Helm of Darkness"},
            description="God of the underworld"
        ),
        MythEntity(
            name="Heracles",
            entity_type=EntityType.HERO,
            mythology=Mythology.GREEK,
            aliases=["Hercules"],
            attributes={"famous_for": "Twelve Labors"},
            description="Greatest Greek hero"
        ),
        MythEntity(
            name="Hydra",
            entity_type=EntityType.CREATURE,
            mythology=Mythology.GREEK,
            attributes={"heads": 9, "location": "Lerna"},
            description="Multi-headed serpent"
        ),
        MythEntity(
            name="Olympus",
            entity_type=EntityType.LOCATION,
            mythology=Mythology.GREEK,
            description="Home of the Olympian gods"
        )
    ]
    
    relations = [
        MythRelation(
            source="Zeus",
            target="Poseidon",
            relation_type=RelationType.SIBLING_OF,
            confidence=1.0,
            source_text="Zeus and Poseidon were brothers"
        ),
        MythRelation(
            source="Zeus",
            target="Hades",
            relation_type=RelationType.SIBLING_OF,
            confidence=1.0,
            source_text="Zeus and Hades were brothers"
        ),
        MythRelation(
            source="Zeus",
            target="Heracles",
            relation_type=RelationType.FATHER_OF,
            confidence=1.0,
            source_text="Zeus was the father of Heracles"
        ),
        MythRelation(
            source="Heracles",
            target="Hydra",
            relation_type=RelationType.KILLED,
            confidence=1.0,
            source_text="Heracles killed the Hydra"
        ),
        MythRelation(
            source="Zeus",
            target="Olympus",
            relation_type=RelationType.RULES,
            confidence=1.0,
            source_text="Zeus ruled from Olympus"
        )
    ]
    
    return entities, relations


if __name__ == "__main__":
    # Test básico
    print("=== NLP Mythology Knowledge Graph - Utils Test ===\n")
    
    # Crear datos de ejemplo
    entities, relations = create_sample_data()
    
    print(f"Entidades creadas: {len(entities)}")
    for e in entities:
        print(f"  - {e.name} ({e.entity_type.value})")
    
    print(f"\nRelaciones creadas: {len(relations)}")
    for r in relations:
        print(f"  - {r.source} --[{r.relation_type.value}]--> {r.target}")
    
    # Construir grafo
    G = build_networkx_graph(entities, relations)
    metrics = calculate_graph_metrics(G)
    
    print(f"\nMétricas del grafo:")
    for k, v in metrics.items():
        print(f"  - {k}: {v}")
    
    print("\n✓ Utils funcionando correctamente!")
