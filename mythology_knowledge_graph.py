#!/usr/bin/env python3
"""
NLP Mythology Knowledge Graph - Main Script
============================================

Sistema de procesamiento de lenguaje natural para construir un grafo
de conocimiento mitológico a partir de textos.

Author: César Adrián Delgado Díaz
Portfolio: https://tu-portfolio.com
LinkedIn: https://www.linkedin.com/in/cesar-delgado-diaz
GitHub: https://github.com/cesar530
License: MIT
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Imports del proyecto
try:
    from utils import (
        MythEntity, MythRelation, EntityType, RelationType, Mythology,
        build_networkx_graph, calculate_graph_metrics, get_node_color,
        get_edge_color, save_graph_to_json, load_graph_from_json,
        export_to_cypher, extract_relations_with_patterns, clean_text,
        detect_mythology, normalize_entity_name, identify_entity_type,
        create_sample_data, KNOWN_ENTITIES
    )
except ImportError:
    # Si se ejecuta como módulo
    from .utils import (
        MythEntity, MythRelation, EntityType, RelationType, Mythology,
        build_networkx_graph, calculate_graph_metrics, get_node_color,
        get_edge_color, save_graph_to_json, load_graph_from_json,
        export_to_cypher, extract_relations_with_patterns, clean_text,
        detect_mythology, normalize_entity_name, identify_entity_type,
        create_sample_data, KNOWN_ENTITIES
    )


# =============================================================================
# CLASE PRINCIPAL: MythologyKnowledgeGraph
# =============================================================================

class MythologyKnowledgeGraph:
    """
    Sistema principal para construir grafos de conocimiento mitológico.
    
    Integra NLP con spaCy, extracción de relaciones con HuggingFace,
    y almacenamiento/visualización con NetworkX y Neo4j.
    """
    
    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        use_transformers: bool = False,
        neo4j_config: Optional[Dict] = None
    ):
        """
        Inicializa el sistema.
        
        Args:
            spacy_model: Modelo de spaCy a usar
            use_transformers: Si usar modelos transformer para RE
            neo4j_config: Configuración de Neo4j (opcional)
        """
        self.spacy_model_name = spacy_model
        self.use_transformers = use_transformers
        self.neo4j_config = neo4j_config
        
        # Almacenamiento interno
        self.entities: List[MythEntity] = []
        self.relations: List[MythRelation] = []
        self.graph = None
        
        # Componentes NLP (cargados bajo demanda)
        self._nlp = None
        self._ner_pipeline = None
        self._re_pipeline = None
        self._neo4j_driver = None
        
        logger.info("MythologyKnowledgeGraph inicializado")
    
    # =========================================================================
    # CARGA DE MODELOS
    # =========================================================================
    
    def load_spacy_model(self) -> None:
        """Carga el modelo de spaCy."""
        try:
            import spacy
            logger.info(f"Cargando modelo spaCy: {self.spacy_model_name}")
            self._nlp = spacy.load(self.spacy_model_name)
            logger.info("Modelo spaCy cargado exitosamente")
        except OSError:
            logger.warning(f"Modelo {self.spacy_model_name} no encontrado. Descargando...")
            os.system(f"python -m spacy download {self.spacy_model_name}")
            import spacy
            self._nlp = spacy.load(self.spacy_model_name)
    
    def load_transformer_models(self) -> None:
        """Carga modelos transformer de HuggingFace."""
        if not self.use_transformers:
            return
        
        try:
            from transformers import pipeline
            
            logger.info("Cargando modelos transformer...")
            
            # Pipeline para NER
            self._ner_pipeline = pipeline(
                "ner",
                model="dslim/bert-base-NER",
                aggregation_strategy="simple"
            )
            
            # Pipeline para extracción de relaciones (zero-shot)
            self._re_pipeline = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
            
            logger.info("Modelos transformer cargados exitosamente")
            
        except Exception as e:
            logger.error(f"Error cargando transformers: {e}")
            self.use_transformers = False
    
    def connect_neo4j(self) -> bool:
        """
        Conecta a Neo4j si está configurado.
        
        Returns:
            True si la conexión fue exitosa
        """
        if not self.neo4j_config:
            return False
        
        try:
            from neo4j import GraphDatabase
            
            uri = self.neo4j_config.get("uri", "bolt://localhost:7687")
            user = self.neo4j_config.get("user", "neo4j")
            password = self.neo4j_config.get("password", "")
            
            self._neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
            
            # Verificar conexión
            with self._neo4j_driver.session() as session:
                session.run("RETURN 1")
            
            logger.info("Conexión a Neo4j establecida")
            return True
            
        except Exception as e:
            logger.error(f"Error conectando a Neo4j: {e}")
            self._neo4j_driver = None
            return False
    
    # =========================================================================
    # EXTRACCIÓN DE ENTIDADES (NER)
    # =========================================================================
    
    def extract_entities_spacy(self, text: str) -> List[MythEntity]:
        """
        Extrae entidades usando spaCy.
        
        Args:
            text: Texto a procesar
            
        Returns:
            Lista de entidades extraídas
        """
        if self._nlp is None:
            self.load_spacy_model()
        
        doc = self._nlp(text)
        entities = []
        seen_names = set()
        
        # Procesar entidades del modelo
        for ent in doc.ents:
            name = normalize_entity_name(ent.text)
            
            if name and name not in seen_names:
                seen_names.add(name)
                
                # Determinar tipo de entidad
                entity_type = self._map_spacy_label_to_type(ent.label_, name, text)
                
                if entity_type:
                    mythology = detect_mythology(text, [name])
                    
                    entities.append(MythEntity(
                        name=name,
                        entity_type=entity_type,
                        mythology=mythology,
                        description=f"Extracted from: '{ent.sent.text[:100]}...'"
                    ))
        
        # Buscar entidades mitológicas conocidas
        text_lower = text.lower()
        for myth, entities_by_type in KNOWN_ENTITIES.items():
            for entity_type, entity_list in entities_by_type.items():
                for known_entity in entity_list:
                    if known_entity.lower() in text_lower and known_entity not in seen_names:
                        seen_names.add(known_entity)
                        entities.append(MythEntity(
                            name=known_entity,
                            entity_type=entity_type,
                            mythology=myth
                        ))
        
        return entities
    
    def extract_entities_transformers(self, text: str) -> List[MythEntity]:
        """
        Extrae entidades usando transformers de HuggingFace.
        
        Args:
            text: Texto a procesar
            
        Returns:
            Lista de entidades extraídas
        """
        if self._ner_pipeline is None:
            self.load_transformer_models()
        
        if self._ner_pipeline is None:
            logger.warning("Pipeline de NER no disponible")
            return []
        
        entities = []
        seen_names = set()
        
        # Procesar con transformer
        results = self._ner_pipeline(text)
        
        for result in results:
            name = normalize_entity_name(result['word'])
            
            if name and name not in seen_names and len(name) > 2:
                seen_names.add(name)
                
                # Mapear tipo de entidad
                entity_type = identify_entity_type(name, text)
                
                if entity_type:
                    mythology = detect_mythology(text, [name])
                    
                    entities.append(MythEntity(
                        name=name,
                        entity_type=entity_type,
                        mythology=mythology,
                        attributes={"confidence": result.get('score', 0.0)}
                    ))
        
        return entities
    
    def _map_spacy_label_to_type(
        self,
        label: str,
        name: str,
        context: str
    ) -> Optional[EntityType]:
        """Mapea etiquetas de spaCy a tipos de entidad mitológica."""
        
        # Primero verificar si es una entidad conocida
        known_type = identify_entity_type(name, context)
        if known_type:
            return known_type
        
        # Mapeo basado en etiquetas de spaCy
        label_mapping = {
            "PERSON": EntityType.HERO,
            "GPE": EntityType.LOCATION,
            "LOC": EntityType.LOCATION,
            "ORG": EntityType.DEITY,  # Panteones
            "PRODUCT": EntityType.ARTIFACT,
            "EVENT": EntityType.EVENT
        }
        
        return label_mapping.get(label)
    
    # =========================================================================
    # EXTRACCIÓN DE RELACIONES
    # =========================================================================
    
    def extract_relations(
        self,
        text: str,
        entities: List[MythEntity]
    ) -> List[MythRelation]:
        """
        Extrae relaciones entre entidades.
        
        Args:
            text: Texto fuente
            entities: Entidades identificadas
            
        Returns:
            Lista de relaciones
        """
        entity_names = {e.name for e in entities}
        
        # Extraer con patrones
        relations = extract_relations_with_patterns(text, entity_names)
        
        # Si hay transformers, enriquecer con zero-shot
        if self.use_transformers and self._re_pipeline:
            transformer_relations = self._extract_relations_transformers(
                text, entities
            )
            relations.extend(transformer_relations)
        
        # Eliminar duplicados
        seen = set()
        unique_relations = []
        for rel in relations:
            key = (rel.source, rel.target, rel.relation_type.value)
            if key not in seen:
                seen.add(key)
                unique_relations.append(rel)
        
        return unique_relations
    
    def _extract_relations_transformers(
        self,
        text: str,
        entities: List[MythEntity]
    ) -> List[MythRelation]:
        """Extrae relaciones usando zero-shot classification."""
        relations = []
        
        # Definir tipos de relaciones a buscar
        relation_labels = [
            "father of", "mother of", "child of", "sibling of",
            "married to", "killed", "ally of", "rival of"
        ]
        
        # Procesar pares de entidades cercanas en el texto
        entity_names = [e.name for e in entities]
        
        for i, name1 in enumerate(entity_names):
            for name2 in entity_names[i+1:]:
                # Buscar oraciones que mencionen ambas entidades
                sentences = self._find_sentences_with_entities(text, name1, name2)
                
                for sentence in sentences[:3]:  # Limitar procesamiento
                    try:
                        result = self._re_pipeline(
                            f"{name1} and {name2}: {sentence}",
                            relation_labels
                        )
                        
                        if result['scores'][0] > 0.5:
                            rel_type = self._map_label_to_relation(result['labels'][0])
                            if rel_type:
                                relations.append(MythRelation(
                                    source=name1,
                                    target=name2,
                                    relation_type=rel_type,
                                    confidence=result['scores'][0],
                                    source_text=sentence
                                ))
                    except Exception as e:
                        logger.debug(f"Error en clasificación: {e}")
        
        return relations
    
    def _find_sentences_with_entities(
        self,
        text: str,
        entity1: str,
        entity2: str
    ) -> List[str]:
        """Encuentra oraciones que mencionan ambas entidades."""
        import re
        sentences = re.split(r'[.!?]+', text)
        
        matching = []
        for sent in sentences:
            sent_lower = sent.lower()
            if entity1.lower() in sent_lower and entity2.lower() in sent_lower:
                matching.append(sent.strip())
        
        return matching
    
    def _map_label_to_relation(self, label: str) -> Optional[RelationType]:
        """Mapea etiquetas de clasificación a tipos de relación."""
        mapping = {
            "father of": RelationType.FATHER_OF,
            "mother of": RelationType.MOTHER_OF,
            "child of": RelationType.CHILD_OF,
            "sibling of": RelationType.SIBLING_OF,
            "married to": RelationType.SPOUSE_OF,
            "killed": RelationType.KILLED,
            "ally of": RelationType.ALLY_OF,
            "rival of": RelationType.RIVAL_OF
        }
        return mapping.get(label.lower())
    
    # =========================================================================
    # PROCESAMIENTO PRINCIPAL
    # =========================================================================
    
    def process_text(
        self,
        text: str,
        use_transformers: bool = None
    ) -> Tuple[List[MythEntity], List[MythRelation]]:
        """
        Procesa un texto y extrae entidades y relaciones.
        
        Args:
            text: Texto a procesar
            use_transformers: Sobrescribir configuración de transformers
            
        Returns:
            Tupla de (entidades, relaciones)
        """
        if use_transformers is not None:
            self.use_transformers = use_transformers
        
        logger.info("Procesando texto...")
        text_clean = clean_text(text)
        
        # Extraer entidades
        logger.info("Extrayendo entidades...")
        if self.use_transformers:
            entities = self.extract_entities_transformers(text_clean)
            # Complementar con spaCy
            spacy_entities = self.extract_entities_spacy(text_clean)
            seen = {e.name for e in entities}
            for e in spacy_entities:
                if e.name not in seen:
                    entities.append(e)
        else:
            entities = self.extract_entities_spacy(text_clean)
        
        logger.info(f"  → {len(entities)} entidades encontradas")
        
        # Extraer relaciones
        logger.info("Extrayendo relaciones...")
        relations = self.extract_relations(text_clean, entities)
        logger.info(f"  → {len(relations)} relaciones encontradas")
        
        # Actualizar almacenamiento interno
        self.entities.extend(entities)
        self.relations.extend(relations)
        
        return entities, relations
    
    def process_file(self, filepath: str) -> Tuple[List[MythEntity], List[MythRelation]]:
        """
        Procesa un archivo de texto.
        
        Args:
            filepath: Ruta al archivo
            
        Returns:
            Tupla de (entidades, relaciones)
        """
        logger.info(f"Procesando archivo: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return self.process_text(text)
    
    def process_json_data(self, filepath: str) -> Tuple[List[MythEntity], List[MythRelation]]:
        """
        Procesa datos desde un archivo JSON.
        
        Args:
            filepath: Ruta al archivo JSON
            
        Returns:
            Tupla de (entidades, relaciones)
        """
        logger.info(f"Procesando JSON: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        all_entities = []
        all_relations = []
        
        # Procesar cada entrada
        texts = data if isinstance(data, list) else [data]
        
        for item in texts:
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict):
                text = item.get('text', item.get('content', str(item)))
            else:
                continue
            
            entities, relations = self.process_text(text)
            all_entities.extend(entities)
            all_relations.extend(relations)
        
        return all_entities, all_relations
    
    # =========================================================================
    # CONSTRUCCIÓN Y VISUALIZACIÓN DEL GRAFO
    # =========================================================================
    
    def build_graph(self) -> Any:
        """
        Construye el grafo de NetworkX.
        
        Returns:
            Grafo de NetworkX
        """
        logger.info("Construyendo grafo...")
        self.graph = build_networkx_graph(self.entities, self.relations)
        
        metrics = calculate_graph_metrics(self.graph)
        logger.info(f"Grafo construido: {metrics['num_nodes']} nodos, {metrics['num_edges']} aristas")
        
        return self.graph
    
    def visualize_graph(
        self,
        output_path: str = "mythology_graph.html",
        height: str = "800px",
        width: str = "100%"
    ) -> str:
        """
        Genera visualización interactiva del grafo.
        
        Args:
            output_path: Ruta del archivo HTML de salida
            height: Altura del grafo
            width: Ancho del grafo
            
        Returns:
            Ruta al archivo HTML generado
        """
        if self.graph is None:
            self.build_graph()
        
        try:
            from pyvis.network import Network
            
            logger.info("Generando visualización...")
            
            # Crear red de PyVis
            net = Network(height=height, width=width, directed=True)
            net.barnes_hut(gravity=-80000, central_gravity=0.3)
            
            # Agregar nodos
            for node in self.graph.nodes(data=True):
                name = node[0]
                attrs = node[1]
                entity_type = attrs.get('entity_type', 'UNKNOWN')
                
                net.add_node(
                    name,
                    label=name,
                    color=get_node_color(entity_type),
                    title=f"{name}\nType: {entity_type}\nMythology: {attrs.get('mythology', 'Unknown')}",
                    size=25 if entity_type == 'DEITY' else 20
                )
            
            # Agregar aristas
            for edge in self.graph.edges(data=True):
                source, target, attrs = edge
                rel_type = attrs.get('relation_type', 'UNKNOWN')
                
                net.add_edge(
                    source,
                    target,
                    title=rel_type,
                    color=get_edge_color(rel_type),
                    label=rel_type.replace('_', ' ').lower()
                )
            
            # Configurar opciones
            net.set_options("""
            var options = {
                "nodes": {
                    "font": {"size": 14}
                },
                "edges": {
                    "arrows": {"to": {"enabled": true}},
                    "font": {"size": 10, "align": "middle"},
                    "smooth": {"type": "continuous"}
                },
                "physics": {
                    "forceAtlas2Based": {
                        "gravitationalConstant": -50,
                        "springLength": 200
                    },
                    "solver": "forceAtlas2Based"
                }
            }
            """)
            
            # Guardar
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            net.save_graph(output_path)
            
            logger.info(f"Visualización guardada en: {output_path}")
            return output_path
            
        except ImportError:
            logger.error("PyVis no instalado. Instalar con: pip install pyvis")
            return ""
    
    # =========================================================================
    # PERSISTENCIA EN NEO4J
    # =========================================================================
    
    def save_to_neo4j(self) -> bool:
        """
        Guarda el grafo en Neo4j.
        
        Returns:
            True si fue exitoso
        """
        if self._neo4j_driver is None:
            if not self.connect_neo4j():
                logger.error("No hay conexión a Neo4j")
                return False
        
        try:
            with self._neo4j_driver.session() as session:
                # Limpiar datos existentes (opcional)
                # session.run("MATCH (n) DETACH DELETE n")
                
                # Crear nodos
                for entity in self.entities:
                    session.run(
                        f"""
                        MERGE (n:{entity.entity_type.value} {{name: $name}})
                        SET n.mythology = $mythology,
                            n.description = $description
                        """,
                        name=entity.name,
                        mythology=entity.mythology.value,
                        description=entity.description
                    )
                
                # Crear relaciones
                for rel in self.relations:
                    session.run(
                        f"""
                        MATCH (a {{name: $source}}), (b {{name: $target}})
                        MERGE (a)-[r:{rel.relation_type.value}]->(b)
                        SET r.confidence = $confidence,
                            r.source_text = $source_text
                        """,
                        source=rel.source,
                        target=rel.target,
                        confidence=rel.confidence,
                        source_text=rel.source_text
                    )
                
                logger.info("Datos guardados en Neo4j")
                return True
                
        except Exception as e:
            logger.error(f"Error guardando en Neo4j: {e}")
            return False
    
    # =========================================================================
    # EXPORTACIÓN
    # =========================================================================
    
    def save(self, output_dir: str, prefix: str = "mythology") -> Dict[str, str]:
        """
        Guarda todos los resultados.
        
        Args:
            output_dir: Directorio de salida
            prefix: Prefijo para archivos
            
        Returns:
            Diccionario con rutas de archivos generados
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        files = {}
        
        # JSON
        json_path = os.path.join(output_dir, f"{prefix}_graph.json")
        save_graph_to_json(self.entities, self.relations, json_path)
        files['json'] = json_path
        
        # Cypher
        cypher_path = os.path.join(output_dir, f"{prefix}_cypher.txt")
        export_to_cypher(self.entities, self.relations, cypher_path)
        files['cypher'] = cypher_path
        
        # HTML
        html_path = os.path.join(output_dir, f"{prefix}_visualization.html")
        self.visualize_graph(html_path)
        files['html'] = html_path
        
        logger.info(f"Archivos guardados en: {output_dir}")
        return files
    
    def get_summary(self) -> Dict:
        """
        Retorna un resumen del grafo.
        
        Returns:
            Diccionario con resumen
        """
        if self.graph is None:
            self.build_graph()
        
        # Contar por tipo
        entity_counts = {}
        for e in self.entities:
            t = e.entity_type.value
            entity_counts[t] = entity_counts.get(t, 0) + 1
        
        relation_counts = {}
        for r in self.relations:
            t = r.relation_type.value
            relation_counts[t] = relation_counts.get(t, 0) + 1
        
        mythology_counts = {}
        for e in self.entities:
            m = e.mythology.value
            mythology_counts[m] = mythology_counts.get(m, 0) + 1
        
        metrics = calculate_graph_metrics(self.graph)
        
        return {
            "total_entities": len(self.entities),
            "total_relations": len(self.relations),
            "entities_by_type": entity_counts,
            "relations_by_type": relation_counts,
            "entities_by_mythology": mythology_counts,
            "graph_metrics": metrics
        }


# =============================================================================
# FUNCIONES DE CLI
# =============================================================================

def main():
    """Función principal de CLI."""
    parser = argparse.ArgumentParser(
        description="NLP Mythology Knowledge Graph Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python mythology_knowledge_graph.py --input texto.txt --output outputs/
  python mythology_knowledge_graph.py --input data.json --mode json --output outputs/
  python mythology_knowledge_graph.py --demo --output outputs/
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        help='Archivo de entrada (texto o JSON)'
    )
    parser.add_argument(
        '--output', '-o',
        default='outputs/',
        help='Directorio de salida (default: outputs/)'
    )
    parser.add_argument(
        '--mode', '-m',
        choices=['text', 'json', 'ner', 'visualize'],
        default='text',
        help='Modo de operación'
    )
    parser.add_argument(
        '--spacy-model',
        default='en_core_web_sm',
        help='Modelo de spaCy a usar'
    )
    parser.add_argument(
        '--use-transformers',
        action='store_true',
        help='Usar modelos transformer de HuggingFace'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Ejecutar con datos de demostración'
    )
    parser.add_argument(
        '--neo4j-uri',
        help='URI de Neo4j'
    )
    parser.add_argument(
        '--neo4j-user',
        default='neo4j',
        help='Usuario de Neo4j'
    )
    parser.add_argument(
        '--neo4j-password',
        help='Password de Neo4j'
    )
    
    args = parser.parse_args()
    
    # Configurar Neo4j si se proporcionan credenciales
    neo4j_config = None
    if args.neo4j_uri:
        neo4j_config = {
            'uri': args.neo4j_uri,
            'user': args.neo4j_user,
            'password': args.neo4j_password or ''
        }
    
    # Crear instancia
    mkg = MythologyKnowledgeGraph(
        spacy_model=args.spacy_model,
        use_transformers=args.use_transformers,
        neo4j_config=neo4j_config
    )
    
    if args.demo:
        # Modo demostración
        print("\n" + "="*60)
        print("  NLP MYTHOLOGY KNOWLEDGE GRAPH - DEMO")
        print("="*60 + "\n")
        
        # Texto de demostración
        demo_text = """
        Zeus, the king of the Olympian gods, ruled from Mount Olympus. 
        He was the father of many heroes, including Heracles and Perseus.
        Zeus had two brothers: Poseidon, god of the sea, and Hades, ruler of the underworld.
        
        Heracles, son of Zeus and the mortal Alcmene, completed twelve labors.
        He killed the Hydra, a nine-headed serpent, and captured Cerberus from the underworld.
        
        Perseus, another son of Zeus, slew Medusa the Gorgon and used her head as a weapon.
        He later rescued Andromeda from a sea monster sent by Poseidon.
        
        The Titans, including Cronus and Rhea, were the parents of Zeus, Poseidon, and Hades.
        Zeus defeated Cronus and imprisoned the Titans in Tartarus.
        """
        
        entities, relations = mkg.process_text(demo_text)
        
        # Mostrar resultados
        print("ENTIDADES ENCONTRADAS:")
        print("-" * 40)
        for e in entities:
            print(f"  • {e.name} ({e.entity_type.value}) - {e.mythology.value}")
        
        print("\nRELACIONES ENCONTRADAS:")
        print("-" * 40)
        for r in relations:
            print(f"  • {r.source} --[{r.relation_type.value}]--> {r.target}")
        
        # Guardar resultados
        files = mkg.save(args.output, prefix="demo_mythology")
        
        print("\nARCHIVOS GENERADOS:")
        print("-" * 40)
        for name, path in files.items():
            print(f"  • {name}: {path}")
        
        # Resumen
        summary = mkg.get_summary()
        print("\nRESUMEN:")
        print("-" * 40)
        print(f"  Total entidades: {summary['total_entities']}")
        print(f"  Total relaciones: {summary['total_relations']}")
        print(f"  Nodos en grafo: {summary['graph_metrics']['num_nodes']}")
        print(f"  Aristas en grafo: {summary['graph_metrics']['num_edges']}")
        
    elif args.input:
        # Procesar archivo
        if args.mode == 'json':
            mkg.process_json_data(args.input)
        else:
            mkg.process_file(args.input)
        
        files = mkg.save(args.output)
        
        print(f"\nProcesamiento completado. Archivos en: {args.output}")
        
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
