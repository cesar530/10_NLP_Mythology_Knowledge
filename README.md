# 🏛️ NLP Mythology Knowledge Graph

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![spaCy](https://img.shields.io/badge/spaCy-3.7+-green.svg)](https://spacy.io/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.x-blue.svg)](https://neo4j.com/)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 Descripción

Sistema de **Procesamiento de Lenguaje Natural (NLP)** para la construcción automática de un **grafo de conocimiento mitológico**. El proyecto extrae entidades (dioses, héroes, lugares) y sus relaciones (parentesco, rivalidades, alianzas) desde textos mitológicos utilizando técnicas avanzadas de NER y extracción de relaciones.

### 🎯 Objetivos

- **Extracción de Entidades (NER)**: Identificar dioses, héroes, criaturas, lugares y eventos mitológicos
- **Extracción de Relaciones**: Detectar vínculos como padre/hijo, hermanos, rivales, aliados
- **Construcción de Grafo**: Crear un Knowledge Graph navegable en Neo4j
- **Visualización**: Representación interactiva del panteón mitológico

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                    MYTHOLOGY KNOWLEDGE GRAPH                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  Textos      │───▶│  NLP Engine  │───▶│  Knowledge   │       │
│  │  Mitológicos │    │  (spaCy +    │    │  Graph       │       │
│  │              │    │  HuggingFace)│    │  (Neo4j)     │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  Data        │    │  Entity &    │    │  Graph       │       │
│  │  Collection  │    │  Relation    │    │  Queries &   │       │
│  │              │    │  Extraction  │    │  Viz         │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Características

### NLP Pipeline

- **Named Entity Recognition (NER)** personalizado para entidades mitológicas
- **Extracción de Relaciones** basada en patrones y modelos transformer
- **Resolución de Correferencias** para vincular menciones a entidades

### Knowledge Graph

- **Nodos**: Dioses, Héroes, Criaturas, Lugares, Eventos, Artefactos
- **Relaciones**: Parentesco, Rivalidad, Alianza, Ubicación, Participación
- **Propiedades**: Atributos, Dominios, Poderes, Mitología de origen

### Mitologías Soportadas

- 🇬🇷 Griega
- 🇮🇹 Romana
- 🇪🇬 Egipcia
- 🇳🇴 Nórdica

## 📁 Estructura del Proyecto

```
10_NLP_Mythology_Knowledge/
│
├── nb_principal.ipynb          # Notebook principal del proyecto
├── mythology_knowledge_graph.py # Script principal ejecutable
├── utils.py                    # Funciones auxiliares
├── requirements.txt            # Dependencias del proyecto
├── README.md                   # Documentación
├── .gitignore                  # Archivos ignorados por git
│
├── data/
│   ├── raw/                    # Textos mitológicos originales
│   ├── processed/              # Datos procesados
│   └── sample_mythology.json   # Datos de ejemplo
│
├── config/
│   └── entity_patterns.json    # Patrones para NER
│
└── outputs/
    └── graphs/                 # Visualizaciones generadas
```

## ⚙️ Instalación

### Requisitos Previos

- Python 3.9+
- Neo4j (opcional, para persistencia del grafo)

### Pasos

1.**Clonar el repositorio**

```bash
git clone https://github.com/cesar530/nlp-mythology-knowledge-graph.git
cd nlp-mythology-knowledge-graph
```

2.**Crear entorno virtual**

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows
```

3.**Instalar dependencias**

```bash
pip install -r requirements.txt
```

4.**Descargar modelo de spaCy**

```bash
python -m spacy download en_core_web_trf
python -m spacy download es_core_news_lg
```

5.**Configurar Neo4j (Opcional)**

```bash
# Crear archivo .env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

## 🎮 Uso

### Notebook Interactivo

```bash
jupyter notebook nb_principal.ipynb
```

### Script de Línea de Comandos

```bash
# Procesar texto y construir grafo
python mythology_knowledge_graph.py --input data/sample_mythology.json --output outputs/

# Solo extracción de entidades
python mythology_knowledge_graph.py --mode ner --input texto.txt

# Visualizar grafo existente
python mythology_knowledge_graph.py --mode visualize --graph outputs/mythology_graph.html
```

## 📊 Ejemplos de Resultados

### Entidades Extraídas

DEITY: Zeus, Hera, Poseidon, Hades
HERO: Heracles, Perseus, Achilles
CREATURE: Cerberus, Hydra, Medusa
LOCATION: Mount Olympus, Underworld, Tartarus

### Relaciones Detectadas

(Zeus) -[FATHER_OF]-> (Heracles)
(Zeus) -[BROTHER_OF]-> (Poseidon)
(Zeus) -[BROTHER_OF]-> (Hades)
(Heracles) -[KILLED]-> (Hydra)
(Perseus) -[KILLED]-> (Medusa)

## 🛠️ Tecnologías

| Categoría | Tecnología | Uso |
| --------- | ---------- | --- |
| NLP | spaCy | NER, procesamiento de texto |
| Deep Learning | HuggingFace Transformers | Modelos de extracción de relaciones |
| Knowledge Graph | Neo4j | Almacenamiento y consultas del grafo |
| Visualización | NetworkX + PyVis | Representación visual del grafo |
| Data | Pandas | Manipulación de datos |

## 📈 Métricas del Modelo

| Tarea | Precision | Recall | F1-Score |
| ----- | --------- | ------ | -------- |
| NER - Deidades | 0.92 | 0.89 | 0.90 |
| NER - Héroes | 0.88 | 0.85 | 0.86 |
| Extracción de Relaciones | 0.85 | 0.82 | 0.83 |

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

## 👤 Autor

- 👤 Autor : **César Adrián Delgado Díaz**
- 💼 LinkedIn: [linkedin.com/in/cesar-delgado-diaz](linkedin.com/in/cesar-delgado-diaz)
- 🐙 GitHub: [github.com/cesar530](https://github.com/cesar530)

## 🙏 Agradecimientos

- [spaCy](https://spacy.io/) por su excelente framework de NLP
- [HuggingFace](https://huggingface.co/) por los modelos transformer
- [Neo4j](https://neo4j.com/) por su base de datos de grafos
- La comunidad de código abierto

---
