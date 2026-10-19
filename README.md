# Pipeline de Análisis de Sentimiento con Reseñas de Metacritic

##  Fases del Pipeline

### 1. Web Scraping de Metacritic (`src/scraper_metacritic.py`)
- Extrae reseñas de usuarios de títulos balanceados (Elden Ring, Baldur's Gate 3, The Witcher 3, Cyberpunk 2077, Starfield, Warcraft III: Reforged, Diablo IV, Battlefield 2042).
- Aplica el criterio de etiquetado binario de la cátedra:
  - $\text{Score} \ge 7 \longrightarrow 1$ (Positivo)
  - $\text{Score} \le 5 \longrightarrow 0$ (Negativo)
  - $\text{Score} = 6 \longrightarrow$ Descartado (zona ambigua neutral).
- Persiste en `data/raw/metacritic_raw_reviews.csv`.

### 2. Preprocesamiento de Texto (`src/preprocessing.py`)
- Limpieza y decodificación HTML, remoción de URLs, filtrado por expresión regular.
- **Tratamiento estricto de Stopwords para Sentimiento:** Se preservan partículas de negación y conectores adversativos (`no`, `not`, `never`, `nor`, `sin`, `pero`, `although`, `however`, etc.) para asegurar que los modelos capturen la inversión de polaridad.
- Persiste en `data/processed/dataset_preprocesado.csv`.

### 3. Vectorización y Comparativa de Vocabularios (`src/vectorization.py`)
Calcula los tamaños de vocabulario ($|V|$) y las matrices resultantes en 5 esquemas:

| Enfoque de Vectorización | Rango de N-gramas / Dimensión | Tamaño de Vocabulario ($|V|$) | Dimensión de Matriz / Representación | Tipo de Representación |
| :--- | :--- | :--- | :--- | :--- |
| **BoW (CountVectorizer)** | $(1, 1)$ | **7,301** | `(557, 7301)` | Dispersa (Frecuencia léxica) |
| **TF-IDF Unigramas** | $(1, 1)$ | **7,301** | `(557, 7301)` | Dispersa (Ponderada TF-IDF) |
| **TF-IDF Bigramas (1, 2)** | $(1, 2)$ | **30,949** | `(557, 30949)` | Dispersa (Contexto local enriquecido) |
| **TF-IDF Bigramas (2, 2)** | $(2, 2)$ | **23,648** | `(557, 23648)` | Dispersa (Pares contiguos estrictos) |
| **Word2Vec (Embeddings)** | Vector dim = 100 | **7,325** | `(557, 100)` | Densa (Semántica distribuida, Mean Pooling) |
| **Hugging Face (BERT)** | 768 dim base | **30,522** | `Vocab: 30522 | Embs: (N, 768)` | Contextual profunda (WordPiece) |

---

## Oraciones Adversariales (`data/processed/oraciones_adversariales.json`)

1. **Negación directa / Inversión de polaridad:** *"No puedo decir que el rendimiento gráfico o el diseño de niveles sea decepcionante."* $\rightarrow$ Positivo real.
2. **Inversión sintáctica / Cláusula adversativa:** *"Aunque al inicio prometía ser una obra maestra revolucionaria, el juego terminó siendo un desastre injugable lleno de microtransacciones."* $\rightarrow$ Negativo real.
3. **Afijo morfológico negativo:** *"Es una experiencia innegablemente imperfecta y con mecánicas desacertadas."* $\rightarrow$ Negativo real.
