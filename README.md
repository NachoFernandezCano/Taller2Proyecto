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

## 🎯 4. Banco de Oraciones Adversariales (`data/processed/`)

Para la evaluación cualitativa de los modelos frente a casos complejos y trampas sintácticas/morfológicas, se diseñó el banco de oraciones disponible tanto en formato JSON como en CSV (`data/processed/oraciones_adversariales.json` y `data/processed/banco_oraciones_adversariales.csv`):

| ID | Tipo de Trampa | Oración | Sentimiento Real | Justificación Teórica |
|:---|:---|:---|:---:|:---|
| 1 | **Negación directa / Inversión de polaridad** | *"No puedo decir que el rendimiento gráfico o el diseño de niveles sea decepcionante."* | **Positivo** | Contiene el término negativo *"decepcionante"*, pero precedido por doble negación (*"No puedo decir..."*). Modelos BoW sin n-gramas asignan peso negativo por presencia léxica. |
| 2 | **Inversión sintáctica / Cláusula adversativa** | *"Aunque al inicio prometía ser una obra maestra revolucionaria, el juego terminó siendo un desastre injugable lleno de microtransacciones."* | **Negativo** | La primera parte acumula léxico positivo (*"obra maestra"*, *"revolucionaria"*), pero la cláusula principal impone el sentido negativo real (*"desastre injugable"*). |
| 3 | **Afijo morfológico negativo** | *"Es una experiencia innegablemente imperfecta y con mecánicas desacertadas."* | **Negativo** | Emplea prefijos de negación (*"in-"*, *"des-"*). Modelos que no implementan lematización o análisis morfológico de subpalabras suelen fallar en reconocer la polaridad. |
