"""Objetivo: Limpieza, normalización y preservación de términos de polaridad y negación."""

import html
import logging
import re
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_PATH = Path("data/raw/metacritic_raw_reviews.csv")
PROCESSED_DATA_PATH = Path("data/processed/dataset_preprocesado.csv")

# Palabras de negación y conectores adversativos a preservar estrictamente
PRESERVED_SENTIMENT_TOKENS = {
    # Inglés
    "no", "not", "nor", "never", "none", "neither", "nobody", "nothing",
    "nowhere", "hardly", "scarcely", "barely", "without", "but", "however",
    "although", "though", "yet", "except", "despite", "instead", "against",
    # Español
    "no", "ni", "nunca", "jamás", "jamas", "tampoco", "sin", "pero", "sino",
    "aunque", "mas", "nada", "nadie", "ningun", "ninguno", "ninguna", "contra"
}

# Stopwords estándar frecuentes en reseñas (excluyendo negaciones y conectores de polaridad)
BASE_STOPWORDS = {
    # Inglés común
    "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with",
    "by", "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "this", "that", "these", "those", "it", "its",
    "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "their", "as", "from", "into", "about", "which",
    "who", "whom", "will", "would", "shall", "should", "can", "could", "there",
    # Español común
    "el", "la", "los", "las", "un", "una", "unos", "unas", "y", "o", "en", "de",
    "a", "por", "para", "con", "es", "son", "era", "eran", "fue", "fueron", "ser",
    "estar", "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
    "su", "sus", "mi", "mis", "tu", "tus", "que", "se", "lo", "al", "del"
}

FILTERED_STOPWORDS = BASE_STOPWORDS - PRESERVED_SENTIMENT_TOKENS


def clean_review_text(text: str, remove_stopwords: bool = True) -> str:
    """
    Aplica el pipeline de limpieza:
    1. Decodificación de entidades HTML (&quot;, &#39;, &amp;, etc.).
    2. Remoción de URLs y etiquetas HTML.
    3. Conversión a minúsculas.
    4. Conservación exclusiva de caracteres alfabéticos y acentos ([a-záéíóúñü\\s]).
    5. Filtrado de stopwords no críticas (manteniendo negaciones y adversativas).
    6. Normalización de espacios en blanco.
    """
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()

    text = re.sub(r"n['’]t\b", " not", text)
    text = re.sub(r"['’]s\b", " ", text)
    text = re.sub(r"['’]ve\b", " have", text)
    text = re.sub(r"['’]ll\b", " will", text)
    text = re.sub(r"['’]re\b", " are", text)
    text = re.sub(r"['’]d\b", " would", text)

    text = re.sub(r"[^a-záéíóúñü\s]", " ", text)

    tokens = text.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in FILTERED_STOPWORDS or t in PRESERVED_SENTIMENT_TOKENS]

    clean = " ".join(tokens)
    return clean


def preprocess_corpus(
    input_path: Path = RAW_DATA_PATH,
    output_path: Path = PROCESSED_DATA_PATH,
) -> pd.DataFrame:
    """Carga, limpia y persiste el dataset preprocesado."""
    logger.info(f"Cargando dataset crudo desde: {input_path}")
    df = pd.read_csv(input_path)
    initial_len = len(df)
    logger.info(f"Registros iniciales: {initial_len}")

    # Descartar nulos y duplicados
    df = df.dropna(subset=["review_text", "score", "label"])
    df = df.drop_duplicates(subset=["review_text"]).reset_index(drop=True)
    logger.info(f"Registros tras eliminar nulos y duplicados: {len(df)}")

    # Aplicar función de limpieza
    logger.info("Aplicando pipeline de normalización y limpieza léxica...")
    df["clean_text"] = df["review_text"].apply(clean_review_text)

    # Descartar textos que hayan quedado vacíos o excesivamente cortos
    df = df[df["clean_text"].str.split().str.len() >= 3].reset_index(drop=True)
    logger.info(f"Registros finales válidos: {len(df)}")

    # Seleccionar columnas especificadas
    output_df = df[["review_text", "clean_text", "score", "label"]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Dataset preprocesado guardado en: {output_path}")

    # Reporte de muestra
    logger.info("Muestra de transformación (original -> limpio):")
    for i in range(min(2, len(output_df))):
        logger.info(f"  [Original {i+1}]: {output_df.loc[i, 'review_text'][:90]}...")
        logger.info(f"  [Limpio {i+1}]:   {output_df.loc[i, 'clean_text'][:90]}...")

    logger.info(f"Balance final de clases:\n{output_df['label'].value_counts()}")
    return output_df


if __name__ == "__main__":
    preprocess_corpus()
