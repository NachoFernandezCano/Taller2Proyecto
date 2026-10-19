"""Objetivo: Recolectar entre 300 y 600 reseñas de usuarios reales con etiquetas binarias."""

import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional
import requests
from bs4 import BeautifulSoup
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_PATH = Path("data/raw/metacritic_raw_reviews.csv")

TARGET_GAMES = [
    # Títulos altamente valorados (fuente de reseñas positivas)
    {"slug": "elden-ring", "platform": "playstation-5"},
    {"slug": "elden-ring", "platform": "pc"},
    {"slug": "baldurs-gate-3", "platform": "pc"},
    {"slug": "baldurs-gate-3", "platform": "playstation-5"},
    {"slug": "the-witcher-3-wild-hunt", "platform": "pc"},
    {"slug": "red-dead-redemption-2", "platform": "pc"},
    {"slug": "god-of-war-ragnarok", "platform": "playstation-5"},
    # Títulos polarizantes / fuertemente criticados (fuente de reseñas negativas)
    {"slug": "cyberpunk-2077", "platform": "playstation-4"},
    {"slug": "the-last-of-us-part-ii", "platform": "playstation-4"},
    {"slug": "starfield", "platform": "pc"},
    {"slug": "overwatch-2", "platform": "pc"},
    {"slug": "warcraft-iii-reforged", "platform": "pc"},
    {"slug": "diablo-iv", "platform": "pc"},
    {"slug": "battlefield-2042", "platform": "pc"},
    {"slug": "redfall", "platform": "pc"},
    {"slug": "suicide-squad-kill-the-justice-league", "platform": "pc"},
    {"slug": "no-mans-sky", "platform": "playstation-4"},
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
    "Referer": "https://www.metacritic.com/",
}


def parse_score(score_str: str) -> Optional[int]:
    match = re.search(r"\b(10|[0-9])\b", score_str)
    if match:
        return int(match.group(1))
    return None


def extract_reviews_from_soup(soup: BeautifulSoup, game_name: str) -> List[Dict]:
    extracted = []

    # 1. Selector moderno con tarjetas data-testid o contenedor Nuxt
    cards = soup.select('div[data-testid="review-card"]')
    if not cards:
        container = soup.select_one(".c-reviews-container")
        if container:
            cards = container.find_all(recursive=False)

    if not cards:
        # Fallback a clases legadas
        cards = soup.select("div.c-siteReview, div.review_content, div.c-siteReviewUser")

    for card in cards:
        # Extraer puntaje
        score_val = None
        score_el = card.select_one(".c-siteReviewScore, div[class*='c-siteReviewScore'], .metascore_w")
        if score_el:
            score_text = score_el.get_text(strip=True)
            score_val = parse_score(score_text)

        if score_val is None:
            for el in card.find_all(attrs={"aria-label": True}):
                val = parse_score(el["aria-label"])
                if val is not None:
                    score_val = val
                    break

        if score_val is None:
            continue

        # Criterio de etiquetado:
        # Score >= 7 -> 1 (Positivo)
        # Score <= 5 -> 0 (Negativo)
        # Score == 6 -> Descartar (zona neutra/ambigua)
        if score_val >= 7:
            label = 1
        elif score_val <= 5:
            label = 0
        else:
            continue

        quote_el = card.select_one(
            'div[data-testid="review-card-quote-block"], '
            'div.c-siteReview_quote, span.c-siteReview_quote, '
            '.review_body, div[class*="review-quote"]'
        )
        if quote_el:
            text = quote_el.get_text(separator=" ", strip=True)
        else:
            paragraphs = card.find_all("p")
            if paragraphs:
                text = " ".join(p.get_text(strip=True) for p in paragraphs)
            else:
                text = card.get_text(separator=" ", strip=True)

        text = re.sub(
            r"\b(Collapse review|Read full review|Spoiler alert)\b",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = text.strip()

        if len(text) >= 20:
            extracted.append({
                "game": game_name,
                "review_text": text,
                "score": score_val,
                "label": label,
            })

    return extracted


def scrape_metacritic(target_count: int = 550) -> pd.DataFrame:
    seen_texts = set()
    records: List[Dict] = []
    
    if RAW_DATA_PATH.exists():
        try:
            df_old = pd.read_csv(RAW_DATA_PATH)
            for _, row in df_old.iterrows():
                seen_texts.add(row["review_text"])
                records.append(row.to_dict())
            logger.info(f"Cargadas {len(records)} reseñas preexistentes.")
        except Exception as e:
            logger.warning(f"No se pudo cargar el archivo preexistente: {e}")

    session = requests.Session()
    session.headers.update(HEADERS)

    logger.info(f"Iniciando Web Scraping de Metacritic. Meta: {target_count} reseñas únicas.")

    for item in TARGET_GAMES:
        if len(records) >= target_count:
            logger.info(f"Meta alcanzada con {len(records)} reseñas. Finalizando scraping.")
            break

        slug = item["slug"]
        platform = item["platform"]
        url = f"https://www.metacritic.com/game/{slug}/user-reviews/?platform={platform}"
        logger.info(f"Consultando {slug} ({platform})...")

        try:
            response = session.get(url, timeout=15)
            if response.status_code != 200:
                logger.warning(f"  HTTP {response.status_code} en {url}")
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            batch = extract_reviews_from_soup(soup, slug)
            new_added = 0
            for r in batch:
                txt = r["review_text"]
                if txt not in seen_texts:
                    seen_texts.add(txt)
                    records.append(r)
                    new_added += 1

            logger.info(f"  -> {new_added} reseñas nuevas agregadas. Total acumulado: {len(records)}")
            time.sleep(1.2)  # Pausa de cortesía para no saturar el servidor
        except Exception as e:
            logger.error(f"  Error procesando {slug}: {e}")

    df = pd.DataFrame(records)
    logger.info(f"Scraping completado. Total de reseñas únicas: {len(df)}")
    if not df.empty:
        logger.info(f"Distribución de etiquetas:\n{df['label'].value_counts()}")

    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False, encoding="utf-8")
    logger.info(f"Archivo guardado exitosamente en: {RAW_DATA_PATH}")
    return df


if __name__ == "__main__":
    scrape_metacritic(target_count=550)
