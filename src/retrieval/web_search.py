from __future__ import annotations

import logging
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup
from serpapi import GoogleSearch

from config import settings

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}


@dataclass
class SearchResult:
    title: str
    link: str
    snippet: str
    content: str = ""


def google_search(query: str, num_results: int | None = None) -> list[SearchResult]:
    num_results = num_results or settings.search_results
    logger.info("SerpAPI: szukam '%s' (top %d)", query, num_results)

    params = {
        "engine": "google",
        "q": query,
        "num": num_results,
        "hl": "en",
        "api_key": settings.serpapi_api_key,
    }
    search = GoogleSearch(params)
    data = search.get_dict()

    results: list[SearchResult] = []
    for item in data.get("organic_results", [])[:num_results]:
        results.append(
            SearchResult(
                title=item.get("title", ""),
                link=item.get("link", ""),
                snippet=item.get("snippet", ""),
            )
        )
    logger.info("SerpAPI zwróciło %d wyników", len(results))
    return results


def fetch_page_text(url: str) -> str:
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=settings.request_timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Nie udało się pobrać %s: %s", url, exc)
        return ""

    soup = BeautifulSoup(resp.text, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all(["p", "li", "h2", "h3"])]
    text = "\n".join(par for par in paragraphs if len(par) > 40)
    return text


def search_and_scrape(query: str) -> list[SearchResult]:
    results = google_search(query)
    scraped = 0
    for r in results:
        if scraped >= settings.max_pages_to_scrape:
            break
        r.content = fetch_page_text(r.link)
        if r.content:
            scraped += 1
    return results
