import asyncio
import aiohttp
import feedparser
import time
import sys
from typing import List, Dict

# --- ВИДАЛЕНО SettingsLoader ---

# Виправлення для asyncio на Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class NewsFetcher:
    """
    Асинхронний модуль для збору новин.
    Тепер отримує свою конфігурацію ззовні, а не читає з файлу.
    """

    def __init__(self, config: Dict):
        """
        Ініціалізує фетчер, використовуючи передану конфігурацію.

        :param config: Словник з налаштуваннями, що містить ключ 'sources'.
        """
        # Фільтруємо лише активні джерела з переданої конфігурації
        self.sources = [src for src in config.get("sources", []) if src.get("enabled", False)]
        print(f"📰 NewsFetcher ініціалізовано. Активних джерел для обробки: {len(self.sources)}")

    async def _fetch_rss(self, session: aiohttp.ClientSession, source_config: Dict) -> List[Dict]:
        """
        Асинхронно отримує та парсить дані з одного RSS-каналу.
        """
        url = source_config.get("url")
        source_name = source_config.get("name")
        articles = []
        
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            async with session.get(url, timeout=10, headers=headers) as response:
                response.raise_for_status()
                feed_text = await response.text()
                feed = feedparser.parse(feed_text)
                
                for entry in feed.entries:
                    articles.append({
                        "source": source_name,
                        "headline": entry.title,
                        "url": entry.link,
                        "published": entry.get("published_parsed", time.gmtime()),
                        "summary": entry.get("summary", "")
                    })
            print(f"✅ Успішно отримано {len(articles)} новин з {source_name}")
        except Exception as e:
            print(f"❌ Помилка при обробці {source_name}: {e}")
            
        return articles

    async def _fetch_api(self, session: aiohttp.ClientSession, source_config: Dict) -> List[Dict]:
        """Заготовка для отримання новин з REST API."""
        source_name = source_config.get("name")
        print(f"ℹ️ Заглушка: отримання даних з API '{source_name}'...")
        return []

    async def fetch_all(self) -> List[Dict]:
        """Асинхронно запускає збір новин з усіх активованих джерел."""
        all_articles = []
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for source_config in self.sources:
                source_type = source_config.get("type")
                if source_type == "rss":
                    tasks.append(self._fetch_rss(session, source_config))
                elif source_type == "api":
                    tasks.append(self._fetch_api(session, source_config))
            
            results = await asyncio.gather(*tasks)
            
            for article_list in results:
                all_articles.extend(article_list)
        
        all_articles.sort(key=lambda x: x["published"], reverse=True)
        print(f"📈 Всього зібрано {len(all_articles)} новин з {len(self.sources)} джерел.")
        return all_articles
