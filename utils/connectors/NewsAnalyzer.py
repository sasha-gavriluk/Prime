import re
from typing import List, Dict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- ВИДАЛЕНО SettingsLoader ---

class NewsAnalyzer:
    """
    Аналізує текстові дані новин.
    Тепер отримує свою конфігурацію ззовні.
    """

    def __init__(self, config: Dict):
        """
        Ініціалізує аналізатор, використовуючи передану конфігурацію.

        :param config: Словник з налаштуваннями, що містить 'keywords' та 'impact_multipliers'.
        """
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        self.keywords_config = config.get("keywords", {})
        self.impact_multipliers = config.get("impact_multipliers", {})
        
        print("📰 NewsAnalyzer ініціалізовано з переданою конфігурацією.")

    def analyze_sentiment(self, text: str) -> float:
        """Аналізує тональність тексту і повертає зведений бал."""
        score = self.sentiment_analyzer.polarity_scores(text)
        return score['compound']

    def detect_keywords(self, text: str) -> Dict[str, List[str]]:
        """Знаходить ключові слова з конфігурації в тексті."""
        found_by_category = {}
        text_lower = text.lower()
        
        for category, keywords in self.keywords_config.items():
            found = []
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', text_lower):
                    found.append(keyword)
            if found:
                found_by_category[category] = found
        return found_by_category

    def calculate_impact_score(self, sentiment: float, keywords: Dict) -> float:
        """Розраховує фінальний бал впливу."""
        impact_score = sentiment
        for category, found_list in keywords.items():
            if found_list:
                multiplier = self.impact_multipliers.get(category, 1.0)
                impact_score *= multiplier
        return max(-1.0, min(1.0, impact_score))

    def process_articles(self, articles: List[Dict]) -> List[Dict]:
        """Основний метод, який обробляє список статей."""
        processed_articles = []
        for article in articles:
            text_to_analyze = f"{article.get('headline', '')}. {article.get('summary', '')}"
            
            sentiment_score = self.analyze_sentiment(text_to_analyze)
            found_keywords = self.detect_keywords(text_to_analyze)
            impact_score = self.calculate_impact_score(sentiment_score, found_keywords)
            
            article['sentiment_score'] = round(sentiment_score, 3)
            article['keywords'] = found_keywords
            article['impact_score'] = round(impact_score, 3)
            
            processed_articles.append(article)
        return processed_articles
