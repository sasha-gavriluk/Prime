import asyncio
import traceback
import pandas as pd

from typing import Dict, List, Optional
from utils.analysis.DecisionEngine import DecisionEngine
from utils.financial.FinancialAdvisor import FinancialAdvisor
from utils.financial.FinancialContext import FinancialContext
from utils.connectors.NewsFetcher import NewsFetcher
from utils.connectors.NewsAnalyzer import NewsAnalyzer
from utils.common.SettingsLoader import SettingsLoader, SettingsLoaderStrategies
from utils.strategies.StrategyObject import StrategyObject
from utils.analysis.DecisionPolicy import HysteresisCooldownPolicy

class DecisionOrchestrator:
    """
    Orchestrator: автоматично запускає TimeframeAgent-и, опціонально — аналіз новин,
    і формує фінальне рішення через DecisionEngine.
    """

    def __init__(
        self,
        timeframe_agents: List,
        # НОВИЙ ПАРАМЕТР: словник з сирими даними по кожному ТФ
        raw_data_by_tf: Dict[str, pd.DataFrame], 
        timeframe_weights: Dict,
        metric_weights: Dict,
        strategy_configs: Dict,
        user_financial_settings: Dict,
        trade_mode: str = 'spot',
        enable_news: bool = True
    ):
        # ВИПРАВЛЕНО: Приймаємо список агентів, а не створюємо порожній
        self.timeframe_agents = timeframe_agents
        self.timeframe_weights = timeframe_weights
        self.metric_weights = metric_weights
        self.raw_data_by_tf = raw_data_by_tf

        strategy_weights = {name: config.get('weight', 1.0) for name, config in strategy_configs.items()}
        self.engine = DecisionEngine(timeframe_weights, metric_weights, strategy_weights)
        self.strategy_configs = strategy_configs
        
        # Зберігаємо повні конфігурації для використання в цьому класі
        self.strategy_configs = strategy_configs
        self.enable_news = enable_news

        # Цей код залишається, він не є причиною помилки, але може знадобитися для новин
        self.gui_settings_loader = SettingsLoader("GUI")
        self.tf_configurations_from_gui = self.gui_settings_loader.get_nested_setting(['user_selections', 'tf_configurations'], {})

        # --- ІНТЕГРАЦІЯ ФІНАНСОВИХ МОДУЛІВ ---
        self.advisor = FinancialAdvisor()
        self.financial_context = FinancialContext(
            user_settings=user_financial_settings,
            advisor=self.advisor,
            trade_mode=trade_mode
        )

        self.policy = HysteresisCooldownPolicy(conf_margin=0.07, cooldown_bars=3)

        if self.enable_news:
            gui_sources = self.gui_settings_loader.get_nested_setting(['user_selections', 'news_sources'], [])
            gui_multipliers = self.gui_settings_loader.get_nested_setting(['user_selections', 'news_multipliers'], {})
            
            analyzer_defaults = SettingsLoader("NewsAnalyzer").settings
            keywords = analyzer_defaults.get("keywords", {})

            fetcher_config = {'sources': gui_sources}
            analyzer_config = {'keywords': keywords, 'impact_multipliers': gui_multipliers}

            self.news_fetcher = NewsFetcher(config=fetcher_config)
            self.news_analyzer = NewsAnalyzer(config=analyzer_config)
            print("📰 Оркестратор ініціалізував новинні модулі.")

    def _current_bar_index(self) -> int:
        """
        Дуже проста оцінка «поточного бару»: беремо найдовший DataFrame з raw_data_by_tf.
        Якщо хочеш — тут можна підставити індекс primary TF.
        """
        if not self.raw_data_by_tf:
            return 0
        return max(len(df) for df in self.raw_data_by_tf.values() if df is not None)

    def _run_strategy_analysis(self) -> Dict:
        """
        ПОВНІСТЮ ОНОВЛЕНИЙ МЕТОД.
        Запускає аналіз по активних стратегіях, правильно співставляючи їх з даними.
        """
        strategy_signals = {}
        active_strategies = [
            name for name, config in self.strategy_configs.items() if config.get('enabled', False)
        ]
        
        if not active_strategies:
            return {}

        print("\n--- ♟️ Запуск аналізу по активних стратегіях (Нова логіка) ---")
        for strategy_key in active_strategies:
            # 1. Визначаємо, який таймфрейм потрібен цій стратегії
            required_tf = self._peek_strategy_tf(strategy_key)
            if not required_tf:
                print(f"⚠️ Пропускаємо стратегію '{strategy_key}': не вказано 'timeframe_id' в її .json файлі.")
                continue

            # 2. Знаходимо відповідні сирі дані
            raw_ohlcv_data = self.raw_data_by_tf.get(required_tf)
            if raw_ohlcv_data is None or raw_ohlcv_data.empty:
                print(f"⚠️ Пропускаємо стратегію '{strategy_key}': не знайдено сирих даних для її таймфрейму '{required_tf}'.")
                continue

            # 3. Створюємо StrategyObject з правильними даними
            try:
                # Передаємо сирі дані. Решту StrategyObject зробить сам.
                strat = StrategyObject(strategy_key=strategy_key, raw_ohlcv_data=raw_ohlcv_data)
                signals = strat.generate_signals()
                if signals.empty: continue
                
                last_signal = signals.iloc[-1]
                strategy_signals[strategy_key] = last_signal
                print(f"  - Стратегія '{strategy_key}' ({required_tf}) дає сигнал: {last_signal.upper()}")
            except Exception as e:
                print(f"❌ Помилка при аналізі стратегії '{strategy_key}': {e}")
                traceback.print_exc() # Для детального дебагу
                
        return strategy_signals
    
    def _peek_strategy_tf(self, strategy_key: str) -> Optional[str]:
        """Допоміжна функція, щоб "підглянути" timeframe_id у файлі стратегії."""
        try:
            loader = SettingsLoaderStrategies(module_name="Strategies", name_strategy=strategy_key)
            return loader.settings.get("timeframe_id")
        except Exception as e:
            print(f"❌ Не вдалося прочитати timeframe_id для стратегії '{strategy_key}': {e}")
            return None

    def _find_best_timeframe(self, analysis_results: Dict) -> str:
        """
        Визначає найкращий таймфрейм для входу в угоду на основі
        комбінації сили сигналу та надійності таймфрейму.

        :param analysis_results: Словник з результатами аналізу від усіх TimeframeAgent.
        :return: Назва рекомендованого таймфрейму (напр., '1h').
        """
        best_tf = None
        max_score = -1

        print("\n--- 📈 Оцінка придатності таймфреймів для угоди ---")
        for tf, metrics in analysis_results.items():
            # Базові метрики для оцінки
            state_strength = metrics.get('state_strength', 0.0)
            smc_confidence = metrics.get('smc_confidence', 0.0)
            
            # Вага, що відображає надійність старших таймфреймів
            reliability_weight = self.timeframe_weights.get(tf, 1.0)

            # Формула оцінки: (Сила стану + Сила SMC) * Надійність ТФ
            # Це надає перевагу ТФ з чітким сигналом та вищою надійністю.
            score = (state_strength + smc_confidence) * reliability_weight
            
            print(f"  - {tf}: Сила стану={state_strength:.2f}, Впевненість SMC={smc_confidence:.2f}, Вага={reliability_weight:.2f} -> Рахунок: {score:.2f}")

            if score > max_score:
                max_score = score
                best_tf = tf
        
        print(f"🏆 Рекомендований таймфрейм для розрахунків: {best_tf} (Рахунок: {max_score:.2f})")
        return best_tf

    def run(self):
        """
        Запускає аналіз всіх TimeframeAgent, опціонально новин, і повертає рішення.
        """
        # 1. Аналіз таймфреймів (технічний аналіз)
        analysis_results = {}
        for agent in self.timeframe_agents:
            tf_result = agent.run_analysis()
            tf_name = tf_result.get("timeframe")
            if tf_name:
                analysis_results[tf_name] = tf_result
            else:
                print(f"[Warning] TimeframeAgent повернув результат без 'timeframe': {tf_result}")

        # 1.5. Аналіз по стратегіях
        strategy_signals = self._run_strategy_analysis()

        # 2. Аналіз новин (фундаментальний аналіз)
        processed_articles = None
        if self.enable_news:
            print("🔄 Оркестратор запускає отримання та аналіз новин...")
            
            # Використовуємо внутрішню асинхронну функцію для чистоти коду
            async def run_async_tasks():
                articles = await self.news_fetcher.fetch_all()
                if not articles:
                    return None
                return self.news_analyzer.process_articles(articles)

            try:
                processed_articles = asyncio.run(run_async_tasks())
                if processed_articles:
                    print(f"✅ Новини успішно оброблено. Знайдено статей: {len(processed_articles)}")
                else:
                    print("ℹ️ Свіжих новин не знайдено.")
            except Exception as e:
                print(f"❌ Помилка під час асинхронного запуску новин: {e}")

        # 3. Отримання технічного рішення від "мозку"
        technical_decision = self.engine.make_decision(analysis_results, processed_articles, strategy_signals)

        bar_idx = self._current_bar_index()
        final_dir, final_conf = self.policy.decide(
            technical_decision.get("direction", "neutral"),
            float(technical_decision.get("confidence", 0.0)),
            bar_idx
        )
        technical_decision["direction"] = final_dir
        technical_decision["confidence"] = round(final_conf, 3)
        technical_decision["policy_bar_idx"] = bar_idx
        
        # --- ІНТЕГРАЦІЯ: ВИБІР ТФ ТА ЗБАГАЧЕННЯ ФІНАНСОВИМ КОНТЕКСТОМ ---
        
        # 3.1. Визначаємо найкращий таймфрейм на основі аналізу
        recommended_tf = self._find_best_timeframe(analysis_results)
        
        financial_briefing = {}
        if recommended_tf:
            primary_agent = next((agent for agent in self.timeframe_agents if agent.timeframe == recommended_tf), None)
            
            # 3.2. Витягуємо поточну ціну та ATR з рекомендованого ТФ
            current_price = primary_agent.data['close'].iloc[-1]
            atr_column = next((col for col in primary_agent.data.columns if 'ATR' in col), None)
            atr_value = primary_agent.data[atr_column].iloc[-1] if atr_column and not primary_agent.data[atr_column].empty else None
            
            # 3.3. Генеруємо фінансовий брифінг
            financial_briefing = self.financial_context.generate_financial_briefing(
                signal_data=technical_decision,
                current_price=current_price,
                atr_value=atr_value
            )
            # Додаємо інформацію про те, який ТФ був обраний
            financial_briefing['recommended_tf_for_financials'] = recommended_tf
        else:
            financial_briefing = {
                "status": "error", 
                "reason": "Не вдалося визначити рекомендований таймфрейм для фінансових розрахунків."
            }

        # 4. Формуємо фінальний комплексний результат
        final_result = {
            "technical_decision": technical_decision,
            "financial_briefing": financial_briefing,
            "raw_analysis_by_tf": analysis_results
        }
        
        return final_result
    