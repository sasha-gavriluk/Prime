from kivy.uix.screenmanager import ScreenManager

from gui.old.StartedWindow import StartedWindow
from gui.old.JobWindow import JobWindow
from gui.old.SettingsWindow import SettingsWindow
from gui.old.SettingsWindowPlus import SettingsWindowPlus
from gui.old.HistoryWindow import HistoryWindow
from gui.old.SimpleSettingsWindow import SimpleSettingsWindow
from gui.old.StrategyBuilderWindow import StrategyBuilderWindow

from utils.connectors.ConnectionController import ConnectionController
from utils.nn.NeuralNetworkManager import NeuralNetworkManager

class WindowManager(ScreenManager):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (1, 1)

        self.connection_controller = ConnectionController()
        self.nn_manager = NeuralNetworkManager(
            model_path="data/Updates/Neyron/models/1_step30_10milion/final_trading_model.zip"
        )

        # Додаємо екрани
        self.add_widget(StartedWindow(name='start', connection_controller=self.connection_controller))
        self.add_widget(JobWindow(name='job', connection_controller=self.connection_controller, nn_manager=self.nn_manager))
        self.add_widget(SettingsWindow(name='settings', connection_controller=self.connection_controller))
        self.add_widget(SettingsWindowPlus(name='settings_plus', connection_controller=self.connection_controller))
        self.add_widget(SimpleSettingsWindow(name='settings_simple', connection_controller=self.connection_controller, nn_manager=self.nn_manager))
        self.add_widget(HistoryWindow(name='history'))
        self.add_widget(StrategyBuilderWindow(name='strategy_builder'))

        # Вказуємо, який екран стартовий
        self.current = 'start'
