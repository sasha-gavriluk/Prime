# gui/SimpleSettingsWindow.py

from kivy.uix.screenmanager import Screen
from kivy.core.window import Window

from gui.UIManager import UIManager
from utils.common.SettingsLoader import SettingsLoader
from utils.nn.NeuralNetworkManager import NeuralNetworkManager

class SimpleSettingsWindow(Screen):
    """A minimal settings screen with a back button to the start window."""

    def __init__(self, connection_controller=None, nn_manager=None, **kwargs):
        super().__init__(**kwargs)
        self.connection_controller = connection_controller
        self.ui = UIManager(canvas_size=(800, 600))
        self.settings_loader = SettingsLoader("GUI")
        self.nn_enabled = self.settings_loader.get_nested_setting(
            ["user_selections", "neural_network", "enabled"], False
        )

        # Preload manager with path to PPO model; will fallback to dummy if unavailable
        self.nn_manager = nn_manager or NeuralNetworkManager(
            model_path="data/NNUpdates/Neyron/models/1_step30_10milion/final_trading_model.zip"
        )
        self.build_ui()
        root = self.ui.build()
        self.add_widget(root)

        # Cache widgets for hover detection
        self.nn_switch_widget = self.ui.registry.get("nn_switch")
        self.nn_tooltip = self.ui.registry.get("nn_tooltip")
        if self.nn_tooltip:
            self.nn_tooltip.opacity = 0

        if self.nn_enabled and self.nn_switch_widget:
            self.toggle_nn(self.nn_switch_widget, True)

        # Bind mouse movement to show tooltip on hover
        Window.bind(mouse_pos=self.on_mouse_pos)

    def build_ui(self):
        # Header label
        self.ui.add("settings_label", type="Label", text="Налаштування", style_class="label_header")
        self.ui.set_bounds("settings_label", 0, 10, 800, 60)

        # Neural network toggle
        self.ui.add("nn_label", type="Label", text="Нейромережа")
        self.ui.set_bounds("nn_label", 10, 100, 200, 140)
        self.ui.add("nn_switch", type="Switch", active=self.nn_enabled)
        self.ui.set_bounds("nn_switch", 220, 110, 260, 140)
        self.ui.set_action("nn_switch", "active", self.toggle_nn)

        # Tooltip for neural network toggle
        self.ui.add(
            "nn_tooltip",
            type="Label",
            text="Вмикає або вимикає нейромережу для аналізу ринку",
            opacity=0,
        )
        self.ui.set_bounds("nn_tooltip", 220, 140, 600, 170)

        # Back button
        self.ui.add("btn_back", type="Button", text="<-- Назад", style_class="button_secondary")
        self.ui.set_bounds("btn_back", 10, 540, 150, 590)
        self.ui.set_action("btn_back", "on_press", self.go_back)

    def go_back(self, instance):
        # Return to the start screen
        self.save_settings()
        self.manager.current = "start"

    def toggle_nn(self, instance, value):
        self.nn_enabled = value
        # Persist current state so other windows can react without re-toggling
        self.settings_loader.update_nested_setting(
            ["user_selections", "neural_network", "enabled"], value
        )
        if value:
            # Load the neural network model (PPO if available) and lock manual settings
            try:
                self.nn_manager.load()
            except Exception as exc:
                print(f"Failed to load neural network: {exc}")
                self.nn_enabled = False
                if instance:
                    instance.active = False
                return
        else:
            # Unload the network and unlock settings for manual control
            self.nn_manager.unload()

    def on_mouse_pos(self, window, pos):
        if not self.nn_switch_widget or not self.nn_tooltip:
            return
        if self.nn_switch_widget.collide_point(*self.nn_switch_widget.to_widget(*pos)):
            self.nn_tooltip.opacity = 1
        else:
            self.nn_tooltip.opacity = 0

    def on_enter(self, *args):
        """Reload settings and apply current neural network state."""
        self.settings_loader.reload()
        self.nn_enabled = self.settings_loader.get_nested_setting(
            ["user_selections", "neural_network", "enabled"], False
        )
        if self.nn_switch_widget:
            # Avoid triggering toggle twice when setting active programmatically
            self.nn_switch_widget.unbind(active=self.toggle_nn)
            self.nn_switch_widget.active = self.nn_enabled
            self.nn_switch_widget.bind(active=self.toggle_nn)
        self.toggle_nn(self.nn_switch_widget, self.nn_enabled)

    def save_settings(self):
        """Persist current neural network toggle state."""
        self.settings_loader.update_nested_setting(
            ["user_selections", "neural_network", "enabled"], self.nn_enabled
        )

    def on_leave(self, *args):
        self.save_settings()