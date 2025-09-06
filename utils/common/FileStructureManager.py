
import os

from utils.common.other_utils import get_project_root
from utils.common.SettingsLoader import SettingsLoader


class FileStructureManager:
    def __init__(self):
        self.root_path = get_project_root()
        self.settings_loader = SettingsLoader("FileStructure")
        self.structure = self.settings_loader.settings

    def scan_and_save_structure(self):
        """
        Сканує проєкт, будує карту структури і зберігає її у файл налаштувань.
        Ключі для папок і файлів однакові (без 'data/'), шляхи залишаються повними (з 'data/').
        """
        structure = {
            "root_file": "bot.py",
            "root_path": self.root_path,
            "directories": {},
            "files": {}
        }

        for root, dirs, files in os.walk(self.root_path):
            rel_root_full = os.path.relpath(root, self.root_path).replace("\\", "/")  # Повний шлях з data
            rel_root_key = rel_root_full

            if rel_root_key.startswith("data/"):
                rel_root_key = rel_root_key[len("data/"):]  # Для ключа обрізаємо 'data/'

            for d in dirs:
                dir_path_full = os.path.join(rel_root_full, d).replace("\\", "/")
                dir_key = "_".join(os.path.join(rel_root_key, d).replace("\\", "/").split("/"))

                structure["directories"][dir_key] = dir_path_full

            for f in files:
                name, ext = os.path.splitext(f)
                file_path_full = os.path.join(rel_root_full, f).replace("\\", "/")
                file_key = "_".join(os.path.join(rel_root_key, f).replace("\\", "/").split("/")).replace(".csv", "").replace(".json", "").replace(".txt", "")

                structure["files"][file_key] = file_path_full

        # Зберігаємо
        self.structure = structure
        self.settings_loader.settings = structure
        self.settings_loader.save_data()

    def get_path(self, key, is_file=True):
        """
        Отримує шлях до файлу чи папки за ключем.
        Якщо не знайдено — автоматично перескановує структуру.

        :param key: Ім'я файлу або папки без розширення.
        :param is_file: True для файлу, False для папки.
        :return: Повний шлях до файлу або папки.
        """
        if not self.structure:
            print("Структура файлів не знайдена. Скануємо...")
            self.scan_and_save_structure()

        base = "files" if is_file else "directories"

        rel_path = self.structure.get(base, {}).get(key)

        # Якщо ключа немає, або файл реально не існує
        if not rel_path or not os.path.exists(os.path.join(self.root_path, rel_path)):
            print(f"Ключ {key} не знайдено або файл відсутній. Оновлюємо структуру...")
            self.refresh_structure()
            rel_path = self.structure.get(base, {}).get(key)

        if rel_path:
            return os.path.join(self.root_path, rel_path)
        else:
            print(f"Ключ {key} все ще не знайдено після оновлення.")
            return None

    def refresh_structure(self):
        """
        Перескановує структуру папок і файлів, та оновлює збережені дані у налаштуваннях.
        """
        print("Оновлення структури файлів...")
        self.scan_and_save_structure()
        print("Структуру оновлено.")

    def get_all_files_in_directory(self, directory_key, subfolder=None, extensions=None):
        """
        Повертає список всіх файлів у вказаній директорії (і підпапках) з можливістю фільтрації за розширенням.

        :param directory_key: Ключ основної папки (наприклад "Binance")
        :param subfolder: Назва підпапки (наприклад "1m") або None, якщо не потрібно
        :param extensions: Список розширень для фільтрації (наприклад ['.csv', '.json']) або None для всіх файлів
        :return: Список повних шляхів до знайдених файлів
        """
        dir_path = self.get_path(directory_key, is_file=False)

        if not dir_path or not os.path.exists(dir_path):
            print(f"Папка для ключа {directory_key} не знайдена.")
            return []

        # Якщо вказана підпапка — переходимо глибше
        if subfolder:
            dir_path = os.path.join(dir_path, subfolder)
            if not os.path.exists(dir_path):
                print(f"Підпапка {subfolder} в {directory_key} не знайдена.")
                return []

        all_files = []
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if extensions:
                    if any(file.endswith(ext) for ext in extensions):
                        file_path = os.path.join(root, file)
                        all_files.append(file_path)
                else:
                    file_path = os.path.join(root, file)
                    all_files.append(file_path)

        return all_files

    def get_all_strategy_names(self) -> list:
        """
        Сканує папку data/Strategies і повертає імена всіх знайдених стратегій.
        Ім'я стратегії - це назва файлу без суфікса '_strategy.json'.
        
        :return: Список імен стратегій (напр., ['my_super_strategy', 'another_one']).
        """
        strategies_dir = os.path.join(self.root_path, 'data', 'Strategies')
        strategy_names = []
        
        if not os.path.exists(strategies_dir):
            print(f"⚠️ Папка для стратегій не знайдена: {strategies_dir}")
            return []
            
        for filename in os.listdir(strategies_dir):
            if filename.endswith("_strategy.json"):
                strategy_name = filename.replace("_strategy.json", "")
                strategy_names.append(strategy_name)
                
        return strategy_names