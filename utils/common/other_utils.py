
import os
import pandas as pd

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..' , '..'))

def load_files(paths, reset_index=True, limit_rows=None, use_filename_as_key=True):
    """
    Завантажує один або кілька файлів.

    :param paths: шлях (str) або список шляхів (list[str]).
    :param reset_index: чи скидати індекси (default=True).
    :param limit_rows: обмеження кількості останніх рядків (default=None).
    :param use_filename_as_key: якщо True, використовувати ім'я файлу як ключ; інакше повний шлях.
    :return: DataFrame або словник {назва файлу: DataFrame}.
    """
    if isinstance(paths, str):
        paths = [paths]

    results = {}

    for path in paths:
        if not os.path.exists(path):
            print(f"⚠️ Файл не знайдено: {path}")
            continue

        try:
            df = pd.read_csv(path)

            if 'close_time' in df.columns:
                df['close_time'] = pd.to_datetime(df['close_time'])

            if limit_rows is not None:
                df = df.tail(limit_rows)

            if reset_index:
                df = df.reset_index(drop=True)

            # НОВЕ: додати таймфрейм до ключа
            base_name = os.path.splitext(os.path.basename(path))[0]  # BNB_USDT_processing
            timeframe_dir = os.path.basename(os.path.dirname(os.path.dirname(path))) # 1m, 5m, 15m, 30m, 1h, 4h, 1d
            key = f"{timeframe_dir}_{base_name}" if use_filename_as_key else path

            results[key] = df

        except Exception as e:
            print(f"❌ Помилка при завантаженні файлу {path}: {e}")

    # Якщо тільки один файл — повертаємо просто DataFrame
    if len(results) == 1:
        return list(results.values())[0]
    else:
        return results

def tf_to_minutes(tf: str) -> int:
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    if tf.endswith("d"):
        return int(tf[:-1]) * 60 * 24
    raise ValueError(f"Невідомий формат таймфрейму '{tf}'")