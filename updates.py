# NNUpdates.py
import os
import re
import glob
import sys
from typing import List, Optional

from stable_baselines3 import PPO

# Твої модулі
from NNUpdates.Neyron.environment import TradingEnv
from NNUpdates.Neyron.agent import TradingAgent
from NNUpdates.Neyron.data_preparer import create_market_seasons  # інтерактивна генерація сезонів

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    console = Console()
except Exception:
    console = None

def cprint(msg: str):
    if console:
        console.print(msg)
    else:
        print(msg)

MODE_DESCRIPTIONS = {
    "1": "Нове навчання: тренуємо PPO з нуля на обраних сезонах.",
    "2": "Продовження: дозавантажуємо *останній* чекпойнт і вчимо далі.",
    "3": "Донавчання: беремо чекпойнт і вчимо на НОВИХ сезонах з LR для fine-tune.",
    "4": "Демонстрація: ганяємо обрані епізоди й друкуємо сумарну винагороду.",
}

# ====== Константи-дефолти ======
MODELS_DIR_DEFAULT = "data/Updates/Neyron/models"
SEASONS_DIR_DEFAULT = "data/Updates/Neyron/seasons"
NEW_SEASONS_DIR_DEFAULT = "data/Updates/Neyron/seasons"
SKILLS_MANIFEST_DEFAULT = "NNUpdates/Neyron/skills/skills_manifest.txt"
AGENT_PROFILE_DEFAULT = "NNUpdates/Neyron/agents/my_tf_agents.txt"
P_OFF_DEFAULT = 0.0
P_MIDFLIP_DEFAULT = 0.0
TIMESTEPS_DEFAULT = 200_000
LR_FINETUNE_DEFAULT = 1e-4
DEVICE_DEFAULT = "auto"  # 'cpu' | 'cuda' | 'auto'


# ====== Утіліти ======
def prompt_str(msg: str, default: Optional[str] = None) -> str:
    tip = f" [{default}]" if default is not None else ""
    try:
        s = input(f"{msg}{tip}: ").strip()
    except EOFError:
        s = ""
    return s if s else (default or "")

def prompt_int(msg: str, default: int) -> int:
    s = prompt_str(msg, str(default))
    try:
        return int(s)
    except ValueError:
        return default

def prompt_float(msg: str, default: float) -> float:
    s = prompt_str(msg, str(default))
    try:
        return float(s)
    except ValueError:
        return default
    
def prompt_skill_params(eval_mode: bool):
    manifest = prompt_str("Шлях до skills-маніфесту", SKILLS_MANIFEST_DEFAULT)
    profile = prompt_str("Шлях до профілю агента (може бути пусто)", AGENT_PROFILE_DEFAULT)
    p_off = P_OFF_DEFAULT
    p_midflip = P_MIDFLIP_DEFAULT
    if not eval_mode:
        p_off = prompt_float("p_off (ймовірність вимкнення)", P_OFF_DEFAULT)
        p_midflip = prompt_float("p_midflip (ймовірність перемикання)", P_MIDFLIP_DEFAULT)
    return (manifest or None, profile or None, p_off, p_midflip)

def list_csvs(path_or_dir: str) -> List[str]:
    """Повертає всі CSV з директорії (рекурсивно) або один файл."""
    if not path_or_dir:
        return []
    if os.path.isdir(path_or_dir):
        return sorted(glob.glob(os.path.join(path_or_dir, "**", "*.csv"), recursive=True))
    if os.path.isfile(path_or_dir) and path_or_dir.lower().endswith(".csv"):
        return [path_or_dir]
    return []

def find_latest_checkpoint(models_dir: str, prefix: str = "trading_model") -> str:
    """Знаходить останній чекпойнт у models_dir за найбільшим числом кроків."""
    if not os.path.isdir(models_dir):
        return ""

    candidates = glob.glob(os.path.join(models_dir, f"{prefix}_*_steps.zip"))
    best_path = ""
    best_steps = -1
    rx = re.compile(rf"{re.escape(prefix)}_(\d+)_steps\.zip$")

    for p in candidates:
        m = rx.search(os.path.basename(p))
        if m:
            steps = int(m.group(1))
            if steps > best_steps:
                best_steps = steps
                best_path = p

    if best_path:
        return best_path

    final_path = os.path.join(models_dir, "final_trading_model.zip")
    return final_path if os.path.isfile(final_path) else ""

def require_seasons(dir_or_file: str, purpose_name: str) -> List[str]:
    seasons = list_csvs(dir_or_file)
    if not seasons:
        raise FileNotFoundError(
            f"❌ Не знайшов жодного CSV для {purpose_name} у '{dir_or_file}'. "
            f"Перевір шлях або згенеруй сезони."
        )
    return seasons

def banner(msg: str):
    if console:
        cprint(Panel.fit(f" {msg} ", border_style="cyan", title="Меню оновлень"))
    else:
        print("\n" + "=" * 80)
        print(msg)
        print("=" * 80 + "\n")

# ====== Пошук підпапок і меню вибору ======
def _subdirs_with_any(root: str, patterns: List[str]) -> List[str]:
    """Повертає підпапки root (1 рівень), у яких є файли, що матчаться хоч одному pattern (рекурсивно)."""
    if not os.path.isdir(root):
        return []
    subdirs = sorted([os.path.join(root, d) for d in os.listdir(root)
                      if os.path.isdir(os.path.join(root, d))])
    out = []
    for d in subdirs:
        ok = False
        for pat in patterns:
            if glob.glob(os.path.join(d, "**", pat), recursive=True):
                ok = True
                break
        if ok:
            out.append(d)
    return out

def _choose_from_list(title: str, options: List[str], add_manual: bool = True) -> str:
    """Дає вибір через questionary або звичайний ввід номера."""
    opts = list(options)
    if add_manual:
        opts.append("Ввести шлях вручну...")
    try:
        import questionary
        choice = questionary.select(
            title,
            choices=opts,
            default=opts[0] if opts else None,
        ).ask()
        if not choice:
            return ""
        if add_manual and choice == "Ввести шлях вручну...":
            return prompt_str("Вкажіть шлях")
        return choice
    except Exception:
        # Фолбек: номерне меню
        if not opts:
            return prompt_str("Список порожній. Вкажіть шлях вручну")
        print(f"\n{title}")
        for i, o in enumerate(opts, 1):
            print(f"  [{i}] {o}")
        sel = prompt_str("Ваш вибір (номер)", "1")
        try:
            idx = int(sel) - 1
            if idx < 0 or idx >= len(opts):
                raise ValueError
            choice = opts[idx]
            if add_manual and choice == "Ввести шлях вручну...":
                return prompt_str("Вкажіть шлях")
            return choice
        except Exception:
            return opts[0]

def choose_models_dir(root: str = MODELS_DIR_DEFAULT) -> str:
    """Знаходить підпапки з model *.zip і пропонує вибір."""
    patterns = ["*.zip"]
    found = _subdirs_with_any(root, patterns)
    # Якщо нічого — все одно дозволимо вибрати root або вручну
    options = found if found else ([root] if os.path.isdir(root) else [])
    return _choose_from_list("Оберіть папку моделей:", options)

def choose_seasons_dir(root: str = SEASONS_DIR_DEFAULT) -> str:
    """Знаходить підпапки з *.csv сезонами і пропонує вибір."""
    patterns = ["*.csv"]
    found = _subdirs_with_any(root, patterns)
    options = found if found else ([root] if os.path.isdir(root) else [])
    return _choose_from_list("Оберіть папку сезонів:", options)

def choose_new_seasons_dir(root: str = NEW_SEASONS_DIR_DEFAULT) -> str:
    patterns = ["*.csv"]
    found = _subdirs_with_any(root, patterns)
    options = found if found else ([root] if os.path.isdir(root) else [])
    return _choose_from_list("Оберіть папку НОВИХ сезонів:", options)

# ====== Інтерактивна генерація/оновлення сезонів ======
def _rename_manifests_to_txt(root_dir: str):
    patterns = [
        os.path.join(root_dir, "**", "*manifest*.csv"),
        os.path.join(root_dir, "**", "manifest.csv"),
        os.path.join(root_dir, "**", "seasons_manifest.csv"),
    ]
    changed = 0
    for pat in patterns:
        for p in glob.glob(pat, recursive=True):
            base = os.path.splitext(p)[0]
            newp = base + ".txt"
            try:
                if os.path.isfile(p):
                    os.replace(p, newp)
                    changed += 1
            except Exception:
                pass
    if changed:
        print(f"✏️ Перейменовано маніфестів у .txt: {changed}")

def maybe_generate_seasons_interactive():
    ans = input("\n🧩 Згенерувати/оновити сезони перед стартом? (y/N): ").strip().lower()
    if ans != "y":
        return
    source_key = input("Ключ джерела у FileStructureManager (напр., data_Binance) [data_Binance]: ").strip() or "data_Binance"
    out_dir    = input("Куди класти сезони [data/NNUpdates/Neyron/seasons]: ").strip() or "data/NNUpdates/Neyron/seasons"
    try:
        length = int(input("Довжина сезону у свічках [720]: ").strip() or "720")
    except ValueError:
        length = 720
    stride_input = input(f"Stride (крок зсуву) [{length} = без перекриття]: ").strip()
    try:
        stride = int(stride_input) if stride_input else length
    except ValueError:
        stride = length
    mode = input("Режим зі старими сезонами: append / overwrite / archive [append]: ").strip().lower() or "append"
    if mode not in ("append", "overwrite", "archive"):
        mode = "append"

    print("\n⚙️ Генерація сезонів...")
    created = create_market_seasons(
        source_directory_key=source_key,
        season_length_candles=length,
        stride=stride,
        output_dir=out_dir,
        mode=mode,
        validate=True,
    )
    print(f"📦 Створено {len(created)} сезонів у: {out_dir}")

    # ⬇️ На випадок старих версій: прибираємо .csv-маніфести → .txt
    _rename_manifests_to_txt(out_dir)

# ====== Режими ======
def mode_new_training(models_dir: str, seasons_dir: str, timesteps: int, device: str):
    banner("🆕 Нове навчання (fresh train)")
    seasons = require_seasons(seasons_dir, "тренування")

    manifest, profile, p_off, p_midflip = prompt_skill_params(eval_mode=False)

    env = TradingEnv(
        market_season_files=seasons,
        skills_manifest=manifest,
        profile_path=profile,
        p_off=p_off,
        p_midflip=p_midflip,
    )
    agent = TradingAgent(env=env, model_save_path=models_dir)

    if device in ("cpu", "cuda"):
        try:
            agent.model.set_parameters(agent.model.get_parameters(), exact_match=False, device=device)
        except Exception:
            pass
        agent.model.device = device

    agent.train(total_timesteps=timesteps)

def mode_resume(models_dir: str, seasons_dir: str, timesteps: int, device: str):
    banner("⏯️ Продовження навчання (resume)")
    ckpt = find_latest_checkpoint(models_dir)
    if not ckpt:
        raise FileNotFoundError(
            f"❌ Чекпойнт не знайдено у '{models_dir}'. Спочатку запусти тренування (або перевір шлях)."
        )

    seasons = require_seasons(seasons_dir, "resume")
    manifest, profile, p_off, p_midflip = prompt_skill_params(eval_mode=False)
    env = TradingEnv(
        market_season_files=seasons,
        skills_manifest=manifest,
        profile_path=profile,
        p_off=p_off,
        p_midflip=p_midflip,
    )

    model = PPO.load(ckpt, env=env, device=device)
    model.learn(total_timesteps=timesteps, reset_num_timesteps=False, progress_bar=True)

    out_path = os.path.join(models_dir, "final_trading_model.zip")
    model.save(out_path)
    print(f"✅ Продовжене навчання збережено: {out_path}")

def mode_finetune(models_dir: str, new_seasons_dir: str, timesteps: int, lr: float, device: str):
    banner("🧪 Донавчання на нових сезонах (fine-tune)")
    ckpt = find_latest_checkpoint(models_dir)
    if not ckpt:
        raise FileNotFoundError(
            f"❌ Чекпойнт не знайдено у '{models_dir}'. Спочатку запусти тренування (або перевір шлях)."
        )

    new_seasons = require_seasons(new_seasons_dir, "fine-tune")
    manifest, profile, p_off, p_midflip = prompt_skill_params(eval_mode=False)
    env_new = TradingEnv(
        market_season_files=new_seasons,
        skills_manifest=manifest,
        profile_path=profile,
        p_off=p_off,
        p_midflip=p_midflip,
    )

    model = PPO.load(ckpt, device=device)  # завантажуємо без env
    model.set_env(env_new)
    model.learning_rate = lr

    model.learn(total_timesteps=timesteps, reset_num_timesteps=False, progress_bar=True)

    out_path = os.path.join(models_dir, "final_trading_model_finetuned.zip")
    model.save(out_path)
    print(f"✅ Модель донавчена: {out_path}")

def mode_inference(models_dir: str, seasons_dir: str, episodes: int, device: str):
    banner("🎬 Демонстрація (inference)")
    ckpt = find_latest_checkpoint(models_dir)
    if not ckpt:
        raise FileNotFoundError(
            f"❌ Чекпойнт не знайдено у '{models_dir}'. Спочатку натренуй або перевір шлях."
        )

    seasons = require_seasons(seasons_dir, "inference")
    env = TradingEnv(market_season_files=seasons)
    model = PPO.load(ckpt, env=env, device=device)

    rewards_run = []
    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
        print(f"   - Епізод {ep}: кроків={steps}, сума винагороди={total_reward:.4f}")
        rewards_run.append(total_reward)

    try:
        import statistics
        mean_r = sum(rewards_run) / len(rewards_run)
        median_r = statistics.median(rewards_run)
        std_r = statistics.pstdev(rewards_run) if len(rewards_run) > 1 else 0.0
        win_rate = sum(1 for r in rewards_run if r > 0) / len(rewards_run)
        best = max(rewards_run); worst = min(rewards_run)
        print("\n📊 Підсумок демонстрації:")
        print(f"   · Середня: {mean_r:.4f} | Медіана: {median_r:.4f} | Std: {std_r:.4f}")
        print(f"   · Win-rate (>0): {win_rate*100:.1f}%  ({sum(1 for r in rewards_run if r > 0)}/{len(rewards_run)})")
        print(f"   · Найкращий: {best:.4f} | Найгірший: {worst:.4f}")
    except Exception:
        pass

def show_main_menu_table():
    if not console:
        print("\nОберіть режим:")
        for k, v in MODE_DESCRIPTIONS.items():
            print(f"  [{k}] {v}")
        print("  [h] Пояснення режимів докладно")
        return

    from rich.table import Table
    table = Table(show_header=True, header_style="bold")
    table.add_column("№", style="bold cyan", width=3)
    table.add_column("Режим", style="bold")
    table.add_column("Що робить", style="dim")
    table.add_row("1", "🆕 Нове навчання (fresh train)", MODE_DESCRIPTIONS["1"])
    table.add_row("2", "⏯️ Продовження (resume)",      MODE_DESCRIPTIONS["2"])
    table.add_row("3", "🧪 Донавчання (fine-tune)",     MODE_DESCRIPTIONS["3"])
    table.add_row("4", "🎬 Демонстрація (inference)",   MODE_DESCRIPTIONS["4"])
    table.add_row("h", "ℹ️  Пояснення",                 "Детальна інструкція по всіх пунктах")
    cprint(table)

def select_mode_interactive(default_choice: str = "1") -> str:
    try:
        import questionary
        choice = questionary.select(
            "Оберіть режим:",
            choices=[
                questionary.Choice("🆕  Нове навчання (1)", value="1"),
                questionary.Choice("⏯️  Продовження (2)", value="2"),
                questionary.Choice("🧪  Донавчання (3)", value="3"),
                questionary.Choice("🎬  Демонстрація (4)", value="4"),
                questionary.Choice("ℹ️  Пояснення (h)", value="h"),
            ],
            default="1",
        ).ask()
        return choice or default_choice
    except Exception:
        show_main_menu_table()
        return prompt_str("Ваш вибір", default_choice).lower()

def show_help():
    text = (
        "[bold]ℹ️ Пояснення режимів[/bold]\n\n"
        "[bold cyan]1) Нове навчання (fresh train)[/bold cyan]\n"
        "– Візьме CSV сезони з папки, створить середовище і навчить PPO з нуля.\n"
        "– Параметри: [i]папка сезонів[/i], [i]кроки навчання[/i], [i]пристрій cpu/cuda/auto[/i].\n\n"
        "[bold cyan]2) Продовження (resume)[/bold cyan]\n"
        "– Знайде [i]останній[/i] чекпойнт у папці моделей і продовжить навчання, не обнуляючи кроки.\n"
        "– Параметри: [i]папка сезонів[/i], [i]додаткові кроки[/i], [i]пристрій[/i].\n\n"
        "[bold cyan]3) Донавчання (fine-tune)[/bold cyan]\n"
        "– Візьме чекпойнт, підставить [i]НОВІ сезони[/i], змінить [i]LR[/i] і повчить ще.\n"
        "– Параметри: [i]папка НОВИХ сезонів[/i], [i]кроки[/i], [i]LR[/i], [i]пристрій[/i].\n\n"
        "[bold cyan]4) Демонстрація (inference)[/bold cyan]\n"
        "– Завантажить модель і програє вказану кількість епізодів, виведе сумарну винагороду.\n"
        "– Параметри: [i]папка сезонів[/i], [i]кількість епізодів[/i], [i]пристрій[/i].\n\n"
        "[dim]Порада: перед меню можна інтерактивно згенерувати сезони.[/dim]\n"
    )
    if console:
        cprint(Panel(text, border_style="green", title="Довідка"))
    else:
        print(text)

# ====== Точка входу ======
def main():
    cprint("🚀 Запуск модуля оновлень...")
    maybe_generate_seasons_interactive()

    while True:
        banner("Оберіть режим")
        choice = select_mode_interactive(default_choice="1")
        if choice == "h":
            show_help()
            continue

        # === НОВЕ: вибір папки моделей зі списку підпапок ===
        models_dir = choose_models_dir(MODELS_DIR_DEFAULT)

        if choice == "1":
            seasons_dir = choose_seasons_dir(SEASONS_DIR_DEFAULT)
            timesteps  = prompt_int("К-ть кроків навчання", TIMESTEPS_DEFAULT)
            device     = prompt_str("Пристрій (cpu/cuda/auto)", DEVICE_DEFAULT).lower()
            mode_new_training(models_dir, seasons_dir, timesteps, device)

        elif choice == "2":
            seasons_dir = choose_seasons_dir(SEASONS_DIR_DEFAULT)
            timesteps  = prompt_int("Скільки кроків ще вчитись", TIMESTEPS_DEFAULT)
            device     = prompt_str("Пристрій (cpu/cuda/auto)", DEVICE_DEFAULT).lower()
            mode_resume(models_dir, seasons_dir, timesteps, device)

        elif choice == "3":
            new_seasons_dir = choose_new_seasons_dir(NEW_SEASONS_DIR_DEFAULT)
            timesteps  = prompt_int("Скільки кроків донавчати", TIMESTEPS_DEFAULT)
            lr         = prompt_float("LR для fine-tune", LR_FINETUNE_DEFAULT)
            device     = prompt_str("Пристрій (cpu/cuda/auto)", DEVICE_DEFAULT).lower()
            mode_finetune(models_dir, new_seasons_dir, timesteps, lr, device)

        elif choice == "4":
            seasons_dir = choose_seasons_dir(SEASONS_DIR_DEFAULT)
            episodes   = prompt_int("Кількість епізодів для демонстрації", 3)
            device     = prompt_str("Пристрій (cpu/cuda/auto)", DEVICE_DEFAULT).lower()
            mode_inference(models_dir, seasons_dir, episodes, device)

        else:
            cprint("[bold red]❌ Невідомий вибір. Оберіть 1–4 або h для довідки.[/bold red]")
            continue

        cprint("\n[bold green]✅ Готово.[/bold green]")
        break

if __name__ == "__main__":
    main()
