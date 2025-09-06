import os
import ast
import shutil
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple

def analyze_python_file(file_path: Path) -> Dict[str, Any]:
    """
    Аналізує один файл Python за допомогою Abstract Syntax Tree (AST),
    щоб витягти інформацію про імпорти, класи, методи та окремі функції.

    Args:
        file_path: Шлях до файлу Python.

    Returns:
        Словник з проаналізованою інформацією, включаючи детальні імпорти.
        Повертає None у випадку помилки.
    """
    analysis = {
        "imports": [],  # (тип_імпорту, модуль_джерело, імпортоване_ім'я, ім'я_під_яким_доступно)
        "classes": {},  # {class_name: [method_names]}
        "functions": [] # [function_names]
    }
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            source_code = file.read()
            tree = ast.parse(source_code, filename=file_path.name)
    except Exception as e:
        print(f"  [Помилка] Не вдалося прочитати або проаналізувати файл {file_path}: {e}")
        return None

    # Додаємо батьківські посилання до всіх вузлів AST
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node # type: ignore

    for node in ast.walk(tree):
        # Обробка імпортів: import module [as alias]
        if isinstance(node, ast.Import):
            for alias in node.names:
                analysis["imports"].append(("module", alias.name, alias.name, alias.asname or alias.name))
        # Обробка імпортів: from module import name [as alias]
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for alias in node.names:
                    analysis["imports"].append(("from", node.module, alias.name, alias.asname or alias.name))
        # Обробка визначення класів
        elif isinstance(node, ast.ClassDef):
            class_name = node.name
            methods = []
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    methods.append(item.name)
            analysis["classes"][class_name] = methods
        # Обробка визначення функцій на верхньому рівні (не методи)
        elif isinstance(node, ast.FunctionDef) and isinstance(getattr(node, 'parent', None), ast.Module):
             analysis["functions"].append(node.name)
            
    return analysis

def get_user_settings(base_path: Path) -> Dict[str, Any]:
    """
    Збирає налаштування від користувача в інтерактивному режимі.
    """
    settings = {}
    print("--- Налаштування Генератора Документації для Obsidian ---")

    while True:
        obsidian_path_str = input("1. Введіть повний шлях до вашого сховища (vault) Obsidian: ")
        settings['obsidian_path'] = Path(obsidian_path_str).resolve()
        if settings['obsidian_path'].is_dir():
            break
        print(f"  [Помилка] Шлях '{obsidian_path_str}' не існує або не є директорією. Будь ласка, спробуйте ще раз.")

    while True:
        project_path_str = input("2. Введіть повний шлях до директорії проєкту для сканування: ")
        settings['project_path'] = Path(project_path_str).resolve()
        if settings['project_path'].is_dir():
            break
        print(f"  [Помилка] Шлях '{project_path_str}' не існує або не є директорією. Будь ласка, спробуйте ще раз.")

    settings['excluded_dirs'] = set()
    if input("3. Чи бажаєте ви пропустити (виключити) певні директорії зі сканування? (так/ні): ").lower().strip() == 'так':
        try:
            all_dirs = [d for d in settings['project_path'].iterdir() if d.is_dir()]
            if not all_dirs:
                print("  У проєкті не знайдено піддиректорій для виключення.")
            else:
                print("\n  Наявні директорії у вашому проєкті:")
                for i, dirname in enumerate(all_dirs):
                    print(f"    {i + 1}. {dirname.name}")
                
                excluded_indices_str = input("\n  Введіть номери директорій для пропуску через кому (напр. 1, 3): ")
                excluded_indices = [int(i.strip()) - 1 for i in excluded_indices_str.split(',')]
                settings['excluded_dirs'] = {all_dirs[i].name for i in excluded_indices if 0 <= i < len(all_dirs)}
                print(f"  Будуть пропущені: {settings['excluded_dirs']}")
        except (ValueError, IndexError):
            print("  [Помилка] Невірний ввід. Пропуск директорій буде скасовано.")
            settings['excluded_dirs'] = set()

    response = input("5. Згенерувати нотатки в окремій папці всередині сховища Obsidian? (так/ні): ").lower().strip()
    if response == 'так':
        folder_name = settings['project_path'].name + "_docs"
        settings['output_path'] = settings['obsidian_path'] / folder_name
    else:
        settings['output_path'] = settings['obsidian_path']

    return settings

def confirm_and_run() -> None:
    """
    Головний цикл програми: налаштування, підтвердження та запуск.
    """
    base_path = Path.cwd()
    settings = None

    while True:
        if not settings:
            settings = get_user_settings(base_path)

        print("\n" + "="*50)
        print("Будь ласка, перевірте налаштування:")
        print(f"  1. Шлях до сховища Obsidian: {settings['obsidian_path']}")
        print(f"  2. Директорія проєкту для сканування: {settings['project_path']}")
        print(f"  3. Пропущені директорії: {settings['excluded_dirs'] or 'Немає'}")
        print(f"  4. Кінцева папка для згенерованих файлів: {settings['output_path']}")
        print("="*50 + "\n")

        choice = input("Введіть 'так' для запуску, 'змінити' для повторного налаштування, або 'вихід' для завершення: ").lower().strip()

        if choice == 'так':
            generate_obsidian_docs(settings)
            break
        elif choice == 'змінити':
            settings = None
            print("\nПерезапуск налаштувань...\n")
        elif choice == 'вихід':
            print("Роботу програми завершено.")
            break
        else:
            print("  [Помилка] Невідома команда. Спробуйте ще раз.")

def get_qualified_name(base_path: Path, file_path: Path, symbol_name: str, parent_class_name: str = None) -> str:
    """
    Генерує повне кваліфіковане ім'я символу.
    Наприклад: 'utils.my_module.MyClass' або 'utils.my_module.MyClass.my_method'.
    """
    relative_parts = file_path.relative_to(base_path).parts
    module_name_parts = list(relative_parts)

    # Remove .py suffix
    if module_name_parts[-1].endswith('.py'):
        module_name_parts[-1] = module_name_parts[-1][:-3]
    
    # Handle __init__.py for package modules
    if module_name_parts[-1] == '__init__':
        module_name_parts.pop() # Remove __init__

    full_module_name = ".".join(module_name_parts)

    if parent_class_name:
        return f"{full_module_name}.{parent_class_name}.{symbol_name}"
    return f"{full_module_name}.{symbol_name}"

def collect_project_symbols(project_path: Path, excluded_dirs: Set[str]) -> Dict[str, Any]:
    """
    Сканує весь проєкт, щоб зібрати інформацію про всі модулі, класи та функції,
    формуючи повну карту символів для посилань.

    Повертає словник:
    {
        "all_modules_info": { # Інформація про кожен модуль
            "full.module.name": {
                "file_path": Path,
                "classes": {"ClassName": file_path},
                "functions": {"FunctionName": file_path}
            }
        },
        "global_symbol_map": { # Карта кваліфікованих імен символів до їх деталей
            "full.module.name.SymbolName": {
                "type": "class/function/method",
                "file_path": Path_to_file,
                "full_module": "full.module.name",
                "simple_name": "SymbolName",
                "parent_class": "ClassName" (якщо метод),
                "obsidian_note_filename": "Filename_without_md_suffix",
                "obsidian_module_folder_path": "path/to/module/folder/relative/to/output_root"
            }
        },
        "file_to_module_map": { # Карта шляхів до файлів до їхніх повних імен модулів
             Path_to_file: "full.module.name"
        }
    }
    """
    all_modules_info: Dict[str, Dict[str, Any]] = {}
    global_symbol_map: Dict[str, Dict[str, Any]] = {}
    file_to_module_map: Dict[Path, str] = {}

    for root_str, dirs, files in os.walk(project_path, topdown=True):
        root = Path(root_str)
        dirs[:] = [d for d in dirs if d not in excluded_dirs and not d.startswith('.') and d != '__pycache__']

        for filename in files:
            if filename.endswith('.py'):
                file_path = root / filename
                relative_file_path = file_path.relative_to(project_path)
                
                # Обчислюємо повне ім'я модуля (наприклад, 'utils.data_handler')
                module_name_parts = list(relative_file_path.parts)
                if module_name_parts[-1].endswith('.py'):
                    module_name_parts[-1] = module_name_parts[-1][:-3]
                
                if module_name_parts[-1] == '__init__':
                    full_module_name = ".".join(module_name_parts[:-1]) 
                else:
                    full_module_name = ".".join(module_name_parts)

                file_to_module_map[file_path] = full_module_name

                analysis_result = analyze_python_file(file_path)
                if analysis_result:
                    module_info = {
                        "file_path": file_path,
                        "classes": {cls_name: file_path for cls_name in analysis_result['classes'].keys()},
                        "functions": {func_name: file_path for func_name in analysis_result['functions']}
                    }
                    all_modules_info[full_module_name] = module_info

                    # Calculate the path for the module's documentation folder relative to the output_path concept
                    # This will be like 'utils/my_module'
                    module_relative_path_in_docs = (relative_file_path.parent / module_name_parts[-1].replace('.', '_')).as_posix()

                    # Додаємо класи до глобальної карти символів
                    for cls_name in analysis_result['classes'].keys():
                        qualified_cls_name = get_qualified_name(project_path, file_path, cls_name)
                        global_symbol_map[qualified_cls_name] = {
                            "type": "class",
                            "file_path": file_path,
                            "full_module": full_module_name,
                            "simple_name": cls_name,
                            "obsidian_note_filename": f"{cls_name}",
                            "obsidian_module_folder_path": module_relative_path_in_docs
                        }
                        # Додаємо методи до глобальної карти символів
                        for method_name in analysis_result['classes'][cls_name]:
                            qualified_method_name = get_qualified_name(project_path, file_path, method_name, cls_name)
                            global_symbol_map[qualified_method_name] = {
                                "type": "method",
                                "file_path": file_path,
                                "full_module": full_module_name,
                                "simple_name": method_name,
                                "parent_class": cls_name,
                                "obsidian_note_filename": f"{cls_name}_{method_name}",
                                "obsidian_module_folder_path": module_relative_path_in_docs
                            }

                    # Додаємо функції верхнього рівня до глобальної карти символів
                    for func_name in analysis_result['functions']:
                        qualified_func_name = get_qualified_name(project_path, file_path, func_name)
                        global_symbol_map[qualified_func_name] = {
                            "type": "function",
                            "file_path": file_path,
                            "full_module": full_module_name,
                            "simple_name": func_name,
                            "obsidian_note_filename": f"{func_name}",
                            "obsidian_module_folder_path": module_relative_path_in_docs
                        }
    
    return {
        "all_modules_info": all_modules_info,
        "global_symbol_map": global_symbol_map,
        "file_to_module_map": file_to_module_map
    }

class UsageCollector(ast.NodeVisitor):
    """
    Збирає інформацію про використання символів по всьому проєкту.
    """
    def __init__(self, global_symbol_map: Dict[str, Any], file_to_module_map: Dict[Path, str]):
        self.global_symbol_map = global_symbol_map
        self.file_to_module_map = file_to_module_map
        self.project_usages: Dict[str, List[Dict[str, Any]]] = {
            q_name: [] for q_name in global_symbol_map.keys()
        } # {qualified_symbol_name: [{file_path: Path, line: int, snippet: str}]}
        self.current_file_path: Path = Path()
        self.current_file_source_lines: List[str] = []
        self.current_module_name: str = ""
        self.current_imports_map: Dict[str, str] = {} # {alias: full_qualified_original_name}

    def process_file(self, file_path: Path, source_code: str, analysis_result: Dict[str, Any]):
        self.current_file_path = file_path
        self.current_file_source_lines = source_code.splitlines()
        self.current_module_name = self.file_to_module_map.get(file_path, "")
        self.current_imports_map = {}

        # Build local import map (alias -> original qualified name)
        for imp_detail in analysis_result['imports']:
            if imp_detail[0] == "module": # import module_name [as alias]
                original_module_name = imp_detail[1]
                alias_name = imp_detail[3]
                # Check if the imported module is one of our globally indexed modules
                if original_module_name in self.global_symbol_map: # Check if the module itself is a known symbol (e.g., if it's an __init__.py)
                    self.current_imports_map[alias_name] = original_module_name # direct module import
                else: # Try to find its full qualified name from global map if it's not a module but a symbol with same name
                    for q_name, info in self.global_symbol_map.items():
                        if info['simple_name'] == original_module_name and info['type'] in ['class', 'function']:
                            self.current_imports_map[alias_name] = q_name
                            break
            
            elif imp_detail[0] == "from": # from module_name import symbol_name [as alias]
                imported_symbol_original_name = imp_detail[2]
                imported_symbol_alias = imp_detail[3]
                potential_qualified_name_direct = f"{imp_detail[1]}.{imported_symbol_original_name}"
                potential_qualified_name_method = f"{imp_detail[1]}.{imported_symbol_original_name}" # This needs to be more robust for methods

                found_q_name = None
                # Try direct match
                if potential_qualified_name_direct in self.global_symbol_map:
                    found_q_name = potential_qualified_name_direct
                else:
                    # Try to find if it's a method imported (less reliable via simple string match)
                    for q_name, info in self.global_symbol_map.items():
                        if info.get('type') == 'method' and info['simple_name'] == imported_symbol_original_name and q_name.startswith(f"{imp_detail[1]}."):
                            found_q_name = q_name
                            break
                
                if found_q_name:
                    self.current_imports_map[imported_symbol_alias] = found_q_name
                else: # Fallback: try to find it by simple name
                    for q_name, info in self.global_symbol_map.items():
                        if info['simple_name'] == imported_symbol_original_name:
                            self.current_imports_map[imported_symbol_alias] = q_name
                            break

        tree = ast.parse(source_code, filename=file_path.name)
        self.visit(tree)

    def _record_usage(self, qualified_symbol_name: str, node: ast.AST):
        """Записує використання символу."""
        if qualified_symbol_name not in self.project_usages:
            self.project_usages[qualified_symbol_name] = [] 
        
        line_num = getattr(node, 'lineno', 0) # Use getattr for robustness
        snippet = ""
        if 0 <= line_num - 1 < len(self.current_file_source_lines):
            snippet = self.current_file_source_lines[line_num - 1].strip()

        self.project_usages[qualified_symbol_name].append({
            "file_path": self.current_file_path,
            "line": line_num,
            "snippet": snippet
        })

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            # Direct function call or class instantiation (e.g., `my_function()`, `MyClass()`)
            called_name = node.func.id
            
            # Check current module's symbols first
            qualified_name_in_current_module = get_qualified_name(self.current_file_path.parent.parent, self.current_file_path, called_name) # Need project_path here
            # Let's simplify and rely on current_imports_map or global_symbol_map directly

            resolved_q_name = None
            if called_name in self.current_imports_map:
                resolved_q_name = self.current_imports_map[called_name]
            else: # Check if it's a direct symbol in the current module
                potential_q_name = f"{self.current_module_name}.{called_name}"
                if potential_q_name in self.global_symbol_map and \
                   self.global_symbol_map[potential_q_name]['type'] in ['function', 'class'] and \
                   self.global_symbol_map[potential_q_name]['file_path'] == self.current_file_path:
                    resolved_q_name = potential_q_name
            
            if resolved_q_name and resolved_q_name in self.global_symbol_map:
                self._record_usage(resolved_q_name, node)

        elif isinstance(node.func, ast.Attribute):
            # Method call (e.g., `obj.my_method()`, `MyClass.static_method()`)
            method_name = node.func.attr
            
            # Try to resolve the attribute to a known method in global map
            # This is a heuristic and might pick the first method with that name.
            for q_name, info in self.global_symbol_map.items():
                if info.get('type') == 'method' and info.get('simple_name') == method_name:
                    self._record_usage(q_name, node)
                    break 
        
        self.generic_visit(node) # Continue visiting child nodes

    def visit_Name(self, node: ast.Name):
        # Handle references to classes/functions (e.g., `isinstance(obj, MyClass)`, `CLASS_CONSTANT`)
        # Only process if it's a Load context (being read/used, not assigned)
        if isinstance(node.ctx, ast.Load):
            simple_name = node.id
            
            resolved_q_name = None
            if simple_name in self.current_imports_map:
                resolved_q_name = self.current_imports_map[simple_name]
            else: # Check if it's a direct symbol in the current module
                potential_q_name = f"{self.current_module_name}.{simple_name}"
                if potential_q_name in self.global_symbol_map and \
                   self.global_symbol_map[potential_q_name]['type'] in ['function', 'class'] and \
                   self.global_symbol_map[potential_q_name]['file_path'] == self.current_file_path:
                    resolved_q_name = potential_q_name
            
            if resolved_q_name and resolved_q_name in self.global_symbol_map:
                self._record_usage(resolved_q_name, node)

        self.generic_visit(node)

def generate_obsidian_docs(settings: Dict[str, Any]) -> None:
    """
    Основна функція, яка сканує проєкт та генерує документацію.
    """
    project_path = settings['project_path']
    output_path = settings['output_path']
    excluded_dirs = settings['excluded_dirs']

    print(f"\nПочинаю генерацію документації в: {output_path}")

    # Очищуємо/створюємо вихідну директорію
    if output_path.exists():
        print(f"  Очищення існуючої директорії: {output_path}")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # NEW: Збираємо всі символи проєкту
    project_symbols_data = collect_project_symbols(project_path, excluded_dirs)
    all_modules_info = project_symbols_data["all_modules_info"]
    global_symbol_map = project_symbols_data["global_symbol_map"]
    file_to_module_map = project_symbols_data["file_to_module_map"]
    print(f"  Зібрано символи з {len(all_modules_info)} модулів та {len(global_symbol_map)} унікальних символів.")

    # NEW: Збираємо всі використання символів
    usage_collector = UsageCollector(global_symbol_map, file_to_module_map)
    for root_str, dirs, files in os.walk(project_path, topdown=True):
        root = Path(root_str)
        dirs[:] = [d for d in dirs if d not in excluded_dirs and not d.startswith('.') and d != '__pycache__']
        for filename in files:
            if filename.endswith('.py'):
                file_path = root / filename
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        source_code = f.read()
                    analysis_result = analyze_python_file(file_path)
                    if analysis_result:
                        usage_collector.process_file(file_path, source_code, analysis_result)
                except Exception as e:
                    print(f"  [Помилка] Не вдалося проаналізувати використання у файлі {file_path}: {e}")
    project_usages = usage_collector.project_usages
    print(f"  Зібрано інформацію про використання символів.")


    # Генеруємо Markdown-файли
    for root_str, dirs, files in os.walk(project_path, topdown=True):
        root = Path(root_str)
        dirs[:] = [d for d in dirs if d not in excluded_dirs and not d.startswith('.') and d != '__pycache__']

        for filename in files:
            if filename.endswith('.py'):
                file_path = root / filename
                relative_path = file_path.relative_to(project_path)
                
                analysis_result = analyze_python_file(file_path)
                if not analysis_result:
                    continue
                
                # Обчислюємо повне ім'я поточного модуля
                current_full_module_name = file_to_module_map.get(file_path, "")

                # Створюємо окрему папку для документації цього файлу
                module_doc_folder_name = relative_path.with_suffix('').name # e.g., 'DataHandler' for DataHandler.py
                # Replace dots in folder names if they come from nested packages (e.g., 'my_package.my_module' becomes 'my_package_my_module')
                module_doc_folder_name = module_doc_folder_name.replace('.', '_') 
                obsidian_file_parent_dir = output_path / relative_path.parent / module_doc_folder_name
                obsidian_file_parent_dir.mkdir(parents=True, exist_ok=True)
                
                # --- Генеруємо головний файл модуля (index.md) ---
                md_content_module_index = f"# Модуль: {relative_path.name}\n\n"
                md_content_module_index += f"**Повний шлях:** `{relative_path.as_posix()}`\n\n"
                md_content_module_index += "---\n\n"

                # Імпорти в головному файлі модуля
                if analysis_result['imports']:
                    md_content_module_index += "## Імпорти та залежності\n"
                    for imp_detail in analysis_result['imports']:
                        imported_full_name = imp_detail[1] # For 'module' type, this is module name. For 'from', this is source module.
                        imported_symbol_simple_name = imp_detail[2] # For 'from' type, this is the symbol imported
                        display_name = imp_detail[3]
                            
                        found_link = False
                        # Try to link to a specific symbol (class/function/method)
                        for q_name, info in global_symbol_map.items():
                            # Check if the imported symbol's simple name matches and it belongs to the imported module
                            if info['simple_name'] == imported_symbol_simple_name and \
                                (info['full_module'] == imported_full_name or \
                                 (info['type'] == 'method' and f"{imported_full_name}.{info['parent_class']}" in info['full_module'])):
                                
                                link_target = f"{info['obsidian_module_folder_path']}/{info['obsidian_note_filename']}"
                                md_content_module_index += f"- [[{link_target}|{display_name}]]\n"
                                found_link = True
                                break
                        
                        # Fallback: if not a specific symbol, try to link to the module's index if it's part of the project
                        if not found_link and imported_full_name in all_modules_info:
                            module_file_path = all_modules_info[imported_full_name]['file_path']
                            # Calculate its doc folder path
                            module_rel_path_in_docs = (module_file_path.relative_to(project_path).parent / module_file_path.with_suffix('').name.replace('.', '_')).as_posix()
                            link_target = f"{module_rel_path_in_docs}/index"
                            md_content_module_index += f"- [[{link_target}|{display_name}]]\n"
                            found_link = True

                        if not found_link:
                            md_content_module_index += f"- `{display_name}`\n"
                    md_content_module_index += "\n"

                # Класи в головному файлі модуля (лише посилання)
                if analysis_result['classes']:
                    md_content_module_index += "## Класи\n"
                    for class_name in sorted(analysis_result['classes'].keys()):
                        qualified_cls_name = get_qualified_name(project_path, file_path, class_name)
                        if qualified_cls_name in global_symbol_map:
                            info = global_symbol_map[qualified_cls_name]
                            link_target = f"{info['obsidian_module_folder_path']}/{info['obsidian_note_filename']}"
                            md_content_module_index += f"- [[{link_target}|`class {class_name}`]]\n"
                    md_content_module_index += "\n"
                
                # Окремі функції в головному файлі модуля (лише посилання)
                if analysis_result['functions']:
                    md_content_module_index += "## Окремі функції\n"
                    for func_name in sorted(analysis_result['functions']):
                        qualified_func_name = get_qualified_name(project_path, file_path, func_name)
                        if qualified_func_name in global_symbol_map:
                            info = global_symbol_map[qualified_func_name]
                            link_target = f"{info['obsidian_module_folder_path']}/{info['obsidian_note_filename']}"
                            md_content_module_index += f"- [[{link_target}|`{func_name}()`]]\n"
                    md_content_module_index += "\n"

                # Додаємо вихідний код модуля
                original_source_code = ""
                try:
                    with open(file_path, 'r', encoding='utf-8') as f_orig:
                        original_source_code = f_orig.read()
                except Exception as e:
                    print(f"  [Помилка] Не вдалося прочитати вихідний код файлу {file_path}: {e}")

                md_content_module_index += "## Вихідний код\n"
                md_content_module_index += "```python\n"
                md_content_module_index += original_source_code
                md_content_module_index += "\n```\n\n"

                md_content_module_index += "tags: #python #module\n"
                with open(obsidian_file_parent_dir / "index.md", 'w', encoding='utf-8') as md_file:
                    md_file.write(md_content_module_index)

                # --- Генеруємо окремі файли для Класів ---
                for class_name, methods in analysis_result['classes'].items():
                    qualified_cls_name = get_qualified_name(project_path, file_path, class_name)
                    cls_info = global_symbol_map.get(qualified_cls_name, {})

                    md_content_class = f"# Клас: `{class_name}`\n\n"
                    # Посилання на модуль, до якого належить клас
                    module_link_target = f"{cls_info.get('obsidian_module_folder_path')}/index"
                    md_content_class += f"**Входить до модуля:** [[{module_link_target}|{relative_path.name}]]\n\n"
                    md_content_class += "---\n\n"

                    md_content_class += "## Методи\n"
                    if methods:
                        for method_name in sorted(methods):
                            qualified_method_name = get_qualified_name(project_path, file_path, method_name, class_name)
                            method_info = global_symbol_map.get(qualified_method_name, {})
                            link_target = f"{method_info.get('obsidian_module_folder_path')}/{method_info.get('obsidian_note_filename')}"
                            md_content_class += f"- [[{link_target}|`{method_name}()`]]\n"
                    else:
                        md_content_class += "Методів не знайдено.\n"
                    md_content_class += "\n"

                    # Додаємо використання класу
                    if qualified_cls_name in project_usages and project_usages[qualified_cls_name]:
                        md_content_class += "## Використання цього класу\n"
                        for usage in project_usages[qualified_cls_name]:
                            # Link to the module's index file where the usage occurs
                            usage_module_folder_path = (usage['file_path'].relative_to(project_path).parent / usage['file_path'].with_suffix('').name.replace('.', '_')).as_posix()
                            usage_file_link = f"{usage_module_folder_path}/index"
                            md_content_class += f"- [[{usage_file_link}|{usage['file_path'].name}]] (рядок {usage['line']}): `{usage['snippet']}`\n"
                        md_content_class += "\n"
                    
                    md_content_class += "tags: #python #class\n"
                    with open(obsidian_file_parent_dir / f"{class_name}.md", 'w', encoding='utf-8') as md_file:
                        md_file.write(md_content_class)

                    # --- Генеруємо окремі файли для Методів ---
                    for method_name in methods:
                        qualified_method_name = get_qualified_name(project_path, file_path, method_name, class_name)
                        method_info = global_symbol_map.get(qualified_method_name, {})

                        md_content_method = f"# Метод: `{method_name}()`\n\n"
                        # Посилання на клас, до якого належить метод
                        class_link_target = f"{cls_info.get('obsidian_module_folder_path')}/{cls_info.get('obsidian_note_filename')}"
                        md_content_method += f"**Належить класу:** [[{class_link_target}|`{class_name}`]]\n"
                        # Посилання на модуль, до якого належить метод
                        module_link_target = f"{method_info.get('obsidian_module_folder_path')}/index"
                        md_content_method += f"**Входить до модуля:** [[{module_link_target}|{relative_path.name}]]\n\n"
                        md_content_method += "---\n\n"

                        # Додаємо використання методу
                        if qualified_method_name in project_usages and project_usages[qualified_method_name]:
                            md_content_method += "## Використання цього методу\n"
                            for usage in project_usages[qualified_method_name]:
                                # Link to the module's index file where the usage occurs
                                usage_module_folder_path = (usage['file_path'].relative_to(project_path).parent / usage['file_path'].with_suffix('').name.replace('.', '_')).as_posix()
                                usage_file_link = f"{usage_module_folder_path}/index"
                                md_content_method += f"- [[{usage_file_link}|{usage['file_path'].name}]] (рядок {usage['line']}): `{usage['snippet']}`\n"
                            md_content_method += "\n"
                        
                        md_content_method += "tags: #python #method\n"
                        with open(obsidian_file_parent_dir / f"{class_name}_{method_name}.md", 'w', encoding='utf-8') as md_file:
                            md_file.write(md_content_method)

                # --- Генеруємо окремі файли для Окремих Функцій ---
                for func_name in analysis_result['functions']:
                    qualified_func_name = get_qualified_name(project_path, file_path, func_name)
                    func_info = global_symbol_map.get(qualified_func_name, {})

                    md_content_func = f"# Функція: `{func_name}()`\n\n"
                    # Посилання на модуль, до якого належить функція
                    module_link_target = f"{func_info.get('obsidian_module_folder_path')}/index"
                    md_content_func += f"**Входить до модуля:** [[{module_link_target}|{relative_path.name}]]\n\n"
                    md_content_func += "---\n\n"

                    # Додаємо використання функції
                    if qualified_func_name in project_usages and project_usages[qualified_func_name]:
                        md_content_func += "## Використання цієї функції\n"
                        for usage in project_usages[qualified_func_name]:
                            # Link to the module's index file where the usage occurs
                            usage_module_folder_path = (usage['file_path'].relative_to(project_path).parent / usage['file_path'].with_suffix('').name.replace('.', '_')).as_posix()
                            usage_file_link = f"{usage_module_folder_path}/index"
                            md_content_func += f"- [[{usage_file_link}|{usage['file_path'].name}]] (рядок {usage['line']}): `{usage['snippet']}`\n"
                        md_content_func += "\n"
                    
                    md_content_func += "tags: #python #function\n"
                    with open(obsidian_file_parent_dir / f"{func_name}.md", 'w', encoding='utf-8') as md_file:
                        md_file.write(md_content_func)
            else:
                print(f"  Пропускаю файл (не Python): {filename}")


    print("\n" + "*"*25)
    print("Успіх! Генерацію документації завершено.")
    print(f"Тепер ви можете відкрити ваше сховище в Obsidian і побачити нові нотатки.")
    print("*"*25)


if __name__ == "__main__":
    confirm_and_run()
