import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

import re
_NUM_RE = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$')

def _coerce(v: str):
    s = v.strip()
    l = s.lower()
    # bool
    if l in ("true", "false"):
        return l == "true"
    # чисте число (int/float). Не чіпаємо значення типу "2m", "M5", "1h"
    if _NUM_RE.match(s):
        f = float(s)
        return int(f) if f.is_integer() else f
    return s

@dataclass
class SkillConfig:
    name: str
    on: bool
    params: Dict[str, str]

def parse_manifest(manifest_path: str) -> Tuple[List[SkillConfig], Dict[str, str], Dict[str, str]]:
    skills: List[SkillConfig] = []
    multi_tf_cfg: Dict[str, str] | None = None
    tf_flags: Dict[str, str] = {}
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [p.strip() for p in line.split('|')]
            if len(parts) < 2:
                continue
            name = parts[0]
            on = parts[1].lower() == 'on'
            params: Dict[str, str] = {}
            if len(parts) >= 3:
                for kv in parts[2].split(','):
                    kv = kv.strip()
                    if not kv:
                        continue
                    if '=' in kv:
                        k, v = kv.split('=', 1)
                        params[k.strip()] = _coerce(v)
            if name == 'multi_tf':
                multi_tf_cfg = {'on': 'on' if on else 'off'}
                multi_tf_cfg.update(params)
            elif name == 'tf_flags':
                tf_flags = params
            else:
                skills.append(SkillConfig(name=name, on=on, params=params))
    return skills, (multi_tf_cfg or {}), tf_flags

def _apply_skill(df: pd.DataFrame, skill: SkillConfig) -> pd.DataFrame:
    module = importlib.import_module(f".{skill.name}.skill", package=__package__)
    original_cols = set(df.columns)
    df = module.transform(df.copy(), **skill.params)
    new_cols = [c for c in df.columns if c not in original_cols]
    for col in new_cols:
        on_col = f"{col}_on"
        if skill.on:
            df[on_col] = 1.0
        else:
            df[col] = 0.0
            df[on_col] = 0.0
    return df

DEFAULT_MANIFEST = str(Path(__file__).resolve().parent / "skills_manifest.txt")

def apply_skills(df: pd.DataFrame, manifest_path: str = DEFAULT_MANIFEST) -> pd.DataFrame:
    skills, _, tf_flags = parse_manifest(manifest_path)
    out_df = df.copy()
    for sc in skills:
        out_df = _apply_skill(out_df, sc)
    for k, v in tf_flags.items():
        val = _coerce(v)
        out_df[k] = 1.0 if val is True else (0.0 if val is False else float(val))
    return out_df

def apply_skills_multi(dfs_by_tf: Dict[str, pd.DataFrame], base_tf: str,
                       manifest_path: str = DEFAULT_MANIFEST) -> pd.DataFrame:
    skills, multi_cfg, tf_flags = parse_manifest(manifest_path)
    if multi_cfg.get('on', 'off') != 'on':
        raise ValueError("multi_tf must be enabled in manifest for apply_skills_multi")
    tfs = [tf.strip() for tf in multi_cfg.get('tfs', '').split(',') if tf.strip()]
    join = multi_cfg.get('join', 'asof')
    tolerance = multi_cfg.get('tolerance')
    processed: Dict[str, pd.DataFrame] = {}
    for tf in tfs:
        df_tf = dfs_by_tf[tf].copy()
        for sc in skills:
            df_tf = _apply_skill(df_tf, sc)
        processed[tf] = df_tf.add_prefix(f"{tf}_")
    result = processed[base_tf]
    for tf, df_tf in processed.items():
        if tf == base_tf:
            continue
        if join == 'asof':
            result = pd.merge_asof(result.sort_index(), df_tf.sort_index(),
                                   left_index=True, right_index=True,
                                   tolerance=pd.to_timedelta(tolerance) if tolerance else None)
        else:
            result = result.join(df_tf, how='left')
    for k, v in tf_flags.items():
        result[k] = float(v)
    return result