# Step 1.3: 数据集分割（Stratified Split）
import pickle
import random
import json
from pathlib import Path
import GAPS_Project

PROJECT_ROOT = Path(GAPS_Project.__file__).parent


def load_events_from_dir(pkl_dir: Path) -> list:
    """从目录下所有pkl文件加载event列表"""
    events = []
    for pkl_file in sorted(pkl_dir.glob('*.pkl')):
        with open(pkl_file, 'rb') as f:
            payload = pickle.load(f)
        events.extend(payload['events'])
        print(f"  加载: {pkl_file.name} → {len(payload['events'])} events")
    return events


def split_events(events: list, train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42) -> tuple:
    """对单类events做随机分割"""
    random.seed(seed)
    shuffled = events.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return train, val, test


def save_split(events: list, output_path: Path, split_name: str):
    """保存split pkl + summary"""
    label_counts = {}
    for e in events:
        key = str(e['key'])
        label_counts[key] = label_counts.get(key, 0) + 1

    with open(output_path, 'wb') as f:
        pickle.dump({
            'events': events,
            'split': split_name,
            'label_counts': label_counts,
        }, f)

    # summary
    summary_path = output_path.with_suffix("").with_name(output_path.stem + '_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'split': split_name,
            'total_events': len(events),
            'label_counts': label_counts,
        }, f, indent=2, ensure_ascii=False)

    print(f'{split_name:5s}: {len(events):5d} events → {output_path.name}')


def make_split(train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42):
    """
    主函数：加载antiD + antiP, stratified split → train/val/test.pkl
    """
    processed_dir = PROJECT_ROOT / 'dataset' / 'processed'
    split_dir = PROJECT_ROOT / 'dataset' / 'split'
    split_dir.mkdir(parents=True, exist_ok=True)

    # ── 加载两类数据 ──────────────────────────────────────
    print('加载反重氘核（antiD）...')
    antid_events = load_events_from_dir(processed_dir / 'antiD')

    print('\n加载反质子（antiP）...')
    antip_events = load_events_from_dir(processed_dir / 'antiP')

    print(f'\n总计：antiD={len(antid_events)}, antiP={len(antip_events)}')
    print(f'类别比例：antiP:antiD = 1:{len(antid_events) / len(antip_events):.2f}')

    # ── 各类分别 stratified split ─────────────────────────
    antid_train, antid_val, antid_test = split_events(antid_events, train_ratio, val_ratio, seed)
    antip_train, antip_val, antip_test = split_events(antip_events, train_ratio, val_ratio, seed)

    # ── 合并 + 打乱 ───────────────────────────────────────
    random.seed(seed)

    train = antid_train + antip_train
    val = antid_val + antip_val
    test = antid_test + antip_test

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    # ── 保存 ──────────────────────────────────────────────
    print('\n保存split结果：')
    save_split(train, split_dir / 'train.pkl', 'train')
    save_split(val, split_dir / 'val.pkl', 'val')
    save_split(test, split_dir / 'test.pkl', 'test')
    print(f'\n完成 → {split_dir}')

if __name__ == '__main__':
    make_split(train_ratio=0.7, val_ratio=0.15, seed=42)

