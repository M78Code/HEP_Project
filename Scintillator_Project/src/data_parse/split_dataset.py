"""
数据分割，按位置分层，将数据集分割为train/val/test
比例：70% / 15% 15%
"""
import json
import random
from pathlib import Path

# 固定随机种子，保证每次分割结果一致
random.seed(42)# 作用一句话：让随机结果“可复现”每次打乱的结果 完全一样
#train / val / test 的划分 每次都一致

# 路径
input_dir = Path('../../dataset/processed/processed_ch0_and_ch1_to_json')
output_dir = Path('../../dataset/split')
output_dir.mkdir(parents=True, exist_ok=True)

# 分割比例
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 读取所有JSON文件
json_files = sorted(input_dir.glob('*.json'), key=lambda x: int(x.stem.split('_')[1]))

train_events = []
val_events = []
test_events = []

print(f'开始按位置分层分割...\n')
print('='*70)
print(f"{'文件':<20} {'位置(cm)':<10} {'总数':<8} {'Train':<8} {'Val':<8} {'Test':<8}")
print("="*70)

for json_file in json_files:
    with open(json_file, 'r') as f:
        data = json.load(f)

    filename = data['filename']
    position = data['position_label']
    events = data['events']
    n = len(events)

    # 随机打乱（固定种子）
    random.shuffle(events)

    # 计算分割点
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_val - n_train # 剩余全归test

    # 分割
    split_train = events[:n_train]
    split_val = events[n_train: n_train+n_val]
    split_test = events[n_train+n_val:]

    # 为每个event加上position_label（按“位置”分层的数据集）
    for e in split_train:
        e['position_label'] = position
    for e in split_val:
        e['position_label'] = position
    for e in split_test:
        e['position_label'] = position

    train_events.extend(split_train)
    val_events.extend(split_val)
    test_events.extend(split_test)

    print(f"{filename:<20} {position:<10.0f} {n:<8} {len(split_train):<8} {len(split_val):<8} {len(split_test):<8}")

print("="*70)
total = len(train_events) + len(val_events) + len(test_events)
print(f"{'合计':<20} {'':<10} {total:<8} {len(train_events):<8} {len(val_events):<8} {len(test_events):<8}")

# 打乱合并后的数据 eg: 打乱前是[10cm, 10cm, 10cm, 20cm, 20cm]，打乱后是[20cm, 10cm, 20cm, 10cm, 10cm]
random.shuffle(train_events)
random.shuffle(val_events)
random.shuffle(test_events)

# 保存为JSON
splits = {
    'train': train_events,
    'val': val_events,
    'test': test_events
}

for split_name, events in splits.items():
    output = {
        'split': split_name,
        'num_events': len(events),
        'events': events
    }
    output_file = output_dir / f'{split_name}.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n✅ {split_name}.json 已保存（{len(events)} events）→ {output_file}")

print("\n" + "="*70)
print("✅ 数据分割完成！")
print(f"输出目录：{output_dir}")


"""
/opt/homebrew/anaconda3/bin/python /Users/lind/Desktop/ppt/GAPS_Project/Scintillator_Project/src/data_parse/split_dataset.py 
开始按位置分层分割...

======================================================================
文件                   位置(cm)     总数       Train    Val      Test    
======================================================================
run9_15.dat          15         6022     4215     903      904     
run10_25.dat         25         244      170      36       38      
run11_35.dat         35         251      175      37       39      
run12_45.dat         45         4509     3156     676      677     
run8_55.dat          55         301      210      45       46      
run7_65.dat          65         222      155      33       34      
run13_75.dat         75         318      222      47       49      
run20_80.dat         80         3828     2679     574      575     
run14_85.dat         85         284      198      42       44      
run19_95.dat         95         534      373      80       81      
run18_105.dat        105        390      273      58       59      
run17_115.dat        115        466      326      69       71      
run16_125.dat        125        4805     3363     720      722     
run15_135.dat        135        239      167      35       37      
run21_140.dat        140        1189     832      178      179     
======================================================================
合计                              23602    16514    3533     3555    

✅ train.json 已保存（16514 events）→ ../../dataset/split/train.json

✅ val.json 已保存（3533 events）→ ../../dataset/split/val.json

✅ test.json 已保存（3555 events）→ ../../dataset/split/test.json

======================================================================
✅ 数据分割完成！
输出目录：../../dataset/split

Process finished with exit code 0
"""

