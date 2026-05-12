import json
from pathlib import Path

from Scintillator_Project.src.data_parse.dat_file_reader import DatFileReader


def check_samples(data_file: str):
    processed_dir = Path(data_file)
    json_files = sorted(processed_dir.glob('*.json'), key=lambda x: int(x.stem.split('_')[1]))
    print(f"验证所有JSON文件中的样本数...\n")
    print("=" * 70)

    all_valid = True

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)

        filename = data['filename']
        num_events = len(data['events'])
        print(f"\n📄 {filename} ({num_events} events):")
        # print('这是每个文件的CH0,CH1通道的总样本数')
        print("-" * 70)

        for event in data['events']:
            event_id = event['event_id']
            # 检查每个通道的样本数
            ch0_count = len(event['CH0'])
            ch1_count = len(event['CH1'])
            ch2_count = len(event['CH2'])
            ch3_count = len(event['CH3'])

            # 验证
            if ch0_count == 1024 and ch1_count == 1024 and ch2_count == 1024 and ch3_count == 1024:
            # if ch0_count == 1024 and ch1_count == 1024:
                status = "✅"
            else:
                status = "❌"
                all_valid = False
                print(f"  Event {event_id}: {status} CH0:{ch0_count} CH1:{ch1_count} CH2:{ch2_count} CH3:{ch3_count}")
    print("\n" + "=" * 70)
    if all_valid:
        print("✅ 验证通过！所有通道都是1024个样本")
    else:
        print("❌ 发现问题！详见上面标记的❌")


def check_single_sample(dat_file: str):
    """
    检查单个文件
    :param dat_file:
    :return:
    """
    processed_dir = Path(dat_file)
    with open(processed_dir, 'r') as f:
        data = json.load(f)

    filename = data['filename']
    position_label = data['position_label']
    num_events = len(data['events'])
    print(f"文件: {filename}")
    print(f"位置标签: {position_label} cm")
    print(f"总事件数: {num_events}")
    print("=" * 80)
    print(f"{'Event ID':<12} {'CH0':<10} {'CH1':<10} {'CH2':<10} {'CH3':<10} {'状态':<10}")
    print("=" * 80)

    issues = []
    for event in data['events']:
        event_id = event['event_id']

        for ch_name in ['CH0', 'CH1', 'CH2', 'CH3']:
            sample_count = len(event[ch_name])
            if sample_count != 1024:
                issue = f"{filename}, 第{event_id}个事件, {ch_name}通道, {sample_count}个样本（期望1024）"
                issues.append(issue)

    if issues:
        print(f"❌ 发现 {len(issues)} 个异常：\n")
        for issue in issues:
            print(f"  {issue}")
    else:
        print(f"✅ 所有事件所有通道都符合（1024个样本）")

    print("=" * 80)


#     """
#      测试单条数据集dat转json并保存
#     """
#     filepath = '../../dataset/raw/run7_65.dat'
#     output_json = '../../dataset/processed/run7_65.json'
#     reader = DatFileReader(filename=filepath)
#     reader.save_to_json(output_json)
def dat_to_json_and_save(data_file: str, output_dir: str):
    reader = DatFileReader(filename=data_file)
    reader.save_to_json(output_dir)

def check_CH0_CH1_sample(data_file: str):
    processed_dir = Path(data_file)
    json_files = sorted(processed_dir.glob('*.json'), key=lambda x: int(x.stem.split('_')[1]))
    print(f"验证所有JSON文件中的样本数...\n")
    print("=" * 70)

    all_valid = True

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)

        filename = data['filename']
        num_events = len(data['events'])
        print(f"\n📄 {filename} ({num_events} events):")
        # print('这是每个文件的CH0,CH1通道的总样本数')
        print("-" * 70)

        for event in data['events']:
            event_id = event['event_id']
            # 检查每个通道的样本数
            ch0_count = len(event['CH0'])
            ch1_count = len(event['CH1'])

            # 验证
            if ch0_count == 1024 and ch1_count == 1024:
                status = "✅"
            else:
                status = "❌"
                all_valid = False
                print(f"  Event {event_id}: {status} CH0:{ch0_count} CH1:{ch1_count}")
    print("\n" + "=" * 70)
    if all_valid:
        print("✅ 验证通过！所有通道都是1024个样本")
    else:
        print("❌ 发现问题！详见上面标记的❌")



if __name__ == "__main__":
    # dat_to_json_and_save(data_file='../../dataset/raw/run7_65.dat', output_dir='../../dataset/processed/test_run7_65.json')
    #检查所有数据集是否全部符合1024个采样
    # check_samples('../../dataset/processed/')


    # 检查单条数据集是否是1024个采样
    # print(f'{'='*10}下面是单条数据集有问题{'='*10}')
    # check_single_sample("../../dataset/processed/run11_35.json")


    # 提取Scintillator_Project/dataset/processed/中的json数据，只保留CH0, CH1
    # DatFileReader.save_ch0_and_ch1_to_json(input_dir='../../dataset/processed')

    check_CH0_CH1_sample("../../dataset/processed/processed_ch0_and_ch1_to_json")

    # check_single_sample("../../dataset/processed/processed_ch0_and_ch1_to_json/run7_65.json")
