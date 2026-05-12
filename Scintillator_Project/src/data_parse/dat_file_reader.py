import json
from pathlib import Path

class DatFileReader:
    """
    读取闪烁体.dat文件的工具

    文件结构：
    - 每个事件包含：Header + 4通道(CH0~CH3)的波形数据
    - CH0: 闪烁体左端；CH1: 闪烁体右端；位置推定只用CH0和CH1. CH2・CH3是触发用闪烁体的信号
    - 每通道：1024个样本
    """
    def __init__(self, filename):
        self.filename = filename
        self.extract_label()


    def extract_label(self):
        """从文件名提取位置标签"""
        import os
        basename = os.path.basename(self.filename)
        # run7_65.dat -> 65
        self.label = float(basename.split('_')[1].split('.')[0])


    def read_file(self):
        """
        读取整个.dat文件，返回所有events

        :return: list of dict，每个dict包含CH0, CH1, CH2, CH3波形和标签
            events: list of dict
                - 'CH0': [1024] 波形（闪烁体左端）位置推定用
                - 'CH1': [1024] 波形（闪烁体右端）位置推定用
                - 'CH2': [1024] 波形（触发用闪烁体的信号）
                - 'CH3': [1024] 波形（触发用闪烁体的信号）
                - 'label': 位置标签
        """
        events = []

        try:
            # 'rb'打开的是二进制格式文件，'r'打开的是文本文件
            with open(self.filename, 'r') as f:
                lines = f.readlines()

            current_event = {}
            current_channel = None
            current_event_id = None

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # 检测EVENT开始，提取event_id
                if '=== EVENT' in line:
                    # 保存前一个完整事件
                    import re
                    match = re.search(r'EVENT\s+(\d+)', line)
                    event_id = int(match.group(1)) if match else None

                    if current_event_id is not None and len(current_event) ==4:
                        event = {
                            'event_id': current_event_id,
                            'CH0': current_event['CH0'],
                            'CH1': current_event['CH1'],
                            'CH2': current_event['CH2'],
                            'CH3': current_event['CH3'],
                        }
                        events.append(event)

                    # 开始新event
                    current_event = {}
                    current_event_id = event_id
                    current_channel = None

                # 检测通道标记
                elif '=== CH:' in line:
                    if 'CH: 0' in line:
                        current_channel = "CH0"
                        current_event[current_channel] = []
                    elif 'CH: 1' in line:
                        current_channel = "CH1"
                        current_event[current_channel] = []
                    elif 'CH: 2' in line:
                        current_channel = "CH2"
                        current_event[current_channel] = []
                    elif 'CH: 3' in line:
                        current_channel = "CH3"
                        current_event[current_channel] = []
                # 解析波形数据（数值行）
                elif current_channel and not line.startswith('==='):
                    try:
                        values = list(map(float, line.split()))
                        current_event[current_channel].extend(values)
                    except ValueError:
                        pass
            # 保存最后一个事件
            if current_event_id is not None and len(current_event) ==4:
                event = {
                    'event_id': current_event_id,
                    'CH0': current_event['CH0'],
                    'CH1': current_event['CH1'],
                    'CH2': current_event['CH2'],
                    'CH3': current_event['CH3'],
                }
                events.append(event)
            print(f"✅ 成功读取 {len(events)} 个events")
            return events
        except Exception as e:
            print(f"⚠️ 读取失败: {e}")
            return []


    def save_to_json(self, output_path):
        """
        保存为JSON格式
        :param output_path:
        :return: json file
        """
        events = self.read_file()
        data = {
            'filename': Path(self.filename).name,
            'position_label': self.label,
            'position_unit': 'cm',
            'num_events': len(events),
            'samples_per_channel':1024,
            'events': events,
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ JSON已保存到: {output_path}")
        return data

    @staticmethod
    def save_ch0_and_ch1_to_json(input_dir, output_path = '../../dataset/processed/processed_ch0_and_ch1_to_json'):
        """

        :param output_path: 处理后的输出路径
        :param input_dir:
            只保留通道CH0，CH1
        :return:    将第一步runXXX_XXX.dat处理好的runXXX.XXX.json文件处理成只需要用于训练的数据
        """
        processed_dir = Path(input_dir)
        output_dir_ = Path(output_path)
        output_dir_.mkdir(parents=True, exist_ok=True)

        json_files = sorted(processed_dir.glob('*.json'))
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # 只保留CH0和CH1
            new_events = []
            for event in data['events']:
                new_events.append({
                    'event_id': event['event_id'],
                    'CH0': event['CH0'],
                    'CH1': event['CH1'],
                })

            #保存新JSON
            new_data = {
                'filename':data['filename'],
                'position_label':data['position_label'],
                'position_unit':data['position_unit'],
                'num_events':len(data['events']),
                'samples_per_channel':data['samples_per_channel'],
                'events': new_events,
            }
            output_file = output_dir_/json_file.name
            with open(output_file, 'w') as f:
                json.dump(new_data, f, indent=2)

            print(f"✅ {json_file.name} → 提取完成")

        print(f"\n✅ 全部完成！输出目录: {output_dir_}")



#============================ Test start ====================================================#

# if __name__ == "__main__":
#     """
#      测试单条数据集dat转json并保存
#     """
#     filepath = '../../dataset/raw/run7_65.dat'
#     output_json = '../../dataset/processed/run7_65.json'
#     reader = DatFileReader(filename=filepath)
#     reader.save_to_json(output_json)



if __name__ == '__main__':
    """
    批量处理dat数据集转json并保存
    """
    raw_dir = Path('../../dataset/raw')
    output_dir = Path('../../dataset/processed')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有.dat文件
    dat_files = sorted([f for f in raw_dir.glob('*.dat')])
    print(f"找到 {len(dat_files)} 个.dat文件\n")
    print("开始批量处理...\n")

    success_count = 0
    failed_files = []
    for i, dat_file in enumerate(dat_files, 1):
        try:
            print(f"[{i}/{len(dat_files)}] 处理: {dat_file.name}")
            # 创建reader
            reader = DatFileReader(filename=str(dat_file))
            # 输出Json路径
            output_json = output_dir / dat_file.name.replace('.dat', '.json')

            # 保存为Json
            reader.save_to_json(str(output_json))
            success_count += 1
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            failed_files.append(dat_file.name)

#============================ Test end ====================================================#



