import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

import Scintillator_Project
PROJECT_ROOT = Path(Scintillator_Project.__file__).parent

def load_split(split_dir: Path, split: str):
    """从 JSON加载波形和标签"""
    with open(split_dir / f"{split}.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    waveforms, labels = [], []
    for ev in data['events']:
        ch0 = np.array(ev['CH0']) # (1024,)
        ch1 = np.array(ev['CH1']) # (1024,)
        waveforms.append(np.stack([ch0, ch1])) # (2, 1024)
        labels.append(ev['position_label'])
    return np.array(waveforms), np.array(labels) # (N,2,1024), (N,)

def calc_charge(waveforms):
    """电荷量比法特征：Q_L/(Q_L + Q_R)"""
    # 波形是负脉冲，取绝对值再求和
    Q_L = np.abs(waveforms[:, 0, :]).sum(axis=1) # (N,) 所有事件的 CH0
    Q_R = np.abs(waveforms[:, 1, :]).sum(axis=1) # (N,) 所有事件的 CH1
    ratio = Q_L / (Q_L + Q_R + 1e-9) # 1e-9 防止分母为0
    return ratio.reshape(-1, 1) # (N,) -> (N, 1), sklearn 的模型要求输入是二维特征矩阵

def calc_timediff(waveforms):
    """时刻差法特征：argmin(CH0) - argmin(CH1)（负脉冲取谷值）"""
    t_L = np.argmin(waveforms[:, 0, :], axis=1) # (N,)
    t_R = np.argmin(waveforms[:, 1, :], axis=1) # (N,)
    delta_t = (t_L - t_R).reshape(-1, 1)
    return delta_t

def evaluate(name, X_train, y_train, X_test, y_test):
    """线性回归拟合 + RMSE/MAE 计算"""
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rmse = np.sqrt(((pred - y_test) ** 2).mean())
    mae = mean_absolute_error(y_test, pred)
    print(f"{name:12s} | RMSE: {rmse:.4f} cm | MAE: {mae:.4f} cm")
    return rmse


# ============================================================
# 修正版传统方法（对应大場卒论 PDF 实现） Start

# ============原始波形 (CH0 + CH1)
#         ↓
# ① 截取有效脉冲区间（不是整个1024点）
#         ↓
# ② 电荷量比法（修正版）
#         ↓
# ③ CFD时间差法（修正版）
#         ↓
# ④ 加权融合推定
#         ↓
# ⑤ 与 CNN RMSE 比较
# ================================================
def find_pulse_roi(waveform, window=256):
    """以脉冲谷底为中心，截取有效区间
    自动找到脉冲真正出现的位置

    只截取有效区域，而不是整个1024点
    因为1024点波形里，真正有用的脉冲，可能只有中间200~300点，其余部分，全是噪声，全是基线。
    如果把1024点全部积分，会导致电荷量严重不准，所以必须只对脉冲区域积分
    """
    peak_idx = np.argmin(waveform) # 波形是负脉冲
    start = max(0, peak_idx - window // 2) # window=256，截取长度固定为256个采样点
    end = start + window
    if end > len(waveform):
        end = len(waveform)
        start = end - window
    return start, end


def calc_charge_ratio_v2(waveforms):
    """
    修正版电荷比法:  R₀ = ln(Q_R / Q_L)
    - 只对脉冲区间积分 （不用全段1024点）
    - 对应大場卒论公式 R₀ = ln(Q_R/Q_L)
    :param waveforms:
    :return:
    """
    results = []
    for i in range(len(waveforms)):
        ch0 = waveforms[i, 0, :]
        ch1 = waveforms[i, 1, :]
        s0, e0 = find_pulse_roi(ch0)
        s1, e1 = find_pulse_roi(ch1)
        Q_L = np.abs(ch0[s0:e0]).sum()
        Q_R = np.abs(ch1[s1:e1]).sum()
        R0 = np.log((Q_R + 1e-9) / (Q_L + 1e-9))
        results.append(R0)
    return np.array(results).reshape(-1, 1)


def calc_cfd_timediff(waveforms, fraction=0.1):
    """
    CFD法时间差：Δt = t_R(CFD) - t_L(CFD)
    - 对负脉冲：找信号首次低于 fraction × V_min 的时刻
    - fraction=0.3 表示在30%最小值处触发
    :param waveforms:
    :param fraction:
    :return:
    """
    delta_list = []
    for i in range(len(waveforms)):
        ch0 = waveforms[i, 0, :]
        ch1 = waveforms[i, 1, :]
        thresh_L = np.min(ch0) * fraction   # 负脉冲：阈值为负值
        thresh_R = np.min(ch1) * fraction

        # 找首次低于阈值的索引（上升沿检测，信号从基线下降到阈值）
        below_L = np.where(ch0 < thresh_L)[0]
        below_R = np.where(ch1 < thresh_R)[0]

        t_L = below_L[0] if len(below_L) > 0 else np.argmin(ch0)
        t_R = below_R[0] if len(below_R) > 0 else np.argmin(ch1)

        delta_list.append(t_R - t_L)
    return np.array(delta_list).reshape(-1, 1)


def evaluate_fusion(X_train_q, X_train_t, y_train, X_test_q, X_test_t, y_test):
    """
    融合推定：对应大場卒论的加权融合
    x̂_p(ω) = ω × x̂_q + (1-ω) × x̂_t
    最优 ω 使融合残差的方差最小
    :param X_train_q:
    :param X_train_t:
    :param y_train:
    :param X_test_q:
    :param X_test_t:
    :param y_test:
    :return:
    """
    lr_q = LinearRegression().fit(X_train_q, y_train)
    lr_t = LinearRegression().fit(X_train_t, y_train)

    pred_q_train = lr_q.predict(X_train_q)
    pred_t_train = lr_t.predict(X_train_t)

    resid_q = pred_q_train - y_train
    resid_t = pred_t_train - y_train

    # 最优权重解析解
    sigma_q2 = np.var(resid_q)
    sigma_t2 = np.var(resid_t)
    sigma_qt = np.cov(resid_q, resid_t)[0, 1]
    denom = sigma_q2 + sigma_t2 - 2 * sigma_qt
    omega = (sigma_t2 - sigma_qt) / (denom + 1e-9)
    omega = float(np.clip(omega, 0, 1))

    pred_q_test = lr_q.predict(X_test_q)
    pred_t_test = lr_t.predict(X_test_t)
    pred_fusion = omega * pred_q_test + (1 - omega) * pred_t_test

    rmse = np.sqrt(((pred_fusion - y_test) ** 2).mean())
    mae = mean_absolute_error(y_test, pred_fusion)
    print(f"CFD融合法(v2) | RMSE: {rmse:.4f} cm | MAE: {mae:.4f} cm | ω={omega:.3f}")
    return rmse

def main_v2():
    split_dir = PROJECT_ROOT / 'dataset' / 'split'
    X_train_raw, y_train = load_split(split_dir, "train")
    X_test_raw, y_test = load_split(split_dir, "test")

    print('\n===== 修正版传统方法 vs CNN =====\n')

    # 修正版电荷比法
    X_train_q2 = calc_charge_ratio_v2(X_train_raw)
    X_test_q2 = calc_charge_ratio_v2(X_test_raw)
    rmse_q2 = evaluate("电荷比法(v2)", X_train_q2, y_train, X_test_q2, y_test)

    # CFD时刻差法
    X_train_t2 = calc_cfd_timediff(X_train_raw, fraction=0.3)
    X_test_t2 = calc_cfd_timediff(X_test_raw, fraction=0.3)
    rmse_t2 = evaluate("CFD法(v2)  ", X_train_t2, y_train, X_test_t2, y_test)

    # 融合推定
    rmse_f2 = evaluate_fusion(
        X_train_q2, X_train_t2, y_train,
        X_test_q2, X_test_t2, y_test
    )

    cnn_rmse = 5.5244
    best_trad = min(rmse_q2, rmse_t2, rmse_f2)
    diff = cnn_rmse - best_trad
    print(f'\nCNN(本研究)   | RMSE: {cnn_rmse:.4f} cm')
    print(f'修正版最优传统 | RMSE: {best_trad:.4f} cm')
    if diff > 0:
        print(f'CNN 差于传统方法 {diff:.2f} cm')
    else:
        print(f'CNN 优于传统方法 {-diff:.2f} cm')



# ============================================================
# 修正版传统方法（对应大場卒论 PDF 实现） End
# ============================================================


def main():
    split_dir = PROJECT_ROOT / 'dataset' / 'split'

    X_train_raw, y_train = load_split(split_dir, 'train')
    X_test_raw, y_test = load_split(split_dir, 'test')

    print('===== 传统方法 vs CNN =====\n')

    # 电荷量比法
    X_train_q = calc_charge(X_train_raw)
    X_test_q = calc_charge(X_test_raw)
    rmse_q = evaluate("电荷量比法 ", X_train_q, y_train, X_test_q, y_test)

    # 时刻差法
    X_train_t = calc_timediff(X_train_raw)
    X_test_t = calc_timediff(X_test_raw)
    rmse_t = evaluate("时刻差法 ", X_train_t, y_train, X_test_t, y_test)

    # 组合（两个特征一起）
    X_train_c = np.hstack([X_train_q, X_train_t])
    X_test_c = np.hstack([X_test_q, X_test_t])
    rmse_c = evaluate("组合法 ", X_train_c, y_train, X_test_c, y_test)

    # 对比 CNN
    cnn_rmse = 5.5244
    best_trad = min(rmse_q, rmse_t, rmse_c)
    improvement = (best_trad - cnn_rmse) / best_trad * 100
    print(f'\nCNN(本研究) | RMSE: {cnn_rmse:.4f} cm')
    print(f'最优传统方法 | RMSE: {best_trad:.4f} cm')
    print(f'提升幅度    ｜ {improvement:.1f}%')


if __name__ == '__main__':

    main()      # 原版（保留）
    main_v2()   # 修正版
    # import json
    # d = json.load(open('../../dataset/split/test.json'))
    # ev = d['events'][0]
    # print(list(ev.keys()))


"""

===== 传统方法 vs CNN =====

电荷量比法        | RMSE: 10.8716 cm | MAE: 8.2850 cm
时刻差法         | RMSE: 12.7713 cm | MAE: 9.3547 cm
组合法          | RMSE: 9.1726 cm | MAE: 6.7263 cm

CNN(本研究) | RMSE: 5.5244 cm
最优传统方法 | RMSE: 9.1726 cm
提升幅度    ｜ 39.8%

===== 修正版传统方法 vs CNN =====

电荷比法(v2)     | RMSE: 12.0887 cm | MAE: 9.4092 cm
CFD法(v2)     | RMSE: 6.8391 cm | MAE: 4.8911 cm
CFD融合法(v2) | RMSE: 6.7492 cm | MAE: 4.8316 cm | ω=0.292

CNN(本研究)   | RMSE: 5.5244 cm
修正版最优传统 | RMSE: 6.7492 cm
CNN 优于传统方法 1.22 cm

"""