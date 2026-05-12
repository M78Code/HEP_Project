# Step 3.4: Focal Loss（自定义损失函数）和nn.CrossEntropyLoss()一样，只是自己实现

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
解决类不平衡问题。

  训练时真实数据里反质子和反重氘核的数量差距极大（论文提到约
  10⁴:1），如果用普通的交叉熵损失（CrossEntropyLoss），模型会直接学会"全部预测为多数类"来最小化loss，完全不学少数类的特征。

  ---
  普通 CrossEntropyLoss 的问题

  假设 10000 个反质子，1 个反重氘核：
  - 模型全预测反质子 → 准确率 99.99%，但反重氘核完全识别不出来
  - Loss 很小，梯度很小，模型不更新

  ---
  Focal Loss 的解决方式

  在 CrossEntropyLoss 上乘一个权重因子 (1-p)^γ：

  Focal Loss = -(1-p_t)^γ · log(p_t)

  - 模型已经分对的样本（p 接近1）→ (1-p)^γ 接近0 → loss 权重小，贡献少
  - 模型分错的困难样本 → (1-p)^γ 接近1 → loss 权重大，强迫模型重点学这些

  γ=1.5 时（IceCube 论文验证的最优值），困难样本的 loss 被放大，模型被迫去学少数类。

  ---
  对我们项目的意义

  ┌────────────────────┬──────────────────────┬────────────┐
  │                    │   CrossEntropyLoss   │ FocalLoss  │
  ├────────────────────┼──────────────────────┼────────────┤
  │ 多数类（反质子）   │ 主导训练             │ 贡献被压制 │
  ├────────────────────┼──────────────────────┼────────────┤
  │ 少数类（反重氘核） │ 被忽略               │ 被重点关注 │
  ├────────────────────┼──────────────────────┼────────────┤
  │ 最终效果           │ Rejection Power ~10³ │ 目标 ~10⁵  │
  └────────────────────┴──────────────────────┴────────────┘

  现在数据只有反重氘核（57个event），等反质子数据到了类不平衡会很严重，到时候 Focal Loss 是必须的。要继续输出 losses.py 代码吗？

❯ 你是在哪里学到这个的，从docs/papers里面的论文中吗

⏺ 两个来源都有：

  论文中学到的（直接依据）

  - 论文05（IceCube 2025）：明确测试了 γ=0, 1, 1.5, 2, 2.5，结论是 γ=1.5 最优，并写明使用 Focal Loss 应对类不平衡
  - 论文08（GNN综述）：Section 2.1 明确写道"Focal Loss 是应对 10⁴:1 级不平衡的有效方法"，并确认这是 HEP 分类任务的标准做法

  我自己的深度学习背景知识

  Focal Loss 本身是 2017 年 Facebook 的论文《Focal Loss for Dense Object Detection》提出的（论文05的 Reference [9]
  也引用了它），是目标检测领域的经典方法，不是 HEP 专有的。类不平衡问题和 Focal Loss 的原理是通用的深度学习知识。

  ---
  所以结论是：Focal Loss 要用这个判断来自论文，γ=1.5 作为初始值这个具体参数来自论文05的实验结果。
"""
class FocalLoss(nn.Module):
    """
    Focal Loss, 用于应对类不平衡问题（反质子 vs 反重氘核）
    来源：Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    参数参考：IceCube 2025论文（γ=1.5 为最优）
    公式：FL(p_t) = -(1 - p_t)^γ · log(p_t)

    Args:
        gamma   : 聚集参数，γ=0 退化为普通CrossEntropyLoss，γ越大越关注困难样本
        alpha   : 类别权重（list或None），None表示不加权
        reduction : 'mean' | 'sum' | 'none'
    """
    def __init__(self, gamma: float = 1.5, alpha=None, reduction: str='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : 模型输出 [batch_size, num_classes]（未经softmax）
            targets : 真实标签 [batch_size] (long型)
        Returns:
            loss    : scalar
        """

        """
        标准交叉熵（逐样本）
        不是直接平均，而是每个样本单独计算，
            例如：[0.1, 2.3, 0.5, 1.8]
            因为FocalLoss需要对每个样本分别加权，不能直接mean
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none') # [batch_size]

        # p_t: 模型对正确类别的预测概率
        p_t = torch.exp(-ce_loss)

        # Focal 权重
        focal_weight = (1 - p_t) ** self.gamma

        # 类别权重 alpha
        if self.alpha is not None:
            self.alpha = self.alpha.to(logits.device)
            alpha_t = self.alpha[targets]
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# ── 快速测试 ────────────────────────────────────────────
if __name__ == '__main__':
    # 模拟 batch_size=8, 2分类
    logits = torch.randn(8, 2)
    targets = torch.tensor([1, 1, 1, 1, 1, 1, 1, 0]) # 7个反重氘核，1个反质子

    # γ=1.5，不加权
    criterion = FocalLoss(gamma=1.5)
    loss = criterion(logits, targets)
    print(f'FocalLoss (γ=1.5):          {loss.item():.4f}')

    # 对比普通 CrossEntropyLoss（γ=0 等价）
    criterion_ce = FocalLoss(gamma=0)
    loss_ce = criterion_ce(logits, targets)
    print(f'CrossEntropyLoss (γ=0):          {loss_ce.item():.4f}')

    # 加 alpha 权重（反质子少 → 权重高）
    criterion_alpha = FocalLoss(gamma=1.5, alpha=[2.0, 1.0])
    loss_alpha = criterion_alpha(logits, targets)
    print(f"FocalLoss (γ=1.5, alpha):   {loss_alpha.item():.4f}")


"""
FocalLoss (γ=1.5):          0.5112
CrossEntropyLoss (γ=0):          0.9048
FocalLoss (γ=1.5, alpha):   0.6598
"""