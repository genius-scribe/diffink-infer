# DiffInk 训练损失结构：为什么 Writer Classifier 不是"奖励模型"

本文档解释 DiffInk 里 `WriterStyleClassifier` 和 `ChineseHandwritingOCR`
这两个辅助头的角色——它们和 VAE 是**联合训练（joint / multi-task）** 的，
不是像 RLHF 里的奖励模型那样**独立训练好后单独用**。

> 注：本仓库 `diffink-infer` 是**推理仓**，没有训练脚本。
> 下面"训哪些层 / loss 怎么算"的细节基于 `diffink/model/` 下的模型结构 + multi-task 描述合理推断，
> 与官方训练代码可能在常数（loss 权重、是否 detach 等）上有出入，但层归属与计算流程是确定的。

---

## 总览：DiffInk 全模型流程图（阶段 × 模块 × loss）

**图例（emoji 含义）**：

| Emoji | 含义 |
|---|---|
| 🔥 | **训练**：参数被这阶段的 loss 反向更新 |
| ❄️ | **冻结**：参数不更新，但仍 forward 提供输出（`requires_grad=False`） |
| ⛔ | **不调用**：连 forward 都不执行（既不在计算图里，也不出现在数据流上） |
| ➖ | **不存在**：这阶段还没创建 / 没加载这个模块 |

---

### 阶段 1 🔥 训练 VAE

> 产出 `vae_epoch_100.pt`。4 个 loss 联合训练；DiT 还不存在。

```
                          x   [B, 5, T]   输入轨迹
                          │
                          ▼
                   ┌────────────────┐
                   │   encoder  🔥  │   ← 被 ① ② ③ ④ 一起更新
                   └────────┬───────┘
                            │ features
                            ▼
                   ┌────────────────┐
                   │ conv_mu/logv 🔥│   ← 被 ① ② ③ ④ 一起更新
                   └────────┬───────┘
                            │
                  z = reparameterize(mu, logvar)
                            │  (同一个 z 共享给 3 个头)
        ┌───────────────────┼───────────────────┐         (mu, logvar 直连 KL)
        │                   │                   │                       │
        ▼                   ▼                   ▼                       ▼
┌────────────────┐  ┌────────────────┐  ┌────────────────┐    ┌────────────────┐
│ decoder + TD   │  │   OCR head     │  │  Writer Cls    │    │   KL  正则     │
│      🔥        │  │      🔥        │  │      🔥        │    │  (无独立头)    │
│   只被 ① 更新  │  │   只被 ② 更新  │  │   只被 ③ 更新  │    │   只更新 enc   │
└────────┬───────┘  └────────┬───────┘  └────────┬───────┘    └────────┬───────┘
         │                   │                   │                     │
         ▼                   ▼                   ▼                     ▼
     GMM 参数            字符 logits         writer logits          KL 标量
     (123 维)        (T_lat, B, vocab)        (B, 90)
         │                   │                   │                     │
         ▼                   ▼                   ▼                     ▼
    ① GMM_NLL             ② CTC              ③ CE                  ④ KL
         └───────────────────┴─────────┬─────────┴─────────────────────┘
                                       ▼
                       total = λ₁·① + λ₂·② + λ₃·③ + λ₄·④
                                       │
                                       ▼
                     反向 backward → 更新所有 🔥 标记的模块
```

---

### 阶段 2 🔥 训练 DiT （VAE ❄️ 全部冻结）

> 产出 `dit_epoch_1.pt`。VAE 整体 ❄️ 冻结；OCR / Writer / decoder 在这阶段甚至 ⛔ 不调用。

```
   x  [B, 5, T]                                    text  字符序列
        │                                                │
        ▼                                                ▼
  ┌────────────────┐                            ┌────────────────┐
  │   encoder  ❄️   │                            │  TextEmbed  🔥 │
  └────────┬───────┘                            └────────┬───────┘
           ▼                                             │
  ┌────────────────┐                                     │
  │ conv_mu/logv ❄️│                                     │
  └────────┬───────┘                                     │
           │                                             │
       z_clean = mu  (推理式编码,无 reparam)              │
           │                                             │
           │ + 噪声 ε(t),时间步 t                         │
           ▼                                             │
  ┌──────────────────────────────────────────────┐      │
  │  z_noisy   ← TimestepEmbed 🔥                │      │
  │            ← ConvPosEmbed  🔥                │ ◄────┘ (text_cond)
  └──────────────────────┬───────────────────────┘
                         ▼
                 ┌────────────────┐
                 │  DiTBlock × N  │ 🔥  ← 被 ⑤ 更新
                 │ + AdaLN_Final  │ 🔥
                 └────────┬───────┘
                          ▼
                       v_pred
                          │
                          ▼
                  ⑤ MSE(v_pred, target)
                          │
                          ▼
              反向 backward → 只更新 DiT 内的 🔥
              VAE 的 ❄️ 完全不动

   ⛔ vae.decoder + transformer_decoder        (此阶段不调用)
   ⛔ vae.ocr_model                             (此阶段不调用)
   ⛔ vae.style_classifier                      (此阶段不调用)
```

---

### 推理 ❄️ inference （`no_grad`，全冻结）

> 加载 `vae_epoch_100.pt + dit_epoch_1.pt`，从 prefix 真迹 + text 字符生成新轨迹。

```
   x_prefix (前 N 字真迹)                          text  字符序列
         │                                              │
         ▼                                              ▼
   ┌────────────────┐                            ┌────────────────┐
   │   encoder  ❄️   │                            │  TextEmbed  ❄️ │
   └────────┬───────┘                            └────────┬───────┘
            ▼                                             │
   ┌────────────────┐                                     │
   │ conv_mu/logv ❄️│                                     │
   └────────┬───────┘                                     │
            │                                             │
        z_prefix                                          │
            │                                             │
            └────────────────────┬────────────────────────┘
                                 │ (z_prefix 作为风格条件)
                                 ▼
       纯噪声 z_T  ──►  ┌──────────────────────┐
                        │   DiT 多步去噪采样   │ ❄️
                        │ (Time + ConvPos +    │
                        │  Block×N + AdaLN)    │
                        └──────────┬───────────┘
                                   ▼
                                 z_gen
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │   decoder + TD   ❄️  │
                        └──────────┬───────────┘
                                   ▼
                              GMM 参数
                                   │
                                   ▼
                  从 GMM 采样得到 x_gen (生成的笔迹轨迹)

   ⛔ vae.ocr_model              (推理时不调用)
   ⛔ vae.style_classifier       (推理时不调用)
```

---

### ① ~ ⑤ 五个 loss 各自更新哪些模块

| Loss | 含义 | 反向梯度路径上的参数（被这路 loss 更新的模块） |
|---|---|---|
| ① **GMM_NLL** | 重建 | encoder + conv_mu/logv + **decoder + transformer_decoder** |
| ② **CTC** | 内容 | encoder + conv_mu/logv + **ocr_model** ★ 不经过 decoder |
| ③ **CE** | 风格 | encoder + conv_mu/logv + **style_classifier** ★ 不经过 decoder |
| ④ **KL** | 正则 | encoder + conv_mu/logv ★ 不经过 decoder/z |
| ⑤ **MSE** | 扩散 | **DiT 全部子模块**（TextEmbed + TimestepEmbed + ConvPosEmbed + DiTBlock×N + AdaLN_Final） |

---

### 每个模块 × 每个阶段的状态

| 模块 | 阶段 1 训 VAE | 阶段 2 训 DiT | 推理 |
|---|---|---|---|
| `encoder` | 🔥 被 ①②③④ 更新 | ❄️ forward 给 z | ❄️ encode 前缀 |
| `conv_mu` / `conv_logvar` | 🔥 被 ①②③④ 更新 | ❄️ forward 给 z | ❄️ encode 前缀 |
| `decoder` + `transformer_decoder` | 🔥 只被 ① 更新 | ⛔ 不调用 | ❄️ decode 出轨迹 |
| `ocr_model` | 🔥 只被 ② 更新 | ⛔ 不调用 | ⛔ 不调用 |
| `style_classifier` | 🔥 只被 ③ 更新 | ⛔ 不调用 | ⛔ 不调用 |
| **── DiT 内部 ──** | | | |
| `TextEmbedding` | ➖ 不存在 | 🔥 被 ⑤ 更新 | ❄️ text 条件 |
| `TimestepEmbedding` | ➖ 不存在 | 🔥 被 ⑤ 更新 | ❄️ 步数 t 条件 |
| `ConvPositionEmbedding` | ➖ 不存在 | 🔥 被 ⑤ 更新 | ❄️ |
| `DiTBlock × N` | ➖ 不存在 | 🔥 被 ⑤ 更新 | ❄️ 多步去噪 |
| `AdaLayerNormZero_Final` | ➖ 不存在 | 🔥 被 ⑤ 更新 | ❄️ |

---

**怎么读这套图**：

- **看流程图**：每个 emoji 直接告诉你"这个模块这阶段是要 🔥 训 / 要 ❄️ forward / 还是 ⛔ 干脆不调用"
- **看 loss 表**：每路 loss 的反向梯度只走它前向经过的路径——`decoder` 仅被 ① 触及；OCR / Writer 头不更新 decoder；DiT 阶段的 ⑤ 完全不影响 VAE
- **看模块状态表**：想知道"阶段 X 的模块 Y 状态如何"，一行一查即可

下面 §1 ~ §7 是对这套总览的详细展开。

---

## 1. 你可能以为的架构（奖励模型思路）

这是一种常见误解，源自强化学习 / RLHF 的思路：

```
          ┌──────────────────────┐
          │  第 1 步：单独训练   │
          │  reward_model(x)→ r  │   ← 单独训练，冻结
          └──────────────────────┘
                      │
                      ▼  用它的分数当监督信号
          ┌──────────────────────┐
          │  第 2 步：训主模型   │
          │   主模型生成 → r 打分 → 反向更新主模型
          └──────────────────────┘
```

在这种设定里，`reward_model` 是一个**独立的评估器**，它自己有训练流程
和自己的数据标签，训好之后**不会再被更新**，只用来给主模型打分 /
提供梯度信号。

**DiffInk 不是这样的。**

---

## 2. DiffInk 实际的架构（multi-task 联合训练）

`VAE` 是一个大的 `nn.Module`，它内部同时包含多个头。下面分**前向**和**反向**两段看：

```
─── 前向 forward ──────────────────────────────────────────────────────────
                          ┌─────────────┐
       x ────────────────►│   Encoder   │
     [B, 5, T]            └──────┬──────┘
                                 │  features
                                 ▼
                          ┌─────────────┐    mu, logvar
                          │conv_mu/logv │──────────────────────────────┐
                          └──────┬──────┘                              │
                                 │                                     │
                          z = reparam(mu, logvar)                      │
                                 │  (同一个 z 共享给 3 个头)            │
            ┌────────────────────┼────────────────────┐                │
            │                    │                    │                │
            ▼                    ▼                    ▼                ▼
     ┌──────────┐          ┌──────────┐      ┌──────────────────┐  ┌────────┐
     │ Decoder  │          │ OCR head │      │  Writer Style    │  │   KL   │
     │ +TransDec│          │  (CTC)   │      │   Classifier     │  │  正则  │
     └─────┬────┘          └─────┬────┘      └────────┬─────────┘  └────┬───┘
           │                     │                    │                 │
           ▼                     ▼                    ▼                 ▼
       GMM 参数            字符序列 logits      90-way writer logits  KL 标量
       (123 维)         (T_lat, B, num_classes)    (B, 90)
           │                     │                    │                 │
           ▼                     ▼                    ▼                 ▼
       ① GMM_NLL             ② CTC loss           ③ CE loss         ④ KL loss
           └─────────────┬───────────────────┬──────────────────────────┘
                         │  total = λ₁·① + λ₂·② + λ₃·③ + λ₄·④
                         ▼
                    一次反向 backward


─── 反向 backward (每路梯度只沿自己的前向路径回流) ──────────────────────────
   ① GMM_NLL  ──► Decoder+TransDec ──► conv_mu/logv ──► Encoder
   ② CTC      ──► OCR head         ──► conv_mu/logv ──► Encoder    ← 不经过 Decoder
   ③ CE       ──► Writer Cls       ──► conv_mu/logv ──► Encoder    ← 不经过 Decoder
   ④ KL       ──►       (mu,logvar 直连) ──► conv_mu/logv ──► Encoder    ← 不经过 Decoder/z 采样


─── 每个模块被"几路"梯度更新 ─────────────────────────────────────────────
   ★ Encoder              ← ① ② ③ ④   (4 路同时拉扯 → latent 同时编码"形状+内容+风格")
   ★ conv_mu / conv_logvar ← ① ② ③ ④   (4 路全部)
     Decoder + TransDec   ← ①          (★只被重建 loss 更新)
     OCR head             ← ②          (只被 CTC 更新)
     Writer Classifier    ← ③          (只被 CE 更新)
```

**关键点**：
- 所有头（decoder、ocr_model、style_classifier）都是 `VAE` 的子模块
- 训练时一次 forward 同时算四个 loss，加权求和后一起反向传播
- Encoder 的参数被 4 路梯度一起推着走
- 训完之后，**所有权重都保存在同一个文件** `vae_epoch_100.pt`

### 2.1 阶段 1 训 VAE 时具体更新哪些层

`VAE.parameters()` 一次性返回下表所有子模块的参数，**全部参与反向传播**：

| 子模块（`self.xxx`） | 实现位置 | 内部组成 | 角色 |
|---|---|---|---|
| `encoder` | [blocks.py:35-54](diffink/model/blocks.py#L35-L54) | 3 × (stride-2 Conv1d + ResidualStack(4 层)) | 把轨迹 `[B,5,T]` 下采样 8 倍到特征 `[B, hidden_dims[-1], T/8]` |
| `conv_mu` / `conv_logvar` | [vae.py:16-17](diffink/model/vae.py#L16-L17) | 两个 1×1 Conv1d | 把 encoder 输出投到 `latent_dim`，给出高斯参数 |
| `decoder` | [blocks.py:57-76](diffink/model/blocks.py#L57-L76) | 3 × (ResidualStack + ConvTranspose1d) | 把 latent z 上采样回原 T |
| `transformer_decoder` | [blocks.py:79-97](diffink/model/blocks.py#L79-L97) | Linear + N 层 TransformerEncoderLayer + Linear | 输出 GMM 参数（123 维 = 20 高斯 × 6 + 笔状态 3） |
| `ocr_model` | [ocr.py:22-51](diffink/model/ocr.py#L22-L51) | Linear + 正弦 PE + N 层 TransformerEncoder + Linear | 在 latent 上识别字符序列，提供 CTC 监督 |
| `style_classifier` | [writer.py:5-45](diffink/model/writer.py#L5-L45) | LayerNorm + 3 路 masked 池化拼接 + 3 层 MLP | 在 latent 上识别 writer，提供 CE 监督 |

**总损失**：
```
L_vae = λ_recon · GMM_NLL + λ_kl · KL + λ_ctc · CTC + λ_ce · CE
```
- `GMM_NLL`、`KL` 来自主路（重建轨迹 + 正则 latent 分布）
- `CTC`、`CE` 来自两个辅助头（详见 §2.2 / §2.3）

**重要：每个 loss 只更新它前向计算图上经过的参数**

PyTorch autograd 的硬规则——梯度只沿前向计算图反向流动。所以下面这张表说明哪些层被哪些 loss 更新：

| Loss | 前向计算路径 | 反向梯度真正更新到的参数 |
|---|---|---|
| `GMM_NLL` | x → **encoder → conv_mu/logvar → z → decoder → transformer_decoder** → GMM 参数 | encoder + conv_mu/logvar + **decoder + transformer_decoder** |
| `KL` | encoder → **conv_mu/logvar 出 mu, logvar** | encoder + conv_mu/logvar |
| `CTC` | x → encoder → conv_mu/logvar → z → **ocr_model** → 字符 logits | encoder + conv_mu/logvar + **ocr_model**（**不经过 decoder**） |
| `CE` | x → encoder → conv_mu/logvar → z → **style_classifier** → writer logits | encoder + conv_mu/logvar + **style_classifier**（**不经过 decoder**） |

**关键观察**（回答"OCR/Writer 头是不是只训 encoder、不训 decoder？"）：
- **encoder + conv_mu/logvar 被 4 路梯度共享拉扯**——这是辅助任务能"塑造编码空间"的根本原因
- **decoder + transformer_decoder 只被 GMM_NLL 一路梯度更新**——CTC 和 CE 完全不影响 decoder（前向就没经过它们，反向自然也回不去）
- `ocr_model` 自身只被 CTC 更新
- `style_classifier` 自身只被 CE 更新

所以更准确的说法是：**OCR 头和 Writer 头训练时同时更新它们自己 + encoder（含 conv_mu/logvar），但不会更新 decoder。** Decoder 只靠重建 loss 学。

最终 latent 同时携带"轨迹形状 + 内容字符 + writer 风格"信息——但"轨迹形状"是重建头逼出来的，"内容字符"是 OCR 头逼出来的，"writer 风格"是 Writer 头逼出来的，三种信息分别由不同 loss 在 encoder 上施加压力。

### 2.2 CTC loss 端到端怎么算

**注册**（见 [vae.py:34](diffink/model/vae.py#L34)）：
```python
self.ctc = nn.CTCLoss(blank=0, zero_infinity=True)
```

**forward 流程**（结合 [ocr.py:44-51](diffink/model/ocr.py#L44-L51)）：

```python
# 1. 拿 latent
z, mu, logvar = vae.encode(x)            # z: [B, latent_dim, T_lat]   T_lat = T/8

# 2. 过 OCR 头
logits_btc = vae.ocr_model(z)            # 内部:
                                         #   z.permute(0,2,1)         → [B, T_lat, latent_dim]
                                         #   input_proj + PE          → [B, T_lat, hidden]
                                         #   TransformerEncoder       → [B, T_lat, hidden]
                                         #   output_fc                → [B, T_lat, num_classes]
                                         #   permute(1,0,2)           → [T_lat, B, num_classes]   ← CTC 要求时间维在前

# 3. 转 log 概率
log_probs = F.log_softmax(logits_btc, dim=-1)    # [T_lat, B, num_classes]

# 4. 算 loss
ctc_loss = vae.ctc(
    log_probs,
    targets         = char_ids_concat,           # [sum(target_lengths)],所有样本的字符 id 拼接
    input_lengths   = torch.full((B,), T_lat),   # 每个样本的 latent 帧数(无 padding 时全相等)
    target_lengths  = char_lens,                 # [B],每个样本真实字符数
)
```

**4 个参数到底是什么——用一个具体例子说清楚**

假设 batch=2，两个样本的标签字符串分别是 "你好" 和 "再见"，字符表中：
```
0:  <blank>     (CTC 专用空白符,固定占 id=0)
5:  你
8:  再
12: 好
21: 见
```
那么传给 `vae.ctc(...)` 的 4 个参数实际长这样：

| 参数 | 形状 | 例子里的实际值 | 含义 |
|---|---|---|---|
| `log_probs` | `[T_lat, B, C]` | `[T_lat, 2, vocab_size]` 浮点张量 | 每一帧对所有字符（含 blank）的对数概率。**注意时间维 T 在前**，这是 PyTorch CTCLoss 的硬性要求 |
| `targets` | `[sum(target_lengths)]` | `[5, 12, 8, 21]`，shape `[4]` | **把 batch 所有样本的字符 id 首尾拼接成一条 1D 向量**。前 2 个 `[5,12]` = 你好，后 2 个 `[8,21]` = 再见 |
| `input_lengths` | `[B]` | `[T_lat, T_lat]` 或 `[120, 95]`（若样本 2 短一些） | 告诉 CTC 每个样本在时间维上"有效"几帧。超出部分（padding 帧）被忽略 |
| `target_lengths` | `[B]` | `[2, 2]` | 告诉 CTC 在 `targets` 里每个样本各占几个字符。CTC 用 cumsum 切片：样本 1 = `targets[0:2]`、样本 2 = `targets[2:4]` |

> 为什么 `targets` 用 1D 拼接而不是 2D padded？
> 因为不同样本字符数不一样，2D 要 padding（用 0 填）；但 0 就是 blank id，会和真实 blank 混淆。
> 1D 拼接 + `target_lengths` 切片更干净，也更省内存。
> （PyTorch 也允许传 2D `[B, max_len]`，但要保证 padding 值不和真实字符 id 冲突。）

**CTC 拿到这四样东西后做的事**

对每个样本 `b`：
1. 从 `log_probs[:, b, :]` 取出前 `input_lengths[b]` 帧 → 形状 `[L_b, C]`，记 `L_b = input_lengths[b]`
2. 从 `targets` 切出对应的 `target_lengths[b]` 个字符 id → 记为 `y_b`，长度 `S_b = target_lengths[b]`
3. 枚举**所有合法的"长度 L_b 的帧序列 → 长度 S_b 的字符序列"对齐路径**
   - 每帧可以选 blank（id=0）也可以选真实字符
   - 合并规则：连续相同字符合并成一个；blank 直接删掉
   - 例如 `L_b=4, S_b=2`，"你好" 的合法路径有 `你你好好`、`你-好-`、`-你好-`、`你好--` 等
4. 每条路径的概率 = 各帧对应字符的概率连乘
5. 所有路径概率**求和**（forward 算法用动态规划，不用真的枚举）→ `P(y_b | x_b)`
6. 该样本 loss = `-log P(y_b | x_b)`
7. 整个 batch 取均值（或求和，由 `reduction` 参数决定，默认 `mean`）

**`zero_infinity=True` 的兜底**：如果某个样本 `S_b > L_b`（字符比帧还多，根本没有合法对齐），`P(y_b | x_b) = 0` → `-log 0 = +inf` → 反传 NaN 训崩。`zero_infinity=True` 会把这种样本的 loss 直接设成 0、跳过它，避免崩溃。

**CTCLoss 内部在做什么**（一句话版）：
对每个样本，枚举所有合法的"帧→字符"对齐路径（允许在帧间插入 `blank=0`、允许同字符连续重复后合并），
对每条路径的对数概率求和（forward 算法），取负号当 loss。
`blank=0` 表示"空白符"对应的类别 id 是 0；`zero_infinity=True` 在 target 太长（无可行对齐）时返回 0 而非 `+inf`，防止训练崩溃。

**为什么 [ocr.py:39-42](diffink/model/ocr.py#L39-L42) 把 blank 的 bias 初始化成 -5**：
```python
self.output_fc.bias.data.zero_()
self.output_fc.bias[0].copy_(torch.tensor(-5.0))   # blank 的初始 logit 拉低
```
CTC 训练初期模型最容易塌缩成"全部预测 blank"（这种解 loss 也不算太大）。
把 blank 的初始概率压低，逼模型早期就尝试输出真实字符，避免陷在塌缩解里。

**这个 loss 给 encoder 什么信号**：
梯度从 `ctc_loss` → `output_fc` → `Transformer` → `input_proj` → **回到 latent z** → 回到 `encoder`。
效果：encoder 必须把"哪段时间是哪个字"的信息**沿时间维**保留在 latent 里，否则 OCR 头解不出来。

### 2.3 CE loss 端到端怎么算

**forward 流程**（见 [writer.py:22-45](diffink/model/writer.py#L22-L45)）：

```python
# 1. 拿一份特征(按 input_dim=config.style_classifier_dim 推断,通常就是 latent z 或 encoder 最后一层)
feat = z                                  # [B, C, T]
mask = valid_mask                         # [B, T],1=有效,0=padding;可为 None

# 2. 三路 masked 池化(见 writer.py:25-33)
avg_pool = masked_mean(feat, mask)        # [B, C]
std_pool = masked_std(feat, mask)         # [B, C]
max_pool = masked_max(feat, mask)         # [B, C]   padding 位置先填 -inf

# 3. 各自 LayerNorm,再拼接
pooled = cat([LN(avg), LN(max), LN(std)], dim=-1)   # [B, 3C]
pooled = Dropout(pooled)

# 4. 3 层 MLP 出 logits
logits = MLP(pooled)                      # [B, num_writers]   num_writers = 90

# 5. 标准交叉熵
ce_loss = F.cross_entropy(logits, writer_ids)       # writer_ids: [B] long
```

**CrossEntropy 内部在做什么**，等价于：
```
ce_loss = mean_over_batch( -log( softmax(logits)[writer_id] ) )
```
即把 logits 过 softmax 得到 90 类概率，挑出真实 writer 那一类的概率，取 -log，再对 batch 求均值。

**为什么用"三路池化"而不是简单 mean**：
- `mean`：捕捉"平均风格"
- `std`：捕捉"风格波动幅度"（同一笔画内速度/笔压的方差）
- `max`：捕捉"最显著的笔画特征"（突出尖峰）

把三者拼接相当于一个手工设计的"风格统计向量"，比单一 mean 信息量大得多，对小样本（90 人）的分类更稳。
masked 版本保证 padding 位置不污染统计量。

**这个 loss 给 encoder 什么信号**：
梯度从 `ce_loss` → `MLP` → `LayerNorm` → 池化（梯度按 mask 平均/最大处分配）→ **回到 latent z** → 回到 `encoder`。
效果：encoder 必须让 **同一 writer 的不同字 latent 距离近、不同 writer 距离远**——
也就是把"风格指纹"压进 latent 空间的全局结构里。
推理时给前 N 字真实轨迹做 style prefix，DiT 拿到的 prefix latent 自然带上风格条件（详见 §6）。

### 2.4 训完 VAE 之后，OCR 头和 Writer 头还用吗？

**短答**：不用了。它们**只在阶段 1 训 VAE 的那段时间**被 forward 调用过；阶段 2 训 DiT 不调，最终推理也不调。

**完整对照表**：

| 阶段 | `encoder` | `conv_mu/logvar` | `decoder + transformer_decoder` | `ocr_model` | `style_classifier` |
|---|---|---|---|---|---|
| 阶段 1 训 VAE | 训练（更新） | 训练（更新） | 训练（更新） | **训练（更新）** | **训练（更新）** |
| 阶段 2 训 DiT | 冻结，只 forward 给 latent | 冻结，只 forward | 不调用 | **不调用** | **不调用** |
| 推理 [test_local.py](test_local.py) | forward (`encode`) | forward (`encode`) | forward (`decode`) | **不调用** | **不调用** |

**那它们存在的意义就只剩"训练时塑造 encoder"？是的。**

- 它们的梯度在阶段 1 把 `encoder` 的权重塑造好——latent z 里**永久**留下了"哪段时间是哪个字"和"这是谁的风格"两类信息
- 阶段 1 结束、`encoder` 一旦定型，这两个头就完成使命了
- 阶段 2 训 DiT 时，`encoder` 给出的 latent 已经天然带上字符内容和风格指纹，DiT 只需在这个 latent 空间里学"怎么从噪声采样出合法的轨迹 latent"，根本不需要再去算 OCR loss / writer loss
- 推理同理，只走 `encode → DiT 采样 → decode` 三步

**那它们的权重为什么还留在 `vae_epoch_100.pt` 里？**

因为保存时是 `torch.save(vae.state_dict(), ...)`——一次性把整个 `VAE` 的所有子模块都序列化了，没人专门去 pop 掉这两个头。
所以 ckpt 文件里**确实**有 `ocr_model.*` 和 `style_classifier.*` 这些 key（也是第 7 节的验证方法），但加载到推理代码里之后，**没有任何一行代码会去 `vae.ocr_model(...)` 或 `vae.style_classifier(...)`**——它们只是静静占着内存，不参与任何计算。

> **打个比方**：辅助头就像**模具**——把陶器（encoder）烧制成想要的形状。
> 烧好之后陶器拿去用，模具仍放在角落里（仍存在 ckpt，但不再发力）。
> 你可以把它们从 ckpt 里删掉来减小文件体积，但功能上没区别。

---

## 3. 代码证据

看 `diffink/model/vae.py` 的 `VAE.__init__`（第 12-38 行）：

```python
class VAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        # --- 主路：Encoder 和 Decoder ---
        self.encoder = Encoder(...)
        self.conv_mu = nn.Conv1d(...)
        self.conv_logvar = nn.Conv1d(...)
        self.decoder = Decoder(...)
        self.transformer_decoder = TransformerDecoder(...)

        # --- 辅助头 1：OCR（内容监督）---
        self.ocr_model = ChineseHandwritingOCR(...)   # ← VAE 的子模块
        self.ctc = nn.CTCLoss(blank=0, zero_infinity=True)

        # --- 辅助头 2：Writer Classifier（风格监督）---
        self.style_classifier = WriterStyleClassifier(
            input_dim=config.style_classifier_dim,
            num_writers=config.num_writer,            # = 90
        )
```

三件事说明它们是联合训练的：

1. **都写成 `self.xxx = nn.Module(...)`** —— 成为 `VAE` 的成员，自动注册
   参数到 `VAE.parameters()`
2. **`nn.CTCLoss` 也写进 `__init__`**（第 34 行）—— 说明 CTC 监督就是
   VAE forward 里算的
3. **推理代码 `test_local.py` 只 `vae.encode()` / `vae.decode()`**——
   OCR 和 style_classifier 在推理时**根本不被调用**，它们的作用是"训练时
   给 encoder 梯度提示"，训完就只留下被塑造好的 encoder 权重

## 4. 为什么要这么设计？

单独训一个风格分类器，再用它打分的坏处：
- 两阶段训练，工程复杂
- 打分信号是标量，丢了很多结构信息（只告诉你"像/不像"，不告诉"哪里不像"）
- RLHF 里需要引入 PPO / GRPO 等复杂 RL 算法

联合训练（DiffInk 采用）的好处：
- 一次前向、一次反向，工程简单
- 梯度直接流进 encoder 的所有参数：
  - OCR 头逼 latent 保留**字是什么**的信息
  - Writer 头逼 latent 保留**谁写的**的信息
  - 重建头逼 latent 保留**轨迹长什么样**的信息
- 相当于"用辅助任务塑造 encoder 的隐空间结构"，属于标准的
  **multi-task learning + auxiliary loss** 套路

这个思路不是 DiffInk 独创，在 VAE 类模型（LDM、VAE-GAN、语音合成 VAE
encoder 等）里都很常见。

### 4.1 怎么"确定"梯度塑造的是 encoder，而不是 head 自己变强？

> 这是 multi-task 训练里一个绕不开的"灵魂问题"——既然 OCR 头和 Writer 头会被 ② / ③ 反向更新，那有没有可能 **head 越练越强、把烂 latent 也硬解出来**，最后 encoder 反而没被塑造？

**短答**：理论上**没法用一个开关"强制保证"**这件事——multi-task 的本质就是 head 和 encoder 同时被更新。但实操上有 3 个机制让"塑造 encoder"成为大概率结果，加上论文的消融实验**事后**验证了这一点。

#### 机制 1：head 是刻意做小的（容量瓶颈）

论文原话称这两路是 **"lightweight regularization losses"** 和 **"lightweight modules"**——
- OCR head：一个浅层 Transformer（[ocr.py:22-51](diffink/model/ocr.py#L22-L51)）
- Writer Style Classifier：池化 + 3 层 MLP（[writer.py:5-45](diffink/model/writer.py#L5-L45)）

如果 latent z 信息不够，head 自身参数容量根本不足以"硬解"——它撑不住，梯度会反传逼回 encoder 去多保留有用信息。如果 head 做得很大很深，确实可能被 head 自己消化掉，那就退化成"head 在学任务，encoder 没被塑造"。**做小是关键的工程选择**。

#### 机制 2：辅助 loss 权重比主 loss 小（不喧宾夺主）

论文给的训练权重（来自论文 §4 实现细节）：

| 权重 | 数值 | 来源 |
|---|---|---|
| `λ_gmm` (高斯部分重建) | **1.0** | 主任务 |
| `λ_pen` (笔状态重建) | **2.0** | 主任务 |
| `λ_ocr` (CTC) | 1.0 | 辅助 |
| `λ_sty` (CE) | 0.5 | 辅助 |
| `λ_kl` (KL) | 1×10⁻⁶ | 极小,几乎不正则 |

注意：**主任务（重建）梯度强度 ≈ 1.0 + 2.0 = 3.0**，**辅助任务 ≈ 1.0 + 0.5 = 1.5**——主任务权重是辅助的 2 倍。
所以重建 loss 才是 encoder 的主要"塑形者"，辅助 head 只是"温和地往里加点结构信息"，不会主导优化方向。

#### 机制 3：encoder 比 head 大，训练期内 head 容易先饱和

encoder（3 层 conv + 3 个 4 层 ResidualStack）参数量远大于 OCR / Writer head。
训练早期 head 很快收敛（参数少 + loss 信号直接），之后 head 的梯度变小、几乎不再更新自己；
而 encoder 因为参数多 + 梯度被 4 路混合，需要更长时间才"调好"——
**结果就是后期梯度主要在塑造 encoder**。

#### 论文的事后证据：消融实验表 2

这是最有说服力的一点——论文在 **CASIA-OLHWDB 2.0–2.2** 上对比了 Vanilla VAE（无辅助 head）vs InkVAE（带 OCR + Writer 头）：

| 模型 | VAE 重建 AR ↑ | VAE 重建 DTW ↓ | 下游 DiT 生成质量 |
|---|---|---|---|
| Vanilla VAE | **97.59%** | **0.014** | 较差 |
| InkVAE（+OCR+Writer 辅助头） | **97.65%** | 0.016 | **大幅提升** |

**关键观察**：
- 重建指标几乎一样（AR 反而略升 0.06%、DTW 略升 0.002）→ **辅助 loss 没破坏 encoder 的重建能力**
- 但下游生成性能大幅领先 → **encoder 的 latent 空间结构确实被塑造得更利于生成**

这等于事后证明：**辅助 loss 的梯度的确流到了 encoder 上、改善了 latent 空间，而不是被 head 自己吃掉**。如果 head 把烂 latent 硬解出来了，重建会保持原样、但生成不会有提升。

#### 论文里**没有**显式讨论的（要承认的局限）

| 问题 | 论文是否讨论 |
|---|---|
| 是否用 `stop gradient` / `detach` 隔离 head 梯度 | ❌ 没有,默认让梯度全部回流 |
| head 容量的消融（小 vs 大）| ❌ 没有 |
| 损失权重敏感度分析（变化 λocr / λsty 会怎样）| ❌ 没有,只给了一组权重 |
| 显式的"信息瓶颈"理论分析 | ❌ 没有,只在直觉层面说"disentangle content and style" |

所以你这个问题在论文里**没有理论保证**，只有**经验观察**：选了合适的 head 容量、合适的权重，加上 latent z 维度本身有限（VAE 天然有信息瓶颈），最后训出来的 encoder 确实把"内容 + 风格"压进了 latent。这是"工程调通"的结果，不是"理论必然"。

> **一句话总结**：
> 多任务联合训练能否塑造好 encoder，依赖三件事——**head 不能太大、辅助权重不能太大、latent 维度不能太大**。
> 三者构成"信息瓶颈"，逼 encoder 把全部任务需要的信息都压进 latent。
> DiffInk 论文用消融实验事后证明这套配置 work，但没给出理论保证。

---

## 5. 两阶段训练：VAE → DiT

真正的"两阶段独立训练"发生在 VAE 和 DiT 之间：

```
阶段 1（vae_epoch_100.pt）：
    4 个 loss 联合训练 VAE（含 Encoder + Decoder + OCR + Writer 头）
    详见 §2.1

阶段 2（dit_epoch_1.pt）：
    冻结 VAE Encoder
    在 VAE 的 latent 空间上独立训练 DiT（扩散模型）
    loss 是 diffusion 的去噪 loss
    DiT 根本不知道 OCR 头和 Writer 头的存在
    详见 §5.1
```

这是 **Latent Diffusion Model (LDM)** 的标准配方：
- 先训个好 VAE 把高维数据压到低维 latent
- 再在 latent 空间训扩散模型（比直接在 pixel 空间训便宜很多）

### 5.1 阶段 2 训 DiT 时具体更新哪些层

**冻结**（`requires_grad_(False)` 且只调 `vae.encode()`）：
- `vae.encoder` / `conv_mu` / `conv_logvar` / `decoder` / `transformer_decoder`
- `vae.ocr_model` / `vae.style_classifier`（这两个**完全不参与**第 2 阶段，连 forward 都不调）

**更新**：[diffink/model/dit.py](diffink/model/dit.py) 内的所有模块
- `TextEmbedding`（字符内容条件）
- `TimestepEmbedding`（扩散步 t 的 embedding）
- `ConvPositionEmbedding`（latent 序列位置编码）
- 若干 `DiTBlock`（核心去噪 transformer）
- `AdaLayerNormZero_Final`（输出归一化）
- 旋转位置编码 `RotaryEmbedding`（如有可学习参数）

**损失**（标准 flow-matching / 去噪 loss，与 OCR / Writer 头无关）：
```
z_clean    = vae.encode(x).z                       # 冻结 VAE 给的 latent
z_noisy    = α(t)·z_clean + σ(t)·ε,  ε ~ N(0,I)
v_pred     = DiT(z_noisy, t, text_cond, prefix_cond)
L_dit      = MSE(v_pred, target)                   # target 为 ε 或速度场,按实现而定
```

> 关键差异：阶段 1 训练**编码空间**（teach encoder what to encode），
> 阶段 2 在固定的编码空间上训练**生成模型**（teach DiT how to sample）。

## 6. "像不像写字者"具体怎么体现？

推理时**没有**独立的 "style similarity 评分"。机制是隐式的：

1. 训练时 style_classifier 逼迫 encoder 产出的 latent 能区分 90 个 writer
   → latent 空间里"同一 writer 的不同字"距离近，"不同 writer"距离远
2. 推理时给前 N 字真实轨迹作 **style prefix**
   → encode 得到的 latent 就带着那个 writer 的"风格指纹"
3. DiT 以 prefix latent 为条件生成后续 latent
   → 因为 latent 空间被风格分开得很好，DiT 的生成自然"延续同一风格"

这也是你跑 `image.png` 效果差的一个重要解释：image 里的真人 writer
**不在训练的那 90 人里**，encoder 给它的 latent 是 OOD
→ DiT 的风格 prefix 条件也是 OOD → 生成 suffix 自然乱。

## 7. 怎么从 checkpoint 验证？

加载 `vae_epoch_100.pt` 看 state_dict 里的 key，应该能看到：

```
encoder.*               # 主 encoder 参数
conv_mu.*, conv_logvar.*
decoder.*               # 主 decoder 参数
transformer_decoder.*
ocr_model.*             # OCR 头参数     ← 证明它存在同一 ckpt 里
style_classifier.*      # Writer 头参数  ← 证明它存在同一 ckpt 里
```

所有参数都在一个文件 = 同时训练 / 同时保存。

---

## 总结对照表

| 维度 | "奖励模型"思路（你原以为） | DiffInk 实际做法 |
|------|---------------------------|------------------|
| 是否独立训练 | 是 | 否（与 VAE 联合） |
| 训练轮数 | 独立 epoch | 和 VAE 共享 epoch |
| 训练数据 | 独立数据（可能有 writer 标签+字符标签的数据） | 和 VAE 同一份数据 |
| 参数存放 | 独立 checkpoint 文件 | 和 VAE 同在 `vae_epoch_100.pt` |
| 推理时作用 | 给主模型打分 / RL 信号 | **不调用**，只留下被塑造的 encoder |
| 给主模型的信号 | 标量 reward | 梯度直接流进 encoder |
| 类比 | RLHF 里的 reward model | Multi-task learning 的 auxiliary head |
