# 1.9 前沿模型案例研究

!!! abstract "本章你将学到"
    - 如何阅读前沿模型的架构表，并把每一项翻译成它所代表的权衡。
    - 对 **Llama 3 405B、DeepSeek-V3、Qwen3-235B-A22B、gpt-oss-120b、Kimi K2 和 Gemma 3 27B** 做并排比较，只使用它们的技术报告和已发布模型配置中公布的数字。
    - 如何根据这些数字计算并比较它们的 KV cache 占用和训练算力。
    - 每家实验室的独特选择换来了什么，并把 1.1–1.8 章串联起来。

## 为什么重要 {#why-this-matters-the-problem}

现在你已经了解了现代 LLM 的每一个组件。真实的模型，是在种种约束之下对这些组件所做的组合：算力预算、硬件集群、服务成本目标、上下文长度目标、团队对风险的胃口。带着这样的视角去读技术报告，是跟上最新进展最快的办法，因为新模型大多只是已知思路的新*组合*和新*调优*。本章就在六个有影响力的 2024–2025 年模型上练习这项技能。

!!! note "来源与约定"
    数字来自各模型的技术报告或模型卡，精确的维度则来自随权重一起发布的配置文件。“—”表示该数字没有公布（或我们无法确认），而不是表示零。上下文长度是所支持的最大长度；“K”表示各实验室所用的 $2^{10}$ 个 token（128K = 131,072）。

## 直觉 {#intuition}

每个模型对同样三个问题给出了不同的回答：

1. **我们如何换取质量？** 更多参数（稠密或 MoE）、更多 token、更好的数据、更好的优化器。
2. **我们如何让训练保持稳定且负担得起？** 精度（BF16 还是 FP8）、并行布局、均衡方案、归一化技巧。
3. **我们如何让推理服务保持便宜？** 更少的激活参数（MoE）、更小的 KV cache（GQA、MLA、滑动窗口）、对量化友好的设计。

一个稠密的 405B 模型（Llama 3）把全部筹码押在 (1) 和稳定性上；DeepSeek-V3 和 Kimi K2 借助 MLA 和极度稀疏的 MoE 在 (3) 上用力极深；gpt-oss 针对单块 GPU 上的部署做了优化；Gemma 3 则瞄准在人们实际拥有的加速器上获得出色的质量。

!!! tip "读下去之前先试一试"
    对每个模型，仅凭这张表预测：

    - 它的权重在 BF16 下大约需要多少 GPU 显存（2 字节 × 总参数量）；
    - 每生成一个 token 要花多少计算量（$2 \times$ 激活参数量）；
    - 它在 128K 上下文下的 KV cache，能否与权重一起舒适地放进一台 8 卡节点。

    然后对照下面的实例演算检查你的答案。

## 工作原理：对比表 {#how-it-works-the-comparison-table}

| | Llama 3 405B | DeepSeek-V3 | Qwen3-235B-A22B | gpt-oss-120b | Kimi K2 | Gemma 3 27B |
|---|---|---|---|---|---|---|
| 类型 | 稠密 | MoE | MoE | MoE | MoE | 稠密 |
| 层数 | 126 | 61 | 94 | 36 | 61 | 62 |
| $d_{\text{model}}$ | 16,384 | 7,168 | 4,096 | 2,880 | 7,168 | 5,376 |
| query 头数 / KV 头数 | 128 / 8 | 128 / — (MLA) | 64 / 4 | 64 / 8 | 64 / — (MLA) | 32 / 16 |
| 头大小 | 128 | 128（q/k 另有 64 个 RoPE 维度） | 128 | 64 | 128（q/k 另有 64 个 RoPE 维度） | 128 |
| 注意力 | GQA | MLA（latent 512） | GQA + QK-norm | GQA；全注意力 / 128 token 带状窗口交替；可学习的 sink | MLA（latent 512） | GQA + QK-norm；5 个局部（窗口 1024）: 1 个全局 |
| FFN | SwiGLU，$d_{ff}$ 53,248 | 前 3 层稠密；之后 256 个路由专家 + 1 个共享专家，top-8，专家隐藏维度 2,048 | 128 个专家，top-8，无共享专家，专家隐藏维度 1,536 | 128 个专家，top-4，专家隐藏维度 2,880 | 第 1 层稠密；之后 384 个路由专家 + 1 个共享专家，top-8，专家隐藏维度 2,048 | GeGLU，$d_{ff}$ 21,504 |
| 总参数 / 激活参数 | 405B / 405B | 671B / 37B | 235B / 22B | ~117B / ~5.1B | 1.04T / 32B | 27B / 27B |
| 词表 | 128K | 128K（129,280） | 151,936 | 201,088（`o200k_harmony`） | 160K（163,840） | 262K |
| 上下文 | 128K | 128K | 原生 32K，配合 YaRN 达 128K | 128K（从 4K 起用 YaRN） | 128K | 128K |
| 预训练 token 数 | 15.6T | 14.8T | 36T | — | 15.5T | 14T |
| RoPE 底数 | 500,000 | 10,000（+YaRN） | 1,000,000 | 150,000（+YaRN） | 50,000（+YaRN） | 全局 1M / 局部 10K |
| 优化器 / 精度要点 | AdamW，BF16 | FP8 混合精度；无辅助损失的均衡；MTP | 全局批次均衡损失 | 发布时 MoE 权重已做 MXFP4 量化 | MuonClip | 从更大的教师模型蒸馏 |

这六个模型都使用 Pre-norm RMSNorm、RoPE 和门控 FFN。差异在于注意力（如何缩小 KV cache）、FFN（稠密还是稀疏到什么程度），以及训练技巧。

### 阅读任何模型配置的检查清单 {#a-checklist-for-reading-any-model-config}

新模型发布时，它的 `config.json` 和报告能回答大部分架构问题。请按顺序过一遍这些字段，并对每一项问：它花了什么代价，又换来了什么：

1. **`vocab_size`** → 分词粒度和多语言覆盖；嵌入层和输出层的大小（见 [1.2](02-tokenization.md)）。
2. **`num_hidden_layers`、`hidden_size`** → 深度和宽度；再加上 FFN 的大小，就得到参数量（见 [1.3](03-transformer.md)）。
3. **`num_attention_heads`、`num_key_value_heads`、`head_dim`**（对 MLA 则是 `kv_lora_rank`、`qk_rope_head_dim`）→ 注意力类型和每个 token 的 KV cache 大小（见 [1.4](04-modern-block.md)）。
4. **`sliding_window`、层类型列表** → 局部/全局交错，以及缓存中有多大一部分是有上限的（见 [1.6](06-long-context-and-efficient-attention.md)）。
5. **`rope_theta`、`rope_scaling`、`max_position_embeddings`** → 位置频率，以及上下文是否用 YaRN 或类似方法扩展过（见 [1.6](06-long-context-and-efficient-attention.md)）。
6. **`intermediate_size`、`moe_intermediate_size`、路由/共享专家数量、每个 token 的专家数、最前面的稠密层数** → FFN 是稠密还是稀疏，总参数与激活参数（见 [1.5](05-mixture-of-experts.md)）。
7. **与归一化相关的字段**（`rms_norm_eps`、QK-norm 开关）、**`tie_word_embeddings`**、偏置 → 稳定性与小模型的经济性（见 [1.4](04-modern-block.md)）。
8. **只在报告里才有的信息**：训练 token 数、数据配比、优化器、精度、并行方式、调度（见 [1.7](07-scaling-laws-and-data.md)、[1.8](08-training-at-scale.md)）。

### 设计选择及其换来的东西 {#design-choices-and-what-they-buy}

| 设计选择 | 谁在用（本章范围内） | 换来了什么 | 付出了什么 | 章节 |
|---|---|---|---|---|
| 稠密 FFN | Llama 3 405B、Gemma 3 27B | 简单、训练稳定、易于部署 | 单 token 计算量随全部参数增长 | [1.3](03-transformer.md) |
| 细粒度 MoE | DeepSeek-V3、Qwen3、Kimi K2、gpt-oss | 以很小的激活计算量获得大量参数 | 所有专家占用的内存、all-to-all 通信、均衡 | [1.5](05-mixture-of-experts.md) |
| 共享专家 | DeepSeek-V3、Kimi K2 | 通用知识集中在一处；路由专家得以专业化 | 每层多一个始终开启的 FFN | [1.5](05-mixture-of-experts.md) |
| KV 头很少的 GQA | Llama 3、Qwen3、gpt-oss、Gemma 3 | 更小的 KV cache、更快的解码 | KV 头减少时，质量有小幅风险 | [1.4](04-modern-block.md) |
| MLA | DeepSeek-V3、Kimi K2 | 在 MHA 水平的质量下，全注意力缓存最小 | kernel 更复杂；需要解耦的 RoPE | [1.4](04-modern-block.md) |
| 局部/全局交错 | gpt-oss、Gemma 3 | 大部分层的缓存和计算都有上限 | 局部层的远距离访问更弱 | [1.6](06-long-context-and-efficient-attention.md) |
| QK-norm / QK-clip | Qwen3、Gemma 3 / Kimi K2 | 注意力 logit 不会爆炸 | 小额开销；QK-norm 与 MLA 配合不太方便 | [1.8](08-training-at-scale.md) |
| 较大的 RoPE 底数和/或 RoPE 重新缩放（YaRN、插值） | 六个模型都用，只是组合不同 | 主要靠短上下文预训练就获得长上下文 | 需要一个长上下文训练阶段 | [1.6](06-long-context-and-efficient-attention.md) |
| 极大的 token 预算 | Qwen3（36T） | 更好的小型和中型模型 | 数据获取和过滤的投入 | [1.7](07-scaling-laws-and-data.md) |
| FP8 训练 | DeepSeek-V3 | 矩阵乘法吞吐量约为 BF16 的 ~2× | 需要谨慎缩放以避免精度损失 | [1.8](08-training-at-scale.md) |
| Muon 优化器 | Kimi K2 | 算力效率优于 AdamW（据 Moonlight） | 较新，需要稳定化手段（QK-clip） | [1.8](08-training-at-scale.md) |

## 代码 {#code}

下面的脚本把这张表变成对成本最重要的两个数字：每个 token 的 KV cache（服务）和训练 FLOPs（$C \approx 6 N_{\text{active}} D$，见 [1.7](07-scaling-laws-and-data.md)）。

```python
# 根据已公布的配置计算每个 token 的 KV cache 和训练算力（BF16 缓存，每元素 2 字节）。
MODELS = {
    # 每个全注意力层、每个 token 的缓存元素数；全注意力层 / 窗口层的层数；窗口大小
    "Llama 3 405B":       dict(kv_per_layer=2 * 8 * 128,  full=126, local=0,  window=0),
    "DeepSeek-V3":        dict(kv_per_layer=512 + 64,     full=61,  local=0,  window=0),   # MLA latent + RoPE key
    "Qwen3-235B-A22B":    dict(kv_per_layer=2 * 4 * 128,  full=94,  local=0,  window=0),
    "gpt-oss-120b":       dict(kv_per_layer=2 * 8 * 64,   full=18,  local=18, window=128),
    "Kimi K2":            dict(kv_per_layer=512 + 64,     full=61,  local=0,  window=0),   # MLA
    "Gemma 3 27B":        dict(kv_per_layer=2 * 16 * 128, full=10,  local=52, window=1024),
}
ACTIVE_AND_TOKENS = {  # （激活参数, 训练 token 数），两者都有公布的才列出
    "Llama 3 405B": (405e9, 15.6e12), "DeepSeek-V3": (37e9, 14.8e12),
    "Qwen3-235B-A22B": (22e9, 36e12), "Kimi K2": (32e9, 15.5e12), "Gemma 3 27B": (27e9, 14e12),
}

def kv_cache_bytes(cfg, context, bytes_per_elem=2):
    full = cfg["full"] * context
    local = cfg["local"] * min(context, cfg["window"])
    return (full + local) * cfg["kv_per_layer"] * bytes_per_elem

print(f"{'model':18s} {'KiB/token':>10s} {'GiB @128K':>10s} {'train FLOPs':>12s}")
for name, cfg in MODELS.items():
    per_tok = cfg["full"] * cfg["kv_per_layer"] * 2 / 1024
    at_128k = kv_cache_bytes(cfg, 131072) / 2**30
    flops = ""
    if name in ACTIVE_AND_TOKENS:
        n, d = ACTIVE_AND_TOKENS[name]
        flops = f"{6 * n * d:.1e}"                      # C ≈ 6 N_active D
    print(f"{name:18s} {per_tok:10.1f} {at_128k:10.1f} {flops:>12s}")
```

gpt-oss 的层划分（18 个全注意力层 + 18 个带状层）遵循它的交替模式；对于 Gemma 3 27B，其配置是每五个局部层之后放一个全局层，这样 62 层中就有 10 个全局层和 52 个局部层。

## 实例演算 {#worked-example}

脚本的输出，BF16 缓存，一条 128K-token 的序列：

| 模型 | 每个 token 的 KV cache（全注意力层） | 128K 下的 KV cache | 训练算力 $6 N_{\text{active}} D$ |
|---|---|---|---|
| Llama 3 405B | 504 KiB | 63.0 GiB | $3.8 \times 10^{25}$ |
| DeepSeek-V3 | 68.6 KiB | 8.6 GiB | $3.3 \times 10^{24}$ |
| Qwen3-235B-A22B | 188 KiB | 23.5 GiB | $4.8 \times 10^{24}$ |
| gpt-oss-120b | 36 KiB | 4.5 GiB | — |
| Kimi K2 | 68.6 KiB | 8.6 GiB | $3.0 \times 10^{24}$ |
| Gemma 3 27B | 80 KiB | 10.4 GiB | $2.3 \times 10^{24}$ |

读懂这些数字：

- **Llama 3 405B** 用了约 DeepSeek-V3 的 11 倍训练算力（$3.8 \times 10^{25}$ 对 $3.3 \times 10^{24}$），尽管 DeepSeek-V3 的总参数更多，原因是每个 token 都要经过全部 405B 个参数。对 Llama 3 的 $6ND$ 估算，与其报告中的 $3.8 \times 10^{25}$ FLOPs 吻合。
- **MLA** 使 DeepSeek-V3 和 Kimi K2 每个 token 的缓存，比 Llama 3 405B 的 GQA-8 小约 7×。
- **gpt-oss** 的缓存最小：它有一半的层被限制为 128 token 的带状窗口，其全注意力层使用 8 个大小为 64 的 KV 头。
- **Gemma 3** 的 5:1 局部:全局模式，把它的缓存从约 62 GiB（假如每一层都是全局的）降到了约 10 GiB。
- 每生成一个 token，推理计算量约为 $2 N_{\text{active}}$：Llama 3 405B 是 810 GFLOPs，DeepSeek-V3 是 74，Qwen3-235B-A22B 是 44，gpt-oss-120b 约为 10。

**根据配置重建参数量。** 与 [1.3](03-transformer.md) 和 [1.5](05-mixture-of-experts.md) 同样的核算方法，就能还原出这些头条数字。（归一化权重和偏置可以忽略，故省略。）

*Llama 3 405B*（$L = 126$，$d = 16{,}384$，128 个 query 头和 8 个大小为 128 的 KV 头，SwiGLU $d_{ff} = 53{,}248$，$V = 128{,}256$，输入输出不共享）：

| 组成部分 | 计算 | 参数量 |
|---|---|---|
| 每层注意力 | $W_Q, W_O$：$2 \times 16{,}384^2$；$W_K, W_V$：$2 \times 16{,}384 \times 1024$ | 570.4M |
| 每层 FFN | $3 \times 16{,}384 \times 53{,}248$ | 2,617.2M |
| 126 层 | $126 \times 3{,}187.7\text{M}$ | 401.6B |
| 嵌入层 + 输出层 | $2 \times 128{,}256 \times 16{,}384$ | 4.2B |
| **合计** | | **≈ 405.8B** |

请注意 GQA 如何使 $W_K$ 和 $W_V$ 比 $W_Q$ 小 16 倍，以及 FFN 占据了每层参数的 80% 以上。

*Qwen3-235B-A22B*（$L = 94$，$d = 4096$，64 个 query 头和 4 个大小为 128 的 KV 头，128 个隐藏维度为 1,536 的专家，top-8，$V = 151{,}936$，输入输出不共享）：

| 组成部分 | 计算 | 总参数 | 激活参数 |
|---|---|---|---|
| 每层注意力 | $2 \times 4096 \times 8192 + 2 \times 4096 \times 512$ | 71.3M | 71.3M |
| 一个专家 | $3 \times 4096 \times 1536$ | 18.9M | |
| 每层专家 | $128 \times 18.9\text{M}$（激活 8 个） | 2,415.9M | 151.0M |
| 每层路由器 | $4096 \times 128$ | 0.5M | 0.5M |
| 94 层 | | 233.8B | 20.9B |
| 嵌入层 + 输出层 | $2 \times 151{,}936 \times 4096$ | 1.2B | 1.2B |
| **合计** | | **≈ 235.1B** | **≈ 22.2B** |

“235B-A22B”这个名字，恰好就是这样算出来的。请注意，query 投影的宽度是 $d$ 的两倍（$64 \times 128 = 8192$），这一选择把注意力宽度与残差流宽度解耦了。

## 实践中的做法（2025–2026）：各项设计换来了什么 {#in-practice-20252026-what-each-design-buys}

### Llama 3 405B：认真做好的稠密基线 {#llama-3-405b-the-dense-baseline-done-carefully}

Meta 有意选择了**标准的稠密 Transformer**而不是 MoE，目的是“最大化训练稳定性”，并把投入放在数据（15.6T token，配比经过精心调优，见 [1.7](07-scaling-laws-and-data.md)）、128K 的分词器（见 [1.2](02-tokenization.md)）、用于长上下文的 500,000 大 RoPE 底数（见 [1.4](04-modern-block.md)），以及在 16,384 块 H100 上的 4D 并行（见 [1.8](08-training-at-scale.md)）。它还会对打包进同一条训练序列的不同文档之间屏蔽注意力。模型大小是由它自己的缩放定律拟合选定的（预测其预算下算力最优约为 402B）。这份报告是目前对稠密前沿流水线最完整的公开描述，包括分阶段的长上下文训练直至 128K，以及退火。权衡：最大程度的简单与稳健，代价是昂贵的推理（每个 token $2 \times 405\text{B}$ FLOPs 以及很大的 KV cache）。

### DeepSeek-V3：处处讲究效率 {#deepseek-v3-efficiency-everywhere}

几乎每个组件都是为成本而选的：用 **MLA** 获得小 KV cache（见 [1.4](04-modern-block.md)）；用 **细粒度 MoE**，256 个路由专家 + 1 个共享专家，671B 中激活 37B（见 [1.5](05-mixture-of-experts.md)）；用**无辅助损失的均衡**，使均衡不与语言建模损失冲突；用**节点受限路由**限制 all-to-all 的流量；用带细粒度缩放的 **FP8 训练**以及 **DualPipe** 调度（见 [1.8](08-training-at-scale.md)）。它还加入了**多 token 预测（MTP）**：一个额外的串行模块预测下下个 token，使训练信号更加稠密，并且在推理时可以复用为投机解码（speculative decoding）的草稿模型（报告引用了第二个 token 85–90% 的接受率；见 [第 3 部分](../part3/index.md)）。报告称完整训练耗费 2.788M 个 H800 GPU 小时。

### Qwen3-235B-A22B：干净、传统的 MoE {#qwen3-235b-a22b-a-clean-conventional-moe}

Qwen3 把**只有 4 个 KV 头的 GQA**（query 与 KV 的比例为 16:1）、**QK-norm**（并且不像 Qwen2 那样带 QKV 偏置），以及一个 128 个专家、top-8、**无共享专家**的 MoE 结合在一起，并用**全局批次**损失做均衡，使专家可以按领域专业化。它最有特色的押注是数据量：该系列中每个模型都用 **36T token**，包括分阶段的预训练（含一个重推理的阶段和一个长上下文阶段）。权衡：设计简单、被充分理解，数据量非常大，GQA 的缓存比 MLA 的大，但更容易实现和部署。

### gpt-oss-120b：为部署而设计 {#gpt-oss-120b-designed-to-deploy}

OpenAI 的开放权重模型激活计算量很小（约 117B 中激活约 5.1B），并为部署服务做了工程设计：**全注意力与 128 token 带状注意力交替**，使长距离注意力的开销和缓存减半；每个头有**可学习的 attention sink**，让头可以“什么都不关注”，这对窗口注意力有帮助（见 [1.6](06-long-context-and-efficient-attention.md)）；**8 个 KV 头的 GQA**；以及以 **MXFP4**（4 比特）发布的 MoE 权重，据其模型卡称，这使 120b 模型能放进单块 80 GB 的 GPU。它使用 `o200k_harmony` 分词器，带有用于其聊天格式的特殊 token。训练 token 数没有公布。

### Kimi K2：扩大稀疏度与优化器 {#kimi-k2-scaling-sparsity-and-the-optimizer}

Kimi K2 从一个类似 DeepSeek-V3 的设计出发，改了三处。它把稀疏度提高到 **384 个专家**（top-8，1 个共享），总参数达到约 1.04T、激活 32B，其动机是它发现，在激活计算量固定时，专家越多，损失会持续下降（见 [1.5](05-mixture-of-experts.md)）。它把注意力头数**减半**到 64，以降低长智能体（agentic）上下文的推理成本。并且它用 **MuonClip** 训练：Muon 优化器加上 QK-clip，后者在注意力 logit 增长过大时重新缩放 query/key 权重（见 [1.8](08-training-at-scale.md)）。报告描述了 15.5T token 的预训练过程没有出现一次损失尖峰，并大量使用改写（合成）数据来延展高质量 token。

### Gemma 3：每块加速器上的质量 {#gemma-3-quality-per-accelerator}

Gemma 3 面向能在单块 GPU 或 TPU 上运行的模型。27B 模型使用**每个全局层配 5 个局部层（1024 token 窗口）**，使 KV cache 在 128K 上下文下仍然很小；**QK-norm**（取代 Gemma 2 的 logit 软截断）；局部层（10K）和全局层（1M）使用不同的 RoPE 底数；为多语言覆盖而设的 **262K 大词表**（见 [1.2](02-tokenization.md)）；以及预训练期间从更大的教师模型做**知识蒸馏**（蒸馏见 [第 2 部分](../part2/index.md)）。它还是多模态的，带一个基于 SigLIP 的视觉编码器。权衡：在这个规模上用稠密计算换取简单，同时对注意力在内存方面做了激进的工程设计。

### 贯穿各模型的趋势 {#cross-cutting-trends}

- **Pre-RMSNorm + RoPE + 门控 FFN + GQA 或 MLA** 是通用的骨架（见第 [1.3](03-transformer.md)–[1.4](04-modern-block.md) 章）。
- **细粒度专家的 MoE** 在前沿占主导；在面向单设备部署的规模上，稠密模型仍然常见。
- **缩小 KV cache** 在每一个层面都在推进：更少的 KV 头、潜变量缓存、局部窗口、sink；接下来是混合线性注意力（Qwen3-Next、MiniMax-01）和可训练的稀疏注意力（DeepSeek-V3.2）。
- **Token 数：15–36T**，配合大量过滤、合成改写和退火（见 [1.7](07-scaling-laws-and-data.md)）。
- **精度与优化器**如今已成为有竞争力的差异化因素：FP8 训练（DeepSeek-V3）、Muon（Kimi K2）、4 比特发布格式（gpt-oss）。
- **Llama 4**（2025，仅有博客）让 Meta 也转向了 MoE（Scout 和 Maverick），具备原生多模态，并用交错的 RoPE/NoPE 层（“iRoPE”）来处理长上下文。

## 常见误区 {#common-pitfalls-misconceptions}

- **用总参数量来比较 MoE 和稠密模型。** 一个 671B、激活 37B 的 MoE，训练成本远低于 405B 的稠密模型；比较计算时看激活参数，比较内存时看总参数。
- **以为词表或上下文的数字越大，模型就越好。** 这些都是权衡；质量取决于数据和训练。
- **把“128K 上下文”读成“128K 可用”。** 长上下文的质量各不相同；请查看大海捞针（needle-in-a-haystack）和长文档推理的结果。
- **在显存估算中忽略精度。** gpt-oss 能放进单块 GPU 的说法取决于 MXFP4 的专家权重；用 BF16 的话，约 117B 的权重本身就要约 234 GB，大约是三块 80 GB 的 GPU。
- **把模型卡上的数字当作可以互换的。** “128K”词表可能是 128,000、128,256 或 129,280，取决于特殊 token；当精确数字很重要时，请去读配置。

## 本章要点 {#key-takeaways}

- 六个前沿模型共享同一个骨架，差异在于注意力（GQA、MLA、局部/全局）、FFN 的稀疏度、数据量和训练技巧。
- 每个 token 的 KV cache 从约 36 KiB（gpt-oss-120b）到约 504 KiB（Llama 3 405B）不等，这在很大程度上决定了长上下文的服务成本。
- MoE 让 DeepSeek-V3、Qwen3-235B-A22B 和 Kimi K2 只需 Llama 3 405B 约十分之一的算力就能训练，同时拥有相当或更多的总参数。
- 每个独特的选择都对应一种约束：稳定性（Llama 3 稠密）、成本（DeepSeek-V3 的 MLA + FP8）、数据（Qwen3 的 36T）、部署（gpt-oss 的窗口 + MXFP4）、稀疏度与优化器（Kimi K2）、单加速器上的质量（Gemma 3）。
- 学完 1.1–1.8 章，你就能根据任何配置文件自己算出这些权衡。

## 自测 {#check-your-understanding}

??? question "为什么 Qwen3-235B-A22B 每个 token 的 KV cache 比 DeepSeek-V3 大，尽管它的总参数少得多？"
    KV cache 的大小取决于注意力的设计，而不是参数量。Qwen3 缓存 4 个 KV 头 × 128 维 × 2（K 和 V）= 每层 1,024 个元素，共 94 层；DeepSeek-V3 缓存的是 576 个元素的 MLA latent，共 61 层。

??? question "估算一个假设的 MoE 的训练算力：20B 激活参数，在 25T token 上训练。"
    $6 \times 20 \times 10^9 \times 25 \times 10^{12} = 3 \times 10^{24}$ FLOPs，与 DeepSeek-V3 或 Kimi K2 大致相同。

??? question "gpt-oss 的可学习 attention sink 解决了什么问题？为什么它对其带状层格外重要？"
    softmax 强制权重之和为 1，所以一个没有相关内容可关注的头也必须把权重放到某个地方。可学习的 sink logit 给了它一个“无处可去”的选项。在带状层中，通常隐含的 sink（第一个 token）落在了窗口之外，所以显式的 sink 就格外有用。

??? question "Kimi K2 把注意力头数相比 DeepSeek-V3 减半了。既然 MLA 的缓存与头数无关，这样做能省下什么？"
    prefill 和解码时的注意力计算量（每个头都要对整个上下文计算分数），以及 query 和输出投影的大小。对长的智能体上下文，即使缓存很小，每个 token 的注意力计算量仍是一项重要成本。

??? question "尽管 MoE 在计算上有优势，Meta 为什么可能还是为 Llama 3 405B 选择了稠密模型？"
    报告给出的理由是训练稳定性。稠密模型避免了路由坍缩、均衡损失、all-to-all 通信和专家并行的工程，这降低了一次非常大、持续数月的训练的风险。

## 延伸阅读 {#further-reading}

| 报告 | 看什么 |
|---|---|
| :material-star: [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) — Llama Team, Meta, 2024 | 90 多页，涵盖完整的稠密 405B 流水线：数据配比、由缩放定律驱动的规模选择、4D 并行、长上下文阶段、后训练。关于“前沿实验室如何做”的最佳单一参考。 |
| :material-star: [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — DeepSeek-AI, 2024 | MLA + 细粒度 MoE + 无辅助损失均衡 + MTP + FP8 + DualPipe。V3.1/V3.2 后来加入了混合思考和 DeepSeek Sparse Attention。 |
| :material-star: [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) — Alibaba, 2025 | 稠密与 MoE 系列（0.6B → 235B-A22B）、36T token、QK-norm、统一的“思考 / 非思考”模式。Qwen3-Next 加入了混合 Gated DeltaNet + 极度稀疏的 MoE。 |
| :material-new-box: [gpt-oss-120b & gpt-oss-20b Model Card](https://arxiv.org/abs/2508.10925) — OpenAI, 2025 | MoE、稠密/带状窗口注意力交替、attention sink、MXFP4 量化的 MoE 权重、`o200k_harmony` 分词器。 |
| :material-new-box: [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534) — Moonshot AI, 2025 | 用 MuonClip 训练的 1T 总参数 / 32B 激活的 MoE；稀疏度缩放；大量智能体数据合成。 |
| :material-new-box: [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786) — Google DeepMind, 2025 | 5:1 的局部:全局注意力以缩小 KV cache、QK-norm、从更大教师模型蒸馏、多模态。 |
| :material-book-open-variant: Llama 4 ([blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)) — Meta, 2025 | MoE（Scout / Maverick）、通过早期融合实现的原生多模态、用于长上下文的交错 RoPE/NoPE 层（“iRoPE”）。 |
| :material-book-open-variant: [2 OLMo 2 Furious](https://arxiv.org/abs/2501.00656) — AI2, 2025 | 端到端完全可复现；最适合学习，因为数据、代码和检查点全部发布。 |
| :material-book-open-variant: [Better & Faster LLMs via Multi-token Prediction](https://arxiv.org/abs/2404.19737) — Gloeckle et al., 2024 | 额外的头预测多个未来 token：更稠密的训练信号，以及用于投机解码的免费草稿模型（DeepSeek-V3 采用）。 |
| :material-book-open-variant: [The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) — Sebastian Raschka | 对近期架构持续更新、配有插图的对比；本章的好搭档。 |
| :material-book-open-variant: [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) — Hugging Face | 从实践者角度，端到端讲解训练一个现代 LLM 背后的各项决策。 |
