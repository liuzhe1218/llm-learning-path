# 第三部分 · 推理系统与基础设施

训练一个模型只需要一次，而服务它却要重复数十亿次。第三部分讲的就是模型生命周期的后半段：如何把训练好的 Transformer 变成一个能同时回答大量用户、响应够快、成本又划算的服务。你会看到，LLM 推理服务首先是一个**内存问题**。GPU 每秒能做的算术运算，远远多于它能从显存搬运的字节数，而逐 token 生成文本又恰好不断迫使它去搬字节。这一部分里几乎所有技术，要么是**少读字节**（量化（quantization）、更小的 KV cache、MoE），要么是**每读一个字节多做有用的活**（批处理、投机解码（speculative decoding）、融合 kernel）。

我们从最基础的地方讲起。3.1 章用十几行 Python 写出生成循环，再借助 roofline 模型解释为什么这个循环只能跑到 GPU 峰值性能的一小部分。3.2–3.4 章讲每个现代推理引擎都建立在其上的三个核心思想：KV cache、分页内存配合连续批处理（continuous batching），以及 FlashAttention 这类 IO 感知的 attention kernel。3.5–3.6 章讲两个最大的“免费午餐”——投机解码和量化，包括背后的证明与误差分析。3.7 章扩展到多 GPU、多机器（张量并行/专家并行、预填充/解码分离（PD 分离）、万亿参数 MoE 的推理服务），3.8 章最后讲怎么选引擎、怎么诚实地做 benchmark、怎么做容量规划，以及怎么计算每百万 token 的成本。

即使你从不写 CUDA kernel，这部分也很重要。推理成本决定了哪些模型在经济上可部署、推理模型可以“思考”多久、RL 训练循环生成 rollout 能有多快，也决定了用户感受到的延迟。能熟练地做这部分里的估算，是应用 LLM 工作中最有用的技能之一。

## 前置知识与学习时间 {#prerequisites-and-time}

- **数学：** 线性代数（矩阵乘法的形状与 FLOP 计数）、基础概率（分布、期望、采样）。
- **代码：** Python 以及基础的 PyTorch / NumPy。
- **模型：** [第一部分](../part1/index.md)中的 Transformer 架构——attention、多头 / 分组查询注意力、MLP 块、MoE。不需要学过第二部分。
- **时间：** 读完所有章节并跑通代码大约需要 15–25 小时；动手项目再加 10–20 小时。

## 章节地图 {#chapter-map}

| # | 章节 | 一句话简介 |
|---|---|---|
| 3.1 | [推理基础](01-inference-fundamentals.md) | 生成循环、预填充（prefill）与解码（decode）、GPU 内存层级、roofline 模型、延迟指标、采样与约束解码。 |
| 3.2 | [KV cache](02-kv-cache.md) | 为什么要缓存 key 和 value、KV 大小公式、它如何限制批大小，以及如何压缩它（GQA/MLA、量化、驱逐）。 |
| 3.3 | [批处理与内存管理](03-batching-and-memory-management.md) | 连续批处理、PagedAttention 块表、写时复制、分块预填充、前缀缓存（prefix caching）/ RadixAttention、调度与抢占。 |
| 3.4 | [Attention kernel](04-attention-kernels.md) | FlashAttention 的分块与在线 softmax（附 NumPy 实现）、Flash-Decoding、FlashInfer、CUDA graph、kernel 融合。 |
| 3.5 | [投机解码](05-speculative-decoding.md) | 先起草后验证、拒绝规则及其无损性证明、期望加速比、Medusa / EAGLE / MTP 草稿、树状验证。 |
| 3.6 | [量化](06-quantization.md) | 逐位讲解数值格式、量化数学、量化粒度、离群值、GPTQ / AWQ / SmoothQuant、FP8 与 FP4，以及如何评估。 |
| 3.7 | [分布式与分离式推理服务](07-distributed-and-disaggregated-serving.md) | 张量 / 流水线 / 专家并行、预填充–解码分离、以 KV cache 为中心的系统、大规模 MoE 服务、路由与自动扩缩容。 |
| 3.8 | [引擎与生产实践](08-engines-and-production-practice.md) | vLLM、SGLang、TensorRT-LLM、llama.cpp 等；诚实的 benchmark；容量规划；每 token 成本；推理模型与 RL rollout 服务。 |

## 心智模型：一个请求的一生 {#mental-model-the-life-of-a-request}

这一部分的每一章都在优化下图中的某个方框。请把它当作地图记在心里。

```text
 client ──HTTP/gRPC──▶ ROUTER / LOAD BALANCER ──────────────────────────────┐
  "Explain RoPE"        选择一个副本（负载最低，                             │
                        或前缀感知：我的 KV 缓存在哪？）          [3.7]      │
                                                                             ▼
                      ┌──────────────────── 一个引擎副本 ───────────────────────────┐
                      │ tokenizer ─▶ 等待队列                                       │
                      │                 │                                           │
                      │                 ▼                                           │
                      │ 调度器（SCHEDULER，每一步都运行）                  [3.3]     │
                      │   - 有空闲 KV 块时准入请求                                  │
                      │   - 复用已缓存的前缀块（radix tree）                        │
                      │   - 在一个 batch 里混合预填充分块 + 解码 token              │
                      │   - 显存耗尽时抢占（换出 / 重算）                           │
                      │                 │                                           │
                      │                 ▼                                           │
                      │ 模型执行器（GPU，可能跨 GPU 做 TP/EP）          [3.4,3.7]   │
                      │   PREFILL: 整个 prompt 并行 ─▶ 写入 KV cache                │
                      │            计算受限；决定 TTFT                      [3.1]   │
                      │   DECODE:  每步每个序列生成 1 个新 token                    │
                      │            读取全部权重 + KV；显存带宽受限；决定 TPOT       │
                      │            （+ 投机草稿 [3.5]，量化 [3.6]）                 │
                      │                 │ logits                                    │
                      │                 ▼                                           │
                      │ 采样器（temperature / top-p / 语法掩码）          [3.1]     │
                      │                 │ token id                                  │
                      │                 ▼                                           │
                      │ DETOKENIZER ─▶ 把文本块流式返回给客户端（SSE）              │
                      │   循环直到 EOS / max_tokens，然后释放 KV 块                 │
                      └─────────────────────────────────────────────────────────────┘
                      （在分离式服务中，PREFILL 和 DECODE 运行在不同的
                       GPU 池上，KV cache 在两者之间传输            [3.7]）
```

各章之间的依赖关系：

```text
3.1 基础（roofline，prefill/decode）
  ├─▶ 3.2 KV cache ──▶ 3.3 分页与批处理 ──▶ 3.7 分布式 / 分离式
  │                    └─▶ 3.4 attention kernel
  ├─▶ 3.5 投机解码（利用 decode 中空闲的算力）
  └─▶ 3.6 量化（少读字节）
                                 all ──▶ 3.8 引擎、benchmark、成本
```

## 如何学习这一部分 {#how-to-study-this-part}

1. **认真读 3.1 和 3.2。** roofline 论证和 KV cache 公式在之后每一章都会用到。先自己拿计算器做一遍例题，再看答案。
2. **把 3.3 和 3.4 放在一起读。** 3.3 是“操作系统”（谁来运行、内存放在哪里）；3.4 是“CPU 指令集”（在这块内存上 attention 是怎么算的）。运行块分配器模拟和 NumPy 版 FlashAttention，并核对输出是否一致。
3. **3.5 和 3.6 可以任意顺序阅读。** 两者都是想从每个字节里榨出更多价值；两章都有值得亲手推一遍的短证明。
4. **3.7 和 3.8 是系统类章节。** 建议在你至少部署过一次模型之后再读。

动手项目（至少做三个）：

- [ ] 为你选定的模型，在单张 GPU 上计算 KV cache 大小和最大批大小（3.2 章）。
- [ ] 用 vLLM 和 SGLang 各部署一个 7–8B 模型；扫描并发度，画出 TTFT/TPOT 分位数随吞吐变化的曲线（3.8 章）。
- [ ] 对一个很长的共享 system prompt 打开前缀缓存，测量 TTFT 的下降（3.3 章）。
- [ ] 在你关心的任务上，对比 BF16、FP8 和 4-bit 仅权重量化 checkpoint 的速度*和*精度（3.6 章）。
- [ ] 启用投机解码（n-gram 或 EAGLE），测量 batch 1 和 batch 64 下的接受率与加速比（3.5 章）。
- [ ] 实现一个最小的分页 KV cache + 连续批处理循环；[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 是一个约 1,200 行的参考实现。

推荐的配套课程：CMU **[15-442/642 Machine Learning Systems](https://mlsyscourse.org/)** 和 **[GPU MODE](https://github.com/gpu-mode/lectures)** 系列讲座。推荐的配套综述：**[LLM Inference Handbook (BentoML)](https://github.com/bentoml/llm-inference-handbook)** 和 **[Transformer Inference Arithmetic (kipp.ly)](https://kipp.ly/transformer-inference-arithmetic/)**。
