# 第三部分 · 推理系统与基础设施

> **目标：** 理解为什么 LLM 推理服务本质上是一个*显存/访存*问题，现代推理引擎（vLLM、SGLang、TensorRT-LLM）如何榨出吞吐和延迟，以及 2025–2026 年的系统如何大规模部署万亿参数的 MoE 推理模型。

**图例：** :material-star: 必读 · :material-book-open-variant: 深入阅读 · :material-new-box: 2025–2026 前沿

---

## 3.1 全局视角

文本生成分为两个阶段：

| 阶段 | 工作内容 | 瓶颈 | 指标 |
|---|---|---|---|
| **预填充（Prefill）** | 并行处理整段 prompt，构建 KV cache | 计算受限（大矩阵乘法） | **TTFT（首 token 延迟）** — time to first token |
| **解码（Decode）** | 每次生成一个 token，每一步都要读取全部权重 + KV cache | **显存带宽受限** | **TPOT / ITL** — 每个输出 token 的耗时 |

粗略估算：batch size 为 1 时，解码速度 ≈ `HBM 带宽 / 每个 token 需读取的权重字节数`。一个 70B 的 BF16 模型（约 140 GB）跑在带宽约 3.35 TB/s 的 GPU 上，上限大约是 ~24 tokens/s。下面几乎所有技术，要么**少读字节**（量化、MoE、更小的 KV），要么**让每次读取服务更多 token**（批处理、投机解码）。

每个 token 的 **KV cache 大小** = `2 × layers × kv_heads × head_dim × bytes`。以 Llama-3-70B 为例（80 层、8 个 KV head、head 维度 128、BF16）：约 320 KB/token，128K token 的上下文就是约 40 GB。这就是 GQA/MLA（第一部分）和 KV 量化如此重要的原因。

入门推荐：**[LLM Inference Handbook (BentoML)](https://github.com/bentoml/llm-inference-handbook)** 和 **[Transformer Inference Arithmetic (kipp.ly)](https://kipp.ly/transformer-inference-arithmetic/)**。

---

## 3.2 批处理与显存管理 — 所有引擎的核心

| Paper | 摘要 |
|---|---|
| :material-star: [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) — Yu et al., OSDI 2022 | **迭代级调度（连续批处理，continuous batching）**：请求在每个解码步都可以加入或离开 batch，而不必等最慢的序列结束。吞吐成倍提升，如今已成为标配。 |
| :material-star: [Efficient Memory Management for LLM Serving with PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180) — Kwon et al., SOSP 2023 | 像操作系统的虚拟内存页一样，把 KV cache 存放在固定大小的 **block** 中 → 几乎没有碎片，并支持 beam search/并行采样场景下的 copy-on-write 共享。吞吐提升 2–4 倍。vLLM 背后的论文。 |
| :material-star: [SGLang: Efficient Execution of Structured LM Programs (RadixAttention)](https://arxiv.org/abs/2312.07104) — Zheng et al., 2023 | 用**基数树（radix tree）**管理 KV cache，在请求之间自动实现**前缀缓存（prefix caching）**（system prompt、few-shot、多轮对话、agent），另外还带有一个高速的约束解码引擎。 |
| :material-star: [Sarathi-Serve: Taming Throughput-Latency Tradeoff](https://arxiv.org/abs/2403.02310) — Agrawal et al., OSDI 2024 | **分块预填充（chunked prefill）**：把长 prompt 切成多个 chunk，搭载到解码 batch 里一起跑 → 解码不再被卡住，TPOT 保持稳定。vLLM v1 中默认开启。 |
| :material-book-open-variant: [FlashInfer: Efficient and Customizable Attention Engine for LLM Serving](https://arxiv.org/abs/2501.01005) — Ye et al., MLSys 2025 | 针对 paged/ragged KV 布局以及 cascade（共享前缀）attention 的 JIT 编译 attention kernel；vLLM 和 SGLang 内部都在使用。 |
| :material-book-open-variant: [Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) — Dao et al., 2023 (blog) | 解码时把 KV 序列维度切分到多个 thread block 上，让长上下文、小 batch 的解码也能用满整块 GPU。 |

---

## 3.3 投机解码（Speculative decoding）

核心思想：由一个廉价的**草稿（draft）**模型提出 *k* 个 token，再由大的**目标（target）**模型在一次前向中完成验证。由于解码受限于带宽，被接受的 token 几乎是"免费"的。拒绝采样（rejection sampling）保证输出分布与原模型**完全一致**。

| Paper | 摘要 |
|---|---|
| :material-star: [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — Leviathan et al., 2022 | 最初的算法及其无损性证明；2–3 倍加速。 |
| :material-book-open-variant: [Accelerating LLM Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — Chen et al. (DeepMind), 2023 | 同期独立推导出的方法；应用于 Chinchilla-70B。 |
| :material-book-open-variant: [SpecInfer: Tree-based Speculative Inference](https://arxiv.org/abs/2305.09781) — Miao et al., 2023 | 借助 tree attention，在一次前向中验证由候选续写构成的一整棵**树**。 |
| :material-star: [Medusa: Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — Cai et al., 2024 | 不需要单独的草稿模型：在目标模型上额外加几个 head 来预测后续 token；采用树状验证。 |
| :material-star: [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — Li et al., 2024 | 利用目标模型的隐状态，在**特征层面**做草稿预测 → 接受率大幅提高。 |
| :material-new-box: [EAGLE-3: Scaling up Inference Acceleration via Training-Time Test](https://arxiv.org/abs/2503.01840) — Li et al., 2025 | 多层特征融合，并在训练阶段模拟解码过程；约 3–6 倍加速；2025 年成为 SGLang/vLLM 中默认的投机解码方法。 |
| :material-book-open-variant: [Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) — Fu et al., 2024 | 无需草稿模型：用 Jacobi 迭代生成 n-gram，再并行验证。 |

预训练阶段训练出的 **MTP head**（DeepSeek-V3，见第一部分）可以直接充当内置的草稿模型——据报道第二个 token 的接受率约为 80–90%。

---

## 3.4 量化（Quantization）

| Paper | 摘要 |
|---|---|
| :material-book-open-variant: [LLM.int8()](https://arxiv.org/abs/2208.07339) — Dettmers et al., 2022 | 发现了 LLM 激活中的**离群特征（outlier features）**；通过混合精度分解，让这些离群部分保持 FP16。 |
| :material-star: [GPTQ: Accurate Post-Training Quantization for GPTs](https://arxiv.org/abs/2210.17323) — Frantar et al., 2022 | 一次性、基于二阶信息（Hessian）的**仅权重（weight-only）**量化，可压缩到 3–4 bit。 |
| :material-star: [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978) — Lin et al., MLSys 2024 | 根据激活幅值找出约 1% 的显著权重通道，通过逐通道缩放加以保护；得到稳健的 4-bit 权重。已被广泛部署。 |
| :material-star: [SmoothQuant](https://arxiv.org/abs/2211.10438) — Xiao et al., 2022 | 通过数学上等价的重新缩放，把激活中的离群值迁移到权重上 → 实现 **W8A8** 的 INT8 推理。 |
| :material-book-open-variant: [QServe: W4A8KV4 Quantization and System Co-design](https://arxiv.org/abs/2405.04532) — Lin et al., 2024 | 将 4-bit 权重 / 8-bit 激活 / 4-bit KV 与 GPU kernel 协同设计，换来真实的吞吐收益。 |
| :material-book-open-variant: [KIVI: Tuning-Free Asymmetric 2-bit KV Cache Quantization](https://arxiv.org/abs/2402.02750) — Liu et al., 2024 | Key 按通道量化，Value 按 token 量化 → 2-bit KV cache，精度损失很小。 |
| :material-book-open-variant: [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) — Micikevicius et al., 2022 | E4M3/E5M2 格式在 Hopper/Blackwell 上已原生支持；**FP8 推理是 2024–2025 年大模型的默认选择**。 |
| :material-new-box: [Microscaling Data Formats for Deep Learning (MX)](https://arxiv.org/abs/2310.10537) — Rouhani et al., 2023 | 分块缩放的 MXFP8/MXFP6/MXFP4 格式（OCP 标准）。随着 Blackwell 的 FP4 tensor core 到来，**MXFP4/NVFP4** 在 2025 年变得实用——gpt-oss 的 MoE 权重就是以 MXFP4 发布的。 |

经验法则（2025–2026）：FP8（W8A8）基本无损；4-bit 仅权重量化（AWQ/GPTQ/MXFP4）是不错的质量/体积折中；激进的激活或 KV 量化需要在*你自己的任务*上评测——输出很长的推理模型对量化更敏感。

---

## 3.5 KV cache 压缩与长上下文

| Paper | 摘要 |
|---|---|
| :material-star: [Efficient Streaming Language Models with Attention Sinks (StreamingLLM)](https://arxiv.org/abs/2309.17453) — Xiao et al., 2023 | 最开头的几个 token 会吸走大量注意力（"sink"）；保留它们再加一个滑动窗口，就能稳定地生成无限长的文本。启发了 gpt-oss 中可学习的 sink。 |
| :material-book-open-variant: [H2O: Heavy-Hitter Oracle](https://arxiv.org/abs/2306.14048) — Zhang et al., 2023 | 淘汰累计注意力得分很低的 KV 条目。 |
| :material-book-open-variant: [SnapKV](https://arxiv.org/abs/2404.14469) — Li et al., 2024 | 利用 prompt 末尾的观察窗口，为每个 head 选出重要的 KV 位置。 |

架构层面的方案（GQA、MLA、滑动窗口、混合线性注意力、DeepSeek Sparse Attention——见第一部分）通常比事后淘汰更稳健，也是 2025 年前沿模型的选择。

---

## 3.6 PD 分离（Prefill/Decode disaggregation）与以 KV cache 为中心的推理服务 :material-new-box:

预填充是计算受限的，解码是带宽受限的。两者跑在同一批 GPU 上会互相干扰。**PD 分离**把它们放到各自独立的 GPU 池里，并在两者之间传输 KV cache——这是 2025 年大规模部署的主流架构。

| Paper | 摘要 |
|---|---|
| :material-star: [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving](https://arxiv.org/abs/2401.09670) — Zhong et al., OSDI 2024 | 将预填充和解码实例分开，各自采用独立的并行策略，以最大化 **goodput**（同时满足 TTFT 与 TPOT SLO 的请求吞吐）。 |
| :material-book-open-variant: [Splitwise: Efficient Generative LLM Inference Using Phase Splitting](https://arxiv.org/abs/2311.18677) — Patel et al., ISCA 2024 | 微软提出的同一思路；甚至为两个阶段分别使用不同代际的硬件。 |
| :material-star: [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079) — Qin et al. (Moonshot AI), FAST 2025 | Kimi 背后的生产系统：跨 GPU/CPU/SSD 的分布式 KV cache 池，通过 RDMA 传输；调度围绕 KV cache 复用来设计。其 Transfer Engine 已被 SGLang 和 vLLM 采用。 |
| :material-book-open-variant: [Llumnix: Dynamic Scheduling for LLM Serving](https://arxiv.org/abs/2406.03243) — Sun et al., OSDI 2024 | 在实例之间对请求（连同其 KV cache）做在线迁移，实现负载均衡。 |

值得研究的生产级技术栈：**[NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo)**、**[llm-d](https://github.com/llm-d/llm-d)**（Kubernetes 原生，基于 vLLM）、**[LMCache](https://github.com/LMCache/LMCache)**（KV cache 卸载与共享），以及 SGLang 的 PD 分离模式。

---

## 3.7 大规模 MoE 模型的推理部署 :material-new-box:

万亿参数的 MoE 模型（DeepSeek-V3/R1、Kimi K2、Qwen3-235B）改变了游戏规则：每个 token 只会用到少量参数，但所有专家都得有地方放。

- **专家并行（EP）**：把专家分散到大量 GPU 上；token 通过 **all-to-all** 通信发送给对应的专家——参见 DeepSeek 开源的 [DeepEP](https://github.com/deepseek-ai/DeepEP)（all-to-all kernel）和 [EPLB](https://github.com/deepseek-ai/EPLB)（带冗余热点专家的专家负载均衡器）。
- **大规模 EP + PD 分离**：DeepSeek 的[推理系统概览](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)介绍了预填充使用 EP32、解码使用 EP144，并通过双 batch 重叠（dual-batch overlap）来隐藏通信开销。LMSYS 在 96 张 H100 上用 SGLang 复现了该方案（[博客](https://lmsys.org/blog/2025-05-05-large-scale-ep/)）。
- **MLA 专用 kernel**：[FlashMLA](https://github.com/deepseek-ai/FlashMLA)，用于基于 latent KV 的解码。
- **机架级硬件**：NVIDIA GB200/GB300 NVL72 把 72 张 GPU 放进同一个 NVLink 域，使大规模 EP 的成本大幅降低。

---

## 3.8 推理引擎（至少掌握两个）

| 引擎 | 特点 |
|---|---|
| [vLLM](https://github.com/vllm-project/vllm) | 事实上的开源标准；支持 PagedAttention、连续批处理、前缀缓存、分块预填充，模型和硬件覆盖面广；v1 引擎（2025）重写了调度器，降低了 CPU 开销。 |
| [SGLang](https://github.com/sgl-project/sglang) | RadixAttention 前缀缓存、高速结构化输出，对 DeepSeek/MoE 支持很强（大规模 EP、PD 分离）；在 RL rollout 场景中很受欢迎。 |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | NVIDIA 优化的 kernel，在 Hopper/Blackwell 上支持 FP8/FP4，提供 in-flight batching。 |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) / [Ollama](https://ollama.com/) | 面向 CPU、Apple Silicon 和端侧的推理，使用 GGUF 量化。 |
| [MLC-LLM](https://github.com/mlc-ai/mlc-llm) | 基于编译器的部署方案，可跑在手机、浏览器（WebGPU）和各种 GPU 上。 |
| [Hugging Face TGI](https://github.com/huggingface/text-generation-inference) | 与 HF 生态集成的生产级服务器。 |

---

## 3.9 推理模型与 RL 的推理需求 :material-new-box:

推理（reasoning）模型每次回答都要生成成千上万个"思考"token，因此**解码成本占主导**——上面提到的每一项解码优化都变得更加重要。相关趋势：

- **把测试时计算（test-time compute）做成产品旋钮：** 思考预算控制、effort 档位、提前退出。
- **RL rollout 引擎：** RL 训练（第二部分）的大部分时间都花在生成 rollout 上；verl、slime 等框架内嵌 vLLM/SGLang 做生成，并且每一步都要把更新后的权重同步给推理引擎。长尾 rollout 催生了**异步 / 部分 rollout（asynchronous / partial rollouts）**。
- **训练–推理不一致（train–inference mismatch）：** 训练和推理 kernel 之间的数值差异（例如 MoE 路由或低精度场景）可能让 RL 训练失稳；让推理具备 **batch-invariant / 确定性** 成为 2025 年的研究热点（参见 Thinking Machines 的 [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)）。

---

## 3.10 动手实践项目

- [ ] 为你选定的模型，在单卡上计算 KV cache 大小和最大 batch size（使用 3.1 节的公式）。
- [ ] 用 vLLM 和 SGLang 分别部署一个 7–8B 模型；在不同并发度下对 TTFT/TPOT/吞吐做基准测试（`vllm bench serve` 或 SGLang 的 `bench_serving`）。
- [ ] 在一个很长的共享 system prompt 上开启前缀缓存——测量 TTFT 的下降幅度。
- [ ] 对比 BF16、FP8 和 AWQ-INT4 的速度*与*精度（例如 GSM8K）。
- [ ] 启用投机解码（EAGLE-3 或 n-gram），测量接受率和加速比。
- [ ] 用 PyTorch 实现一个最小化的 paged KV cache + 连续批处理循环（可参考约 1,200 行的 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)）。
- [ ] 课程：CMU **[15-442/642 Machine Learning Systems](https://mlsyscourse.org/)**，以及 GPU 编程社区 **[GPU MODE](https://github.com/gpu-mode/lectures)** 的讲座。

## 3.11 自测

??? question "为什么解码是访存受限，而预填充是计算受限？"
    预填充是把权重同时乘以很多个 token（算术强度高）。解码时每个序列一次只有一个 token 与权重相乘，所以除非 batch 很大，每个从 HBM 加载的权重字节只参与极少的 FLOPs。

??? question "为什么投机解码不会改变输出分布？"
    草稿 token 以 min(1, p_target/p_draft) 的概率被接受；一旦被拒绝，就从归一化后的残差分布 max(0, p_target − p_draft) 中重新采样一个 token——这恰好复现了目标模型自身的采样分布。

??? question "PD 分离在什么情况下划算？代价是什么？"
    在大规模部署、TTFT 和 TPOT SLO 都很严格且 prompt 较长的场景下划算；它消除了两个阶段之间的干扰，并允许各阶段采用各自的并行策略。代价是需要传输 KV cache（要求 RDMA/NVLink 之类的高速互联），调度也更复杂。

??? question "为什么前缀缓存对 agent 特别有价值？"
    Agent 循环每一轮都会重发同一份不断增长的上下文（system prompt、工具定义、历史记录）；缓存可以避免重复计算共享前缀，从而大幅降低 TTFT 和成本。
