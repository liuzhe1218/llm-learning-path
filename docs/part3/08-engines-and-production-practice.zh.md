# 3.8 推理引擎与生产实践（Engines & Production Practice）

!!! abstract "本章你将学到"
    - vLLM、SGLang、TensorRT-LLM、llama.cpp、MLC-LLM 和 TGI 分别是什么、是怎么构建的，以及什么时候该选哪一个。
    - 如何**诚实地**对一个推理服务系统做基准测试：负载形态、并发扫描、TTFT/TPOT/ITL 的百分位数、goodput——并附带一个可运行的基准测试客户端。
    - 如何做**容量规划（capacity planning）**：利用 Little 定律和实测的工作点，估算在延迟 SLO 下达到目标 QPS 需要多少 GPU。
    - 如何计算**每百万 token 的成本**，以及哪些因素决定它。
    - 服务**推理模型**和服务 **RL rollout** 时有什么不同（权重同步、异步 rollout、确定性与 batch 不变性）。

## 为什么重要 {#why-this-matters-the-problem}

3.1–3.7 章讲解了各种机制。实践中你很少自己去实现它们：你要选一个引擎、配置它、测量它，然后决定买或租多少 GPU。现实中大多数错误都出在这里——基准测试与线上流量毫不相似、基于平均延迟做的容量规划、假设 100% 利用率的成本估算，或者推理引擎悄悄地与训练器产生分歧的 RL 流水线。本章就是从"理解"通向"运维"的桥梁。

## 直觉 {#intuition}

推理引擎是一个**运行时**（调度器 + 内存管理器 + kernel，见 3.3–3.4 章），外面再包一层 **API 服务器**。引擎之间的差异主要体现在：(1) 面向哪些硬件，(2) 给 CPU 带来多少调度开销，(3) 哪些模型和特性（量化格式、投机方法、并行模式）已经可以直接使用，以及 (4) 修改起来有多容易。

基准测试是一次**实验**：负载是自变量，延迟百分位数和吞吐是因变量。容量规划就是把实验结果去除以你的需求，再加上余量。成本则是 GPU 小时数除以所服务的 token 数。

## 工作原理 {#how-it-works}

### 引擎巡礼 {#a-tour-of-the-engines}

| 引擎 | 它是什么 | 什么时候选它 |
|---|---|---|
| [vLLM](https://github.com/vllm-project/vllm) | 基于 Python/PyTorch 的引擎，首创 PagedAttention；支持连续批处理（continuous batching）、前缀缓存、分块预填充、多种量化和投机方法、TP/PP/EP，以及兼容 OpenAI 的服务器（`vllm serve`）。其 V1 引擎（2025 年起为默认）把 API 前端与一个忙循环的引擎核心分离，以降低 CPU 开销。 | 你想要最广的模型和硬件支持（NVIDIA、AMD、TPU，以及通过插件支持的其他硬件）、庞大的社区，以及从 Hugging Face checkpoint 出发的便捷路径。一个稳妥的默认选择。 |
| [SGLang](https://github.com/sgl-project/sglang) | 围绕 RadixAttention 前缀缓存构建的 Python 引擎，拥有低开销的调度器（让 CPU 调度与 GPU 工作重叠）、快速的语法约束解码，以及对大型 MoE 的强力支持（DeepSeek 式 EP、PD 分离）。 | 前缀密集的负载（agent、多轮对话、few-shot）、结构化输出、大型 MoE 部署、RL rollout（被广泛嵌入各 RL 框架）。 |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | NVIDIA 开源的 LLM 运行时，带手工优化的 kernel、in-flight batching、Hopper/Blackwell 上的 FP8/FP4；常部署在 Triton Inference Server 或 Dynamo 之后。 | 纯 NVIDIA 的集群，并且在最新 GPU 上追求极致性能，同时你能承受更复杂的构建/配置过程。 |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) / [Ollama](https://ollama.com/) | C/C++ 推理，采用 GGUF 格式和多种量化类型；可运行在 CPU、Apple Silicon（Metal）、CUDA、Vulkan 等之上。Ollama 在其外面包了模型下载和本地 API。 | 笔记本、台式机、边缘设备、CPU 服务器、单用户本地使用。并非为高并发的数据中心服务而设计。 |
| [MLC-LLM](https://github.com/mlc-ai/mlc-llm) | 基于编译器的部署方案（Apache TVM）：把模型编译到多种后端，包括手机以及通过 WebGPU 运行的浏览器。 | 端侧和浏览器内推理、特殊的硬件目标。 |
| [Hugging Face TGI](https://github.com/huggingface/text-generation-inference) | 生产级服务器，用 Rust 写的路由器加模型分片，与 HF 生态集成。 | 你已经深度使用 HF 工具链；采用之前，请对照 vLLM/SGLang 检查项目当前的活跃度和特性覆盖。 |

引擎迭代很快——这个季度还是"SGLang 独有"的特性，下个季度可能就进了 vLLM，反之亦然。要用在你自己的模型、硬件和流量上跑出来的基准测试来做决定，而不是看特性清单。

### 真正重要的配置项 {#the-configuration-knobs-that-matter}

每个引擎都把 3.2–3.7 章的机制暴露成设置项。下面用的是 vLLM 的名字（`vllm serve --help`）；SGLang 和 TensorRT-LLM 有名字不同的对应项。

| 设置项（vLLM） | 对应机制 | 调高它的效果 |
|---|---|---|
| `--tensor-parallel-size` | TP 度（3.7） | 总带宽和 KV 空间更多；all-reduce 开销更大 |
| `--max-model-len` | 每个请求的最大上下文（3.2） | 支持更长的请求；能同时容纳的请求变少 |
| `--gpu-memory-utilization` | 引擎可使用的 GPU 显存比例；放完权重之后剩下的部分成为 KV 池（3.3） | KV 块更多 → batch 更大；给其他进程留的余量更少 |
| `--max-num-seqs` | 每个 batch 的最大序列数（3.3） | 在 KV 用完之前吞吐更高；TPOT 更高 |
| `--max-num-batched-tokens` | 每步分块预填充的 token 预算（3.3） | 长 prompt 的 TTFT 更低；正在运行的解码的 ITL 更高 |
| `--enable-prefix-caching` | 基于哈希的前缀缓存（3.3） | prompt 共享前缀时，TTFT 和预填充开销更低 |
| `--kv-cache-dtype fp8` | KV 量化（3.2, 3.6） | KV 容量约 2 倍；有小幅精度风险 |
| `--quantization` | 权重/激活格式，例如 FP8 或 AWQ（3.6） | 显存和带宽占用更少；精度必须检查 |
| `--speculative-config` | 草稿方法和投机 token 数（3.5） | 低负载下延迟更低；高负载下可能有害 |

### 正确地做基准测试 {#benchmarking-correctly}

**1. 定义负载形态。** 要测量或假设的是分布，而不是单个值：

- 输入长度和输出长度的分布（例如聊天：约 1–2K 输入 / 约 300 输出；RAG：约 8K 输入 / 约 200 输出；推理：约 1K 输入 / 8K+ 输出）；
- **前缀结构**——每个 prompt 中有多少是共享的（system prompt、文档、历史），这决定了前缀缓存的命中率；
- **到达过程**——开环（open-loop，请求按某个速率到达，例如泊松过程）还是闭环（closed-loop，固定数量的并发用户，上一个请求结束就发新请求）。

**2. 扫描负载。** 在逐步增大的并发（闭环）或请求速率（开环）下运行：例如 1、4、16、64、128、256。在每个点上记录

- **TTFT** 的 P50/P90/P99、**TPOT** 的 P50/P99、**ITL** 的 P99（卡顿）、**E2E** 的 P99；
- 每秒输出 token 数和每秒请求数；
- **goodput**：满足所有 SLO 的每秒请求数。

画出延迟百分位数相对于吞吐的曲线。结果是一条曲线：一开始延迟是平的，随着系统饱和而急剧上扬。可用的工作点，是 P99 仍满足 SLO 的最高吞吐。

**3. 控制变量。** 测量前先预热服务器（CUDA graph 捕获、各种缓存）；在客户端测量（用户体验到的，包括网络和 detokenization）；比较不同引擎时固定输出长度（例如用 `ignore_eos` 选项，并明确说明）或使用真实输出；使用相同的 tokenizer 和相同的采样设置；重复运行。引擎都自带这类工具——`vllm bench serve` 和 SGLang 的 `python -m sglang.bench_serving`，可在合成或数据集 prompt 上扫描请求速率或并发（当前支持的参数请查看 `--help`）。

### 容量规划（Capacity planning） {#capacity-planning}

**Little 定律**描述了任何稳定系统中负载与延迟的关系：系统中的平均请求数 $N$，等于到达率 $\lambda$ 乘以每个请求在系统中停留的平均时间 $W$，

$$
N = \lambda \cdot W .
$$

用大白话说：如果每秒到达 20 个请求，每个请求端到端耗时 14 s，那么任一时刻大约有 280 个请求在飞行中。这告诉你集群必须支撑的并发数。流程如下：

1. 从基准测试中找出单个副本的**工作点**：在 P99 TTFT 和 TPOT 满足 SLO 的前提下，*基于你的流量形态*（其中包含预填充负载）所能达到的最大并发 $C_{\text{rep}}$（或吞吐 $X_{\text{rep}}$，单位 output token/s）。
2. 计算需求：output token/s $= \lambda \cdot \bar N_{\text{out}}$，飞行中的请求数 $= \lambda \cdot W$，其中 $W \approx \text{TTFT} + \bar N_{\text{out}} \cdot \text{TPOT}$。
3. 所需副本数 $= \max\big(\text{demand tokens/s} / X_{\text{rep}},\ \text{in-flight} / C_{\text{rep}}\big)$，向上取整，再加上应对峰值、故障和发版的**余量**（通常 20–50%）。
4. 对飞行中的请求，按其长度，核对每个副本的 KV 显存是否够用（见 3.2 章）。

### 每百万 token 的成本 {#cost-per-million-tokens}

$$
\text{cost per 1M tokens} = \frac{\text{USD per GPU-hour} \times \text{GPUs}}{\text{tokens served per hour} / 10^6} .
$$

用大白话说：你每小时付出的钱，除以你每小时实际服务了多少百万 token。有三件事主导它：**在你的 SLO 下**的单 GPU 吞吐（延迟要求越紧 → batch 越小 → 成本越高）、集群在一天中的**利用率**（闲置的 GPU 照样花钱），以及输入 token 与输出 token 的计数方式（预填充 token 比解码 token 便宜得多，这也是 API 里输入 token 价格更低、命中缓存的输入 token 更低的原因）。

### 服务推理模型 {#serving-reasoning-models}

推理模型每次回答会输出几千到几万个"思考" token。后果是：

- **解码占主导。** 在 30 ms TPOT 下，一条 10,000 token 的推理链要花 5 分钟。决定用户感知延迟的是 TPOT 而不是 TTFT，每一项解码优化（FP8/FP4 权重、投机解码、MLA/GQA）都更重要。
- **KV 在生成过程中增长。** 长输出会在几分钟里持续占用 KV 显存；并发必须按*最终*长度来规划，如果放进来的请求过多，抢占（preemption，见 3.3 章）就会变得频繁。
- **长尾长度**让静态的容量估算不可靠；要用输出长度的分布来规划，而不是均值。
- **产品层面的开关：** 最大思考预算、effort 档位和提前退出，都是在质量与成本之间做权衡；引擎提供解析器，把推理内容与最终答案在 API 响应中分开。

### 为 RL rollout 提供服务 {#serving-for-rl-rollouts}

在 RL 后训练（第 2 部分）中，大部分墙钟时间花在**生成 rollout** 上。[verl](https://github.com/volcengine/verl) 和 [slime](https://github.com/THUDM/slime) 这样的框架把 vLLM 或 SGLang 嵌入为 rollout 引擎。这种用法有三个特有的问题：

- **权重同步。** 每次优化器步之后，推理引擎都必须收到新权重：通过 NCCL 从训练器广播到推理 GPU；或者当训练和推理共用同一批 GPU（"colocated"）时，通过在两者之间交换显存来实现（引擎提供释放并重新获取 GPU 显存的模式）。同步一个 70B 的 BF16 模型要搬运约 140 GB，所以同步耗时很关键。
- **长尾与异步。** 在同步的一步里，最慢的那条 rollout（最长的推理链）会拖住整个 batch。**异步**流水线让下一个 batch 的生成与当前 batch 的训练重叠，**partial rollout** 则把特别长的生成暂停，稍后再恢复。两者都会让数据略微变成**离策略（off-policy）**的（由更旧的权重生成），RL 算法必须容忍或修正这一点。
- **训练—推理不一致与确定性。** rollout 引擎计算 token 概率时用的 kernel 与训练器不同（而且精度往往更低），所以"同一个"策略给出的 log-probability 会略有差异，这可能让 RL 训练不稳定。Thinking Machines 的 *Defeating Nondeterminism in LLM Inference* 分析了一个根本原因：缺乏 **batch 不变性（batch invariance）**——浮点加法不满足结合律，而 kernel 会根据 batch 大小和序列长度选择不同的归约策略（split-K 矩阵乘、split-KV 注意力、不同的 tile 大小）。因此一个请求的数值结果取决于 batch 里还有什么别的请求。batch 不变的 kernel 无论 batch 如何组成都固定归约顺序，能得到可复现的输出，代价是一些性能。

## 代码 {#code}

一个最小的异步基准测试客户端，适用于任何兼容 OpenAI 的服务器（vLLM、SGLang、通过其 OpenAI 服务器接入的 TensorRT-LLM、TGI）。它运行一个闭环的并发档位，并报告输出吞吐、TTFT/TPOT/ITL 百分位数和 goodput。在多个并发档位上各跑一次，就能描绘出延迟—吞吐曲线。

```python
import asyncio, time, statistics
from openai import AsyncOpenAI     # pip install openai；适用于任何兼容 OpenAI 的服务器

async def one_request(client, model, prompt, max_tokens):
    t0 = time.perf_counter()
    first, last, gaps, n_out = None, None, [], 0
    stream = await client.completions.create(
        model=model, prompt=prompt, max_tokens=max_tokens, stream=True,
        stream_options={"include_usage": True})
    async for chunk in stream:
        now = time.perf_counter()
        if chunk.choices and chunk.choices[0].text:
            if first is None:
                first = now                                  # 第一个 token 到达
            else:
                gaps.append(now - last)                      # token（chunk）间延迟
            last = now
        if chunk.usage:                                      # 最后一个 chunk 带有计数
            n_out = chunk.usage.completion_tokens
    e2e = time.perf_counter() - t0
    ttft = first - t0
    tpot = (e2e - ttft) / max(n_out - 1, 1)
    return dict(ttft=ttft, tpot=tpot, e2e=e2e, n_out=n_out, gaps=gaps)

def pct(xs, p):
    return statistics.quantiles(xs, n=100, method="inclusive")[p - 1]

async def run(base_url, model, prompts, concurrency, max_tokens=300,
              slo_ttft=2.0, slo_tpot=0.05):
    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")
    sem = asyncio.Semaphore(concurrency)                     # 闭环：始终有 N 个请求在飞行中
    async def guarded(p):
        async with sem:
            return await one_request(client, model, p, max_tokens)
    t0 = time.perf_counter()
    res = await asyncio.gather(*(guarded(p) for p in prompts))
    wall = time.perf_counter() - t0
    ttft, tpot = [r["ttft"] for r in res], [r["tpot"] for r in res]
    itl = [g for r in res for g in r["gaps"]]
    good = sum(r["ttft"] <= slo_ttft and r["tpot"] <= slo_tpot for r in res)
    print(f"concurrency={concurrency:4d}  out tok/s={sum(r['n_out'] for r in res) / wall:8.0f}  "
          f"TTFT p50/p99={pct(ttft, 50):.2f}/{pct(ttft, 99):.2f}s  "
          f"TPOT p50/p99={1000 * pct(tpot, 50):.0f}/{1000 * pct(tpot, 99):.0f}ms  "
          f"ITL p99={1000 * pct(itl, 99):.0f}ms  goodput={good / wall:.2f} req/s")

# 针对本地服务器的扫描示例，服务器例如用 `vllm serve <model>` 启动：
# prompts = [open("my_real_prompts.txt").read()] * 200      # 用你自己的流量形态
# for c in (1, 4, 16, 64, 128):
#     asyncio.run(run("http://localhost:8000/v1", "<model>", prompts, c))
```

为什么 batch 不变性是个真问题，四行代码就能说明：同样的数以不同顺序求和，会得到不同的 float32 结果，而 kernel 会随 batch 大小改变求和顺序。

```python
import numpy as np
x = np.random.default_rng(0).standard_normal(4096).astype(np.float32) * 1e3
one_pass = x.sum(dtype=np.float32)                                   # 一种归约顺序
split_4 = sum(c.sum(dtype=np.float32) for c in np.split(x, 4))       # 分成 4 块的 “split-K”
print(one_pass, split_4, "equal:", one_pass == split_4)
```

## 实例演算 {#worked-example}

!!! note "示意性数字"
    下面的基准测试结果和 GPU 价格都是**假设的输入**，选得比较合理；重点是方法。请换成你自己的实测值和价格。

**目标。** 为一个聊天产品服务 Llama-3-8B（BF16）。需求：峰值 $\lambda = 20$ 请求/秒；平均 prompt 1,000 token（无共享前缀），平均输出 300 token。SLO：P99 TTFT ≤ 1 s，P99 TPOT ≤ 50 ms。

**第 1 步——对单个副本（1× H100）做基准测试**，使用这种流量形态。假设扫描显示：并发为 128 时，副本能提供 **2,800 output token/s**，P99 TPOT 为 45 ms，P99 TTFT 为 0.8 s；并发为 192 时，P99 TPOT 超过 50 ms。工作点：$C_{\text{rep}} = 128$，$X_{\text{rep}} = 2{,}800$ token/s。（一致性检查：128 条序列 ÷ 每 token 0.045 s ≈ 2,850 token/s。）

**第 2 步——需求。**

- 每秒输出 token：$20 \times 300 = 6{,}000$。
- 在系统中的停留时间：$W \approx 0.5\ \text{s} + 299 \times 0.045\ \text{s} \approx 14\ \text{s}$；飞行中的请求数：$N = 20 \times 14 = 280$。

**第 3 步——副本数。** 按吞吐算，$6{,}000 / 2{,}800 = 2.14$；按并发算，$280 / 128 = 2.19$。向上取整为 3，再加约 30% 的余量（$2.19 \times 1.3 = 2.85$），3 个副本依然足够。峰值时需要 **3 张 H100**。

**第 4 步——KV 核对。** 280 个飞行中请求 / 3 个副本 ≈ 每张 GPU 93 条序列，每条最长 1,300 token，每 token 128 KiB（见 3.2 章）：$93 \times 1300 \times 128\ \text{KiB} \approx 14.8$ GiB——远低于约 50 GiB 的 KV 预算。预填充负载：$20 \times 1000 = 20{,}000$ prompt token/s，每个副本约 6,700——占了每张 GPU 相当大一部分时间，这就是基准测试必须使用真实 prompt 长度的原因。

**第 5 步——成本。** 假设价格为每 GPU 小时 USD 2.50。峰值时，3 张 GPU 每小时花费 USD 7.50，每小时服务 $6{,}000 \times 3600 = 21.6$ M 输出 token（外加每小时 72 M 输入 token）：

$$
\frac{7.50\ \text{USD/hour}}{21.6\ \text{M tokens/hour}} \approx 0.35\ \text{USD per million output tokens (at peak utilization)}.
$$

如果全天的平均流量是峰值的 40%，但你一直开着 3 张 GPU，那么实际成本为 $0.35 / 0.4 \approx 0.87$ USD 每百万输出 token——利用率和 kernel 速度同样重要。自动扩缩容（见 3.7 章）、FP8 权重（字节更少 → 同样 TPOT 下 batch 更大）和前缀缓存（预填充更少）是压低成本的主要手段；把 TPOT SLO 放宽到 80 ms 也能让 batch 更大，降低每 token 成本。

## 实践中的做法（2025–2026） {#in-practice-20252026}

- 大多数开源模型的部署使用 **vLLM 或 SGLang**；以 NVIDIA 为中心、想在最新 GPU 上追求极致性能的团队使用 **TensorRT-LLM**（常通过 Dynamo）；本地和边缘使用则以 **llama.cpp/Ollama** 和 Apple Silicon 专用的运行时为主。
- 基准测试工具已内置在引擎里（`vllm bench serve`、`sglang.bench_serving`），也有独立的基准测试项目发布跨引擎的对比——在相信它们的排名之前，先读一读它们的负载定义。
- 推理服务栈会暴露 Prometheus 指标（队列长度、KV 利用率、前缀缓存命中率、投机解码接受率、TTFT/TPOT 直方图）；这些就是自动扩缩容和 SLO 告警的输入。
- 推理模型让 **TPOT** 和**长输出下的 KV 容量**成为核心的服务指标，并让投机解码和 FP8/FP4 成为标配。
- RL 框架（verl、slime 等）把 rollout 引擎当作一等公民的组件；异步与 partial rollout、快速权重同步以及 batch 不变 / 确定性推理，是贯穿 2025 年的活跃工程课题。

!!! tip "上线前检查清单"
    - 启动日志中的 KV 容量（token 数或块数）能支撑你规划的并发和真实长度。
    - 在**你自己的**流量形态上做的并发扫描显示，峰值需求加余量下 SLO 依然满足。
    - 已经在你的任务上，对照 BF16 基线评估了所服务配置（量化、KV dtype、投机解码）的质量。
    - 已为 TTFT/TPOT 百分位数、队列长度、KV 利用率、抢占次数和前缀缓存命中率建立了仪表盘和告警。
    - 你清楚新副本的冷启动时间，并且至少留有足以覆盖这段时间的热备容量。

## 常见误区 {#common-pitfalls-misconceptions}

- **只用一种 prompt 长度和一种输出长度做基准测试。** 真实流量是有分布和长尾的；P99 由长尾决定。
- **只报告最大吞吐而不报告延迟。** 它描述的是一个没人会部署的系统。要报告满足 SLO 的吞吐，或者 goodput。
- **只看服务端延迟。** 用户体验到的是客户端 TTFT，包含排队、网络和 detokenization。
- **把闭环结果当成开环结果。** 闭环基准测试在服务器变慢时会自己放慢到达速率，从而掩盖过载；线上流量更接近开环。
- **按 100% 利用率估算成本。** 要除以真实的平均利用率。
- **假设不同 batch 大小或不同引擎下的输出相同。** 没有 batch 不变的 kernel 时，同一个 prompt 可能因并发负载不同而产出不同的 token——这对评测、调试和 RL 都很重要。

## 本章要点 {#key-takeaways}

- 引擎要靠在你自己的模型、硬件和流量上做基准测试来选；vLLM 和 SGLang 覆盖了大多数数据中心的需求，TensorRT-LLM 面向 NVIDIA 上的极致性能，llama.cpp/MLC 面向本地与边缘。
- 用真实的负载形态做基准测试，扫描负载，并报告 TTFT/TPOT/ITL 百分位数、吞吐和 goodput。
- 容量 = 需求（token/s，以及通过 Little 定律得到的飞行中请求数）÷ SLO 下单副本的工作点，再加余量，并用 KV 显存核对。
- 每百万 token 成本 = （每小时 USD）÷（每小时百万 token 数）；SLO 的严格程度和集群利用率与原始速度同样重要。
- 推理模型让解码和 KV 容量成为主导；RL rollout 则带来权重同步、异步以及数值一致性（batch 不变性）方面的要求。

## 自测 {#check-your-understanding}

??? question "你的基准测试显示，并发 512 时有 10,000 output token/s，P99 TPOT 为 140 ms。你的 SLO 是 P99 TPOT ≤ 50 ms。做容量规划时该用哪个数字？"
    都不能直接用：在扫描结果中找到 P99 TPOT ≤ 50 ms 的最高并发，然后用在那一点测得的吞吐。10,000 token/s 那个点违反了 SLO。

??? question "λ = 5 请求/秒，平均端到端延迟 40 s。平均有多少请求在飞行中？"
    由 Little 定律，$N = 5 \times 40 = 200$。

??? question "一张 GPU 每小时花费 USD 3，在你的 SLO 下能持续输出 2,000 token/s。满负荷利用时，每百万输出 token 的成本是多少？50% 利用率时呢？"
    $2000 \times 3600 = 7.2$ M token/小时 → $3 / 7.2 \approx 0.42$ USD 每百万；50% 利用率时约 0.83 USD。

??? question "为什么一个 64 用户的闭环基准测试，相比同样平均速率下的开环流量，会低估过载？"
    在闭环中，每个用户要等到收到响应才发下一个请求，所以服务器变慢时，到达率会自动下降。开环的到达则不管怎样都会持续到来，所以一旦超过容量，队列（以及 TTFT）就会无限增长。

??? question "为什么同一个 prompt 用贪心采样，在服务器繁忙与空闲时可能给出不同的输出？"
    batch 的组成变了，kernel 可能选择不同的归约策略（split-K、split-KV、tile 大小）；浮点数不满足结合律，使 logits 略有变化，argmax 中接近平局的情形就可能翻转，之后两次生成就会分叉。batch 不变的 kernel 可以避免这一点。

## 延伸阅读 {#further-reading}

| 论文 / 资源 | 为什么读 |
|---|---|
| :material-star: [vLLM](https://github.com/vllm-project/vllm) | 事实上的开源标准引擎：PagedAttention、连续批处理、前缀缓存、分块预填充；V1 引擎。 |
| :material-star: [SGLang](https://github.com/sgl-project/sglang) | RadixAttention、快速结构化输出、大规模 EP 与 PD 分离；在 RL rollout 中很流行。 |
| :material-book-open-variant: [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | NVIDIA 优化的 kernel、Hopper/Blackwell 上的 FP8/FP4、in-flight batching。 |
| :material-book-open-variant: [llama.cpp](https://github.com/ggml-org/llama.cpp) / [Ollama](https://ollama.com/) | 面向 CPU/Apple Silicon/边缘设备的推理，带 GGUF 量化。 |
| :material-book-open-variant: [MLC-LLM](https://github.com/mlc-ai/mlc-llm) | 基于编译器的部署，目标包括手机、浏览器（WebGPU）和各类 GPU。 |
| :material-book-open-variant: [Hugging Face TGI](https://github.com/huggingface/text-generation-inference) | 与 HF 生态集成的生产级服务器。 |
| :material-star: [Defeating Nondeterminism in LLM Inference (Thinking Machines)](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) | 为什么推理不具备 batch 不变性，以及如何让它具备；对 RL 和可复现评测至关重要。 |
| :material-book-open-variant: [verl](https://github.com/volcengine/verl) | 把 vLLM/SGLang 嵌入为 rollout 引擎的 RL 后训练框架。 |
| :material-book-open-variant: [slime](https://github.com/THUDM/slime) | 围绕 SGLang rollout 和快速权重同步构建的 RL 框架。 |
| :material-book-open-variant: [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) | 约 1,200 行、可读性强的引擎：用真实代码看懂第 3 部分全部内容的最佳方式。 |
| :material-book-open-variant: [LLM Inference Handbook (BentoML)](https://github.com/bentoml/llm-inference-handbook) | 关于指标、基准测试和部署选择的实用指南。 |
| :material-book-open-variant: [CMU 15-442/642 Machine Learning Systems](https://mlsyscourse.org/) | 涵盖 ML 系统方向（包括推理服务）的大学课程。 |
| :material-book-open-variant: [GPU MODE lectures](https://github.com/gpu-mode/lectures) | 关于 GPU 编程和推理 kernel 的社区讲座。 |
