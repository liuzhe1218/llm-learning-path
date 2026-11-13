# 3.3 批处理与内存管理

!!! abstract "本章你将学到"
    - 为什么**静态批处理（static batching）**会浪费 GPU 时间，以及**连续（迭代级）批处理（continuous batching）**如何解决。
    - 为什么连续的 KV 分配会产生碎片，以及 **PagedAttention** 如何像操作系统的虚拟内存一样使用块表。
    - 实现一个带引用计数和**写时复制（copy-on-write）**的小型分页块分配器。
    - 解释**分块预填充（chunked prefill）**及其所控制的 TTFT/TPOT 权衡。
    - 解释**前缀缓存（prefix caching）**和 SGLang 的 **RadixAttention**，以及调度与抢占（FCFS、换出、重算）。

## 为什么重要 {#why-this-matters-the-problem}

3.1 章说明了解码吞吐几乎随 batch 中的序列数线性增长，3.2 章说明了 KV cache 决定内存里能放下多少个序列。因此推理引擎有两项工作，合起来构成它的“操作系统”：

1. **调度** —— 在满足延迟 SLO 的前提下，让每次前向计算里尽可能多地装满有用的序列。
2. **内存管理** —— 以尽可能少的浪费把 KV cache 装进 GPU 显存，并在请求共享内容时复用它们。

在 2022–2023 年之前，推理系统这两件事都做得很差：它们按整个请求做批处理，并按最大可能长度预留一块连续的 KV 区域。本章的技术——连续批处理（Orca）、PagedAttention（vLLM）、分块预填充（Sarathi-Serve）和前缀缓存（SGLang 中的 RadixAttention）——把 LLM 推理变成了高吞吐的生意，如今每个严肃的引擎都实现了全部这些技术。

## 直觉 {#intuition}

**静态批处理是一辆要等所有乘客都到达各自目的地才发车的巴士。** 四个请求一起开始；一个需要 20 个 token，另一个需要 500 个。短请求早早结束，但它们的座位一直空着，直到最长的那个结束，而新请求只能在队列里等着。

**连续批处理是一列每站都有人上下车的火车。** 每次前向计算（每一次解码“迭代”）就是一个车站：完成的序列下车，等待的序列上车。

**PagedAttention 就是 KV cache 的虚拟内存。** 它不再为每个请求预留一整块连续数组（还要猜它会长到多长），而是把 cache 切成小的固定大小的**块**（页）。每个请求有一张**块表**，把逻辑块号映射到恰好空闲的任意物理块——和操作系统把虚拟页映射到物理页帧的方式一模一样。内存随序列增长一次分配一个块，相同的块还可以在序列之间共享。

## 工作原理 {#how-it-works}

### 静态批处理及其浪费 {#static-batching-and-its-waste}

静态（请求级）批处理中，一个包含 $B$ 个请求的 batch 要一直运行到**全部**结束。如果输出长度是 $N_1,\dots,N_B$，这个 batch 需要 $\max_i N_i$ 步，却只做了 $\sum_i N_i$ 个有用的 token 步。利用率为

$$
U_{\text{static}} = \frac{\sum_i N_i}{B \cdot \max_i N_i}.
$$

换句话说：batch 槽位中真正在干有用活的比例。长度为 20、50、100、500 时，$U = 670 / 2000 = 33.5\%$。真实的输出长度有很长的尾巴，所以静态批处理通常会浪费 batch 的大部分。

### 连续（迭代级）批处理 {#continuous-iteration-level-batching}

Orca 的思路：以一次模型迭代为粒度来调度。每次前向计算之后，调度器 (1) 移除已输出 EOS 或达到长度上限的序列，(2) 在内存允许时准入等待中的请求，(3) 用所有正在运行的序列构建下一个 batch。

```text
时间（迭代） ─▶   1   2   3   4   5   6   7   8   9  10
静态批处理
  槽位 0  请求 A    ■   ■   ■   ·   ·   ·   ·   ·   ·   ·     A 在第 3 步结束，槽位空闲
  槽位 1  请求 B    ■   ■   ■   ■   ■   ■   ■   ■   ·   ·
  槽位 2  请求 C    ■   ■   ■   ■   ■   ·   ·   ·   ·   ·
  （请求 D、E 在队列里等到第 9 次迭代）
连续批处理
  槽位 0            A   A   A   D   D   D   D   E   E   E     D 在第 4 步加入，E 在第 8 步
  槽位 1            B   B   B   B   B   B   B   B   F   F
  槽位 2            C   C   C   C   C   G   G   G   G   G
```

有一个麻烦：batch 里的序列处在不同的位置，上下文长度也不同，所以 attention 不能是一个稠密的 `[batch, seq]` 张量运算。引擎把这一步所有的 token 展平成一个“不规则（ragged）”batch（不做 padding）来做矩阵乘法——矩阵乘法并不关心某个 token 属于哪个序列——并使用接收每个序列长度和 KV 位置的 attention kernel（3.4 章）。

### 内存碎片 {#memory-fragmentation}

如果每个请求在开始时就被分配一块按其最大长度（比如 4,096 token）预留的连续 KV 区域：

- **内部碎片：** 一个 300 token 就结束的请求，在整个生命周期里浪费了其余 3,796 个槽位。
- **预留浪费：** 即使对最终会变长的请求，槽位也是在真正用到之前很早就预留了。
- **外部碎片：** 空闲内存被切成许多太小的间隙，放不下新的最大长度预留。

vLLM 论文测量发现，早期系统中只有少部分 KV 内存真正存放了 token 状态。由于 KV 容量决定批大小，浪费的 KV 内存就直接是浪费的吞吐。

### PagedAttention：块表 {#pagedattention-block-tables}

把 KV 池切成物理块，每个块为每一层存放 $S_{\text{blk}}$ 个连续 token 的 K 和 V（vLLM 默认块大小是 16 个 token）。对每个序列维护一张**块表**：第 $j$ 项是存放 token $[jS_{\text{blk}}, (j+1)S_{\text{blk}})$ 的物理块。位置 $p$ 的 token 存放在

$$
\text{physical block} = \text{table}\big[\lfloor p / S_{\text{blk}} \rfloor\big], \qquad \text{offset} = p \bmod S_{\text{blk}}.
$$

只有当最后一个块写满时才会分配新块，所以每个序列最多浪费 $S_{\text{blk}} - 1$ 个槽位。attention kernel 读取块表，从分散的物理块中收集 K/V；3.4 章讨论这对 kernel 设计的影响。

**例子。** 块大小为 4。序列 A 有 10 个 token，序列 B 有 6 个。

```text
逻辑视图                    块表                      物理 KV 池（块 id: 内容）
A: t0..t9                   A: [7, 2, 5]              0: 空闲     4: B t4-t5（已用 2/4）
B: t0..t5                   B: [3, 4]                 1: 空闲     5: A t8-t9（已用 2/4）
                                                      2: A t4-t7  6: 空闲
                                                      3: B t0-t3  7: A t0-t3
```

token A·t9 位于块 `A_table[9 // 4] = A_table[2] = 5`，偏移 `9 % 4 = 1`。浪费的是块 5 中的 2 个槽位和块 4 中的 2 个槽位——而不是几千个。

### 共享与写时复制 {#sharing-and-copy-on-write}

块表让**共享**变得很便宜。在并行采样（对一个 prompt 生成 $n$ 个补全）或束搜索中，所有补全共享 prompt 的 KV 块。每个物理块带有一个**引用计数**。当一个序列要写入引用计数大于 1 的块（例如只写了一部分的最后一个 prompt 块）时，引擎先**复制**这个块，把该序列的块表指向副本，并把原块的计数减一——这就是写时复制，和操作系统中的 `fork()` 完全一样。prompt 的 KV 只存一份，而不是 $n$ 份。

### 分块预填充 {#chunked-prefill}

连续批处理遗留了一个问题：当一个长 prompt 到达时，它的预填充（几千个 token，几十到几百毫秒）会在某次迭代中运行，该迭代里所有正在解码的序列都要等待。用户会看到卡顿：ITL 出现尖峰。

Sarathi-Serve 的解决办法是给每次迭代一个 **token 预算**（例如 512 或 2,048 个 token）。调度器先为每个正在运行的解码加一个 token，然后用剩余预算装入某个待处理预填充的一个**分块（chunk）**。一个 10,000 token 的 prompt，每块约 1,500 token，会在约 7 次迭代中完成预填充，搭着正在进行的解码一起跑。预填充分块让迭代变成计算受限（充分利用 GPU），而解码最多只会被阻塞一次有界的迭代。

这个**权衡**由预算控制：

- **预算更小** → 每次迭代更短 → TPOT/ITL 低且稳定，但长 prompt 需要更多次迭代 → TTFT 更高。
- **预算更大** → 预填充更快 → TTFT 更低，但迭代更长 → 其他所有人的 TPOT 更高。

分块需要这样的 attention kernel：一个分块的 query 要关注已经包含了前面分块的 KV cache——和下面前缀缓存用的是同一种 kernel。

### 前缀缓存与 RadixAttention {#prefix-caching-and-radixattention}

许多请求共享一个**前缀**：相同的 system prompt、few-shot 示例、工具定义、文档，或者——在多轮对话和 agent 中——整个之前的对话。为每个请求重新计算这个前缀的 KV 纯属浪费。**前缀缓存（prefix caching）**在请求结束后保留 KV 块，并在后来的请求以相同 token 开头时复用它们。

两种常见实现：

- **基于哈希的块（vLLM）。** 每个*完整*的块由它的 token 加上之前所有块的哈希共同标识。新请求逐块对 prompt 做哈希，复用表里能找到的每个开头的块。需要内存时，空闲的缓存块按 LRU 驱逐。
- **Radix tree（SGLang 的 RadixAttention）。** 所有缓存的 token 序列存放在一棵 **radix tree** 中（一种 trie，边上标注 token 序列，在序列分叉处分裂边）。每个节点指向其边上 token 的 KV。新请求沿树行走，找到**最长匹配前缀**，复用那部分 KV，只预填充剩下的部分；新的 token 作为一个分支插入。没有被运行中请求使用的叶子按 LRU 驱逐。SGLang 的调度器还是**缓存感知**的：它优先选择缓存前缀更长的请求，以提高命中率。

**例子。** 一个 system prompt `S`（1,000 token）被三个请求共享。

```text
请求 1: S + "Translate: hello"            ──▶ 未命中：预填充 1,000 + 4 个 token
请求 2: S + "Translate: goodbye"          ──▶ 命中 S + "Translate:" ；预填充 1-2 个 token
请求 3: S + "Summarize: ..." (200 tok)    ──▶ 命中 S ；预填充约 200 个 token

三个请求之后的 radix tree
  root ── S(1000) ──┬── "Translate:" ──┬── "hello"
                    │                  └── "goodbye"
                    └── "Summarize: ..." (200)
```

请求 2 和 3 各跳过了约 1,000 token 的预填充：TTFT 大幅下降，GPU 时间也被腾出来做别的事。在 agent 循环中，第 $k$ 轮会原样重发第 $1..k-1$ 轮，命中率可以接近整个前缀的 100%。

### 调度策略与抢占 {#scheduling-policies-and-preemption}

一个典型的调度器循环，每次迭代都运行：

1. **先保证正在运行的解码** —— 兑现对已准入请求的延迟承诺。
2. 在 KV 块可用时，按 **FCFS**（先来先服务）顺序**准入等待中的请求**，利用前缀缓存命中降低它们的成本，并用分块预填充限制迭代长度。
3. **如果内存耗尽**而运行中的序列还在增长，就**抢占**一个请求（通常是最近准入的，以保持 FCFS 的公平性）并释放它的块。之后有两种方式把它恢复回来：
    - **换出（Swap）**：把它的 KV 块复制到 CPU 内存，恢复时再换回来。代价：PCIe 传输。
    - **重算（Recompute）**：丢弃 KV，恢复时对 prompt + 已生成的 token 重做预填充。代价：GPU 算力——往往比听起来便宜，因为预填充又快又并行。

引擎还提供优先级调度、按租户公平性，以及诸如 `max_num_seqs`（每个 batch 的序列数）和 `max_num_batched_tokens`（分块预填充的预算）这样的限制。频繁的抢占说明引擎准入过量；表现为延迟尖峰和浪费的计算。

## 代码 {#code}

下面是一个带引用计数和写时复制的分页 KV 块分配器的小模拟，之后是静态批处理与连续批处理利用率的对比。纯 Python——逻辑与真实引擎相同，只是没有 GPU 张量。

```python
import random

class BlockAllocator:
    def __init__(self, num_blocks, block_size):
        self.bs = block_size
        self.free = list(range(num_blocks))
        self.ref = [0] * num_blocks
        self.tables, self.lens = {}, {}            # seq_id -> [block ids], token count

    def _alloc(self):
        if not self.free:
            raise MemoryError("out of KV blocks -> scheduler must preempt")
        b = self.free.pop()
        self.ref[b] = 1
        return b

    def add_seq(self, sid, n_tokens):              # 预填充：分配 ceil(n/bs) 个块
        self.tables[sid] = [self._alloc() for _ in range(-(-n_tokens // self.bs))]
        self.lens[sid] = n_tokens

    def append_token(self, sid):                   # 解码：可能需要新块，可能需要 CoW
        table, n = self.tables[sid], self.lens[sid]
        if n % self.bs == 0:                       # 最后一个块已满 -> 新块
            table.append(self._alloc())
        elif self.ref[table[-1]] > 1:              # 要写入共享块 -> 复制
            self.ref[table[-1]] -= 1
            table[-1] = self._alloc()              # （真实引擎：同时复制 KV 数据）
        self.lens[sid] = n + 1

    def fork(self, parent, child):                 # 并行采样：共享所有块
        self.tables[child] = list(self.tables[parent])
        self.lens[child] = self.lens[parent]
        for b in self.tables[child]:
            self.ref[b] += 1

    def free_seq(self, sid):
        for b in self.tables.pop(sid):
            self.ref[b] -= 1
            if self.ref[b] == 0:
                self.free.append(b)
        del self.lens[sid]

alloc = BlockAllocator(num_blocks=8, block_size=4)
alloc.add_seq("p", 6)                  # 6 个 token 的 prompt -> 2 个块（最后一个半满）
alloc.fork("p", "s1"); alloc.fork("p", "s2")
for s in ("s1", "s2"):
    alloc.append_token(s)              # 每个样本写入 token 7 -> 对块 2 做 CoW
print({s: alloc.tables[s] for s in ("p", "s1", "s2")}, "free:", len(alloc.free))
alloc.free_seq("p")
print("ref counts:", alloc.ref, "free:", len(alloc.free))

# 静态 vs 连续批处理：有用槽位 / 总槽位，同样的请求流。
random.seed(0)
lens = [min(int(random.expovariate(1 / 200)) + 1, 2000) for _ in range(256)]
B = 16
static_steps = sum(max(lens[i:i + B]) for i in range(0, len(lens), B))
queue, running, cont_steps = list(lens), [], 0
while queue or running:
    while queue and len(running) < B:   # 每次迭代都准入新请求
        running.append(queue.pop(0))
    running = [r - 1 for r in running if r > 1]
    cont_steps += 1
print(f"static: {static_steps} steps, util {sum(lens) / (B * static_steps):.0%}; "
      f"continuous: {cont_steps} steps, util {sum(lens) / (B * cont_steps):.0%}")
```

预期输出：两个分叉出来的样本各自拿到共享的最后一个 prompt 块的私有副本，而第一个块仍然共享（父序列被释放后引用计数为 2）；连续批处理完成同样的工作所需的迭代次数远少于静态批处理。

## 实例演算 {#worked-example}

**设定。** 一张 H100 80 GB 上的 Llama-3-8B（KV = BF16 下 128 KiB/token，来自 3.2 章）。权重 ≈ 16 GB；假设引擎预留 90% 的显存（72 GB），在扣除权重和工作区之后，留给 KV 的约为 ≈ 54 GB ≈ 50 GiB。

**以块计的容量。** 块大小 16 token 时，一个块是 $16 \times 128\ \text{KiB} = 2\ \text{MiB}$，所以池里有 $50\ \text{GiB} / 2\ \text{MiB} = 25{,}600$ 个块 = **409,600 个 token**。

**连续预留 vs 分页。** 请求的 prompt 为 1,000 token，平均输出 300，允许的最大输出为 4,096。

- 连续预留：每个请求预留 $1{,}000 + 4{,}096 = 5{,}096$ 个槽位 → $409{,}600 / 5{,}096 \approx 80$ 个并发请求。
- 分页：生命周期末尾的请求占用 $\lceil 1{,}300/16 \rceil = 82$ 个块 = 1,312 个槽位；整个生命周期内平均还更少。大约 $409{,}600 / 1{,}312 \approx 312$ 个并发请求——每步的序列数约为 **4 倍**，在带宽受限的区间里，这就接近 4 倍的吞吐。

**前缀缓存。** 如果所有请求共享一个 800 token 的 system prompt，那 50 个块只存一份，而不是每个请求各存一份，每个请求只需预填充 200 个新的 prompt token：每个请求的预填充 FLOPs 下降 80%，TTFT 的大部分也随之下降。

**分块预填充预算。** 根据 3.1 章，这个模型一次解码迭代大约 5–10 ms，预填充在 H100 上大致以 25–35K token/s 的速度运行。2,048 token 的预算会让包含预填充分块的迭代增加大约 60–80 ms；512 的预算增加约 15–20 ms。如果你的 ITL SLO 是 50 ms，512 token 的预算更稳妥——代价是长 prompt 的 TTFT 更慢。（这些耗时只是数量级估计；请在你自己的部署上测量。）

## 实践中的做法（2025–2026） {#in-practice-20252026}

- **所有主流引擎**（vLLM、SGLang、TensorRT-LLM、TGI）都实现了连续批处理和分页 KV。TensorRT-LLM 把连续批处理称为“in-flight batching”。
- **分块预填充在 vLLM V1 引擎中默认开启**，它把预填充和解码统一到一个带 token 预算的调度步骤里。
- **前缀缓存在 vLLM V1（基于哈希）和 SGLang（radix tree）中默认开启。** API 提供方把它以“prompt caching”的形式开放给用户，缓存的输入有折扣价，这就是 agent 框架让 prompt 保持只追加的原因。
- 块大小是 kernel 的调优参数（vLLM 在 GPU 上默认 16 个 token；有些 kernel 偏好更大的页）。
- 教学实现：[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 用约 1,200 行实现了分页 KV、前缀缓存和连续批处理，是在真实代码里看懂这一切的最好方式。

## 常见误区 {#common-pitfalls-misconceptions}

- **“连续批处理意味着更大的批。”** 它意味着*更满*的批：同样的最大 batch，被保持忙碌。它的收益来自消除空闲槽位和排队延迟。
- **把 `max_num_seqs` 设得过大。** 并发序列数超过 KV 能容纳的量，会导致持续的抢占和重算——吞吐下降，尾延迟爆炸。
- **破坏前缀缓存。** 在 system prompt 的*开头*放时间戳或请求 ID，会让每个 prompt 都独一无二，命中率为零。把可变内容放在最后。
- **指望前缀缓存加速解码。** 它只去掉预填充的工作（TTFT、GPU 时间）；解码每一步仍然要读完整的 KV。
- **用随机 prompt 做 benchmark** 就断定前缀缓存没用——或者用完全相同的 prompt 就断定能带来 10 倍加速。要用你真实流量的前缀结构。

## 本章要点 {#key-takeaways}

- 迭代级（连续）批处理在每次前向计算时准入并退出请求；静态批处理会把槽位浪费在已完成的序列上。
- PagedAttention 把 KV 存放在固定大小的块里，通过每个序列的块表寻址；每个序列的浪费不到一个块，并且块可以通过引用计数和写时复制共享。
- 分块预填充用 token 预算限制迭代长度，在 TTFT（预填充速度）与 TPOT/ITL（解码平滑度）之间做权衡。
- 前缀缓存（基于哈希的块或 radix tree）为共享前缀复用 KV，降低 TTFT 和成本——对聊天和 agent 至关重要。
- 内存耗尽时，调度器通过换出到 CPU 或稍后重算来抢占。

## 自测 {#check-your-understanding}

??? question "一个静态 batch 有 4 个请求，输出长度为 100、100、100、1,000。利用率是多少？如果队列永不为空，连续批处理能达到多少？"
    $1{,}300/(4 \times 1{,}000) = 32.5\%$。使用连续批处理且队列非空时，每个完成的槽位在下一次迭代就会被补上，所以利用率趋近 100%（减去调度开销）。

??? question "块大小 16，序列长度 1,000。需要多少个块，浪费多少个槽位？"
    $\lceil 1000/16 \rceil = 63$ 个块 = 1,008 个槽位，所以浪费 8 个槽位。

??? question "为什么只有最后一个、未写满的 prompt 块需要写时复制？"
    写满的块再也不会被写入（过去 token 的 K/V 不会改变），所以可以永久共享。只有未写满的块会接收新 token，而不同的样本会往里写不同的 token。

??? question "你把分块预填充的 token 预算从 2,048 降到 256。哪个指标变好，哪个变差？"
    ITL/TPOT 变得更低、更稳定（迭代更短）。长 prompt 的 TTFT 变差（完成预填充需要更多次迭代），并且吞吐可能略有下降，因为小分块对 GPU 的利用效率较低。

??? question "为什么 agent 工作负载的前缀缓存命中率很高？"
    每一轮都会重发整个之前的上下文（system prompt、工具定义、历史）并追加一小部分；除了新的后缀之外，其余都能匹配到缓存的前缀。

## 延伸阅读 {#further-reading}

| 论文 | 为什么读 |
|---|---|
| :material-star: [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) — Yu et al., OSDI 2022 | 迭代级调度（连续批处理）与选择性批处理；如今已无处不在。 |
| :material-star: [Efficient Memory Management for LLM Serving with PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180) — Kwon et al., SOSP 2023 | 带块表和写时复制的块式 KV；几乎零碎片，吞吐提升 2–4 倍。vLLM 背后的论文。 |
| :material-star: [SGLang: Efficient Execution of Structured LM Programs (RadixAttention)](https://arxiv.org/abs/2312.07104) — Zheng et al., 2023 | 带 LRU 驱逐和缓存感知调度的 radix tree 前缀缓存，外加快速约束解码。 |
| :material-star: [Sarathi-Serve: Taming Throughput-Latency Tradeoff](https://arxiv.org/abs/2403.02310) — Agrawal et al., OSDI 2024 | 分块预填充与无卡顿调度；vLLM 默认调度器的基础。 |
| :material-book-open-variant: [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) | 约 1,200 行、可读性强的分页 KV、前缀缓存和连续批处理实现。 |
