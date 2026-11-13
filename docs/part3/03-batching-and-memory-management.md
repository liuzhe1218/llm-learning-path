# 3.3 Batching & Memory Management

!!! abstract "What you'll learn"
    - Why **static batching** wastes GPU time, and how **continuous (iteration-level) batching** fixes it.
    - Why contiguous KV allocation fragments memory, and how **PagedAttention** uses block tables like OS virtual memory.
    - Implement a small paged block allocator with reference counting and **copy-on-write**.
    - Explain **chunked prefill** and the TTFT/TPOT trade-off it controls.
    - Explain **prefix caching** and SGLang's **RadixAttention**, plus scheduling and preemption (FCFS, swap, recompute).

## Why this matters / the problem

Chapter 3.1 showed that decode throughput grows almost linearly with the number of sequences in the batch, and Chapter 3.2 showed that the KV cache decides how many sequences fit in memory. A serving engine therefore has two jobs, which together form its "operating system":

1. **Scheduling** — keep as many useful sequences in every forward pass as possible, while respecting latency SLOs.
2. **Memory management** — pack KV caches into GPU memory with as little waste as possible, and reuse them when requests share content.

Before 2022–2023, serving systems did both badly: they batched whole requests and reserved a contiguous KV region sized for the maximum possible length. The techniques in this chapter — continuous batching (Orca), PagedAttention (vLLM), chunked prefill (Sarathi-Serve) and prefix caching (RadixAttention in SGLang) — are what turned LLM serving into a high-throughput business, and every serious engine now implements all of them.

## Intuition

**Static batching is a bus that leaves only when every passenger has arrived at their destination.** Four requests start together; one needs 20 tokens, another 500. The short requests finish early, but their slots stay empty until the longest one ends, and new requests wait in the queue.

**Continuous batching is a train where passengers get on and off at every station.** Each forward pass (each decode "iteration") is a station: finished sequences leave, waiting sequences join.

**PagedAttention is virtual memory for the KV cache.** Instead of reserving one big contiguous array per request (and guessing how long it will get), the cache is split into small fixed-size **blocks** (pages). A per-request **block table** maps logical block numbers to whatever physical blocks happen to be free — exactly how an OS maps virtual pages to physical frames. Memory is allocated one block at a time as the sequence grows, and identical blocks can be shared between sequences.

## How it works

### Static batching and its waste

With static (request-level) batching, a batch of $B$ requests runs until **all** finish. If output lengths are $N_1,\dots,N_B$, the batch takes $\max_i N_i$ steps but does only $\sum_i N_i$ useful token-steps. The utilization is

$$
U_{\text{static}} = \frac{\sum_i N_i}{B \cdot \max_i N_i}.
$$

In words: the fraction of batch slots doing useful work. With lengths 20, 50, 100 and 500, $U = 670 / 2000 = 33.5\%$. Real output lengths have a long tail, so static batching typically wastes most of the batch.

### Continuous (iteration-level) batching

Orca's idea: schedule at the granularity of one model iteration. After every forward pass the scheduler (1) removes sequences that emitted EOS or hit their length limit, (2) admits waiting requests if memory allows, and (3) builds the next batch from all running sequences.

```text
time (iterations) ─▶   1   2   3   4   5   6   7   8   9  10
static batching
  slot 0  req A       ■   ■   ■   ·   ·   ·   ·   ·   ·   ·     A done at 3, slot idles
  slot 1  req B       ■   ■   ■   ■   ■   ■   ■   ■   ·   ·
  slot 2  req C       ■   ■   ■   ■   ■   ·   ·   ·   ·   ·
  (req D, E wait in the queue until iteration 9)
continuous batching
  slot 0              A   A   A   D   D   D   D   E   E   E     D joins at 4, E at 8
  slot 1              B   B   B   B   B   B   B   B   F   F
  slot 2              C   C   C   C   C   G   G   G   G   G
```

One complication: sequences in the batch are at different positions and have different context lengths, so attention cannot be one dense `[batch, seq]` tensor operation. Engines flatten all tokens of the step into one "ragged" batch (no padding) for the matrix multiplies, which do not care which sequence a token belongs to, and use attention kernels that take per-sequence lengths and KV locations (Chapter 3.4).

### Memory fragmentation

If each request gets a contiguous KV region reserved for its maximum length (say 4,096 tokens) when it starts:

- **Internal fragmentation:** a request that ends after 300 tokens wasted the other 3,796 slots for its whole lifetime.
- **Reservation waste:** even for requests that eventually grow long, slots are reserved long before they are used.
- **External fragmentation:** free memory ends up split into gaps too small for a new max-length reservation.

The vLLM paper measured that in earlier systems only a minority of KV memory actually held token states. Since KV capacity determines batch size, wasted KV memory is directly wasted throughput.

### PagedAttention: block tables

Split the KV pool into physical blocks, each holding the K and V of $S_{\text{blk}}$ consecutive tokens for every layer (vLLM's default block size is 16 tokens). For each sequence keep a **block table**: entry $j$ is the physical block holding tokens $[jS_{\text{blk}}, (j+1)S_{\text{blk}})$. Token position $p$ lives at

$$
\text{physical block} = \text{table}\big[\lfloor p / S_{\text{blk}} \rfloor\big], \qquad \text{offset} = p \bmod S_{\text{blk}}.
$$

A new block is allocated only when the last one is full, so at most $S_{\text{blk}} - 1$ slots per sequence are wasted. The attention kernel reads the block table and gathers K/V from the scattered physical blocks; Chapter 3.4 discusses what this does to kernel design.

**Example.** Block size 4. Sequence A has 10 tokens, sequence B has 6.

```text
logical view                block tables              physical KV pool (block id: contents)
A: t0..t9                   A: [7, 2, 5]              0: free     4: B t4-t5 (2/4 used)
B: t0..t5                   B: [3, 4]                 1: free     5: A t8-t9 (2/4 used)
                                                      2: A t4-t7  6: free
                                                      3: B t0-t3  7: A t0-t3
```

Token A·t9 is at block `A_table[9 // 4] = A_table[2] = 5`, offset `9 % 4 = 1`. Waste is 2 slots in block 5 and 2 in block 4 — not thousands.

### Sharing and copy-on-write

Block tables make **sharing** cheap. With parallel sampling ($n$ completions of one prompt) or beam search, all completions share the prompt's KV blocks. Each physical block carries a **reference count**. When a sequence must write into a block whose count is greater than 1 (e.g. the partially filled last prompt block), the engine first **copies** the block, points this sequence's table at the copy, and decrements the original's count — copy-on-write, exactly as in `fork()` in an operating system. Prompt KV is stored once instead of $n$ times.

### Chunked prefill

Continuous batching leaves one problem: when a long prompt arrives, its prefill (thousands of tokens, tens to hundreds of milliseconds) runs in some iteration, and every decoding sequence in that iteration waits. Users see a stall: a spike in ITL.

Sarathi-Serve's fix is to give each iteration a **token budget** (e.g. 512 or 2,048 tokens). The scheduler first adds one token for every running decode, then fills the remaining budget with a **chunk** of a pending prefill. A 10,000-token prompt with a ~1,500-token chunk is prefilled over ~7 iterations, piggybacked on the ongoing decodes. The prefill chunks make the iteration compute-bound (good use of the GPU), and decodes are never stalled for more than one bounded iteration.

The **trade-off** is controlled by the budget:

- **Smaller budget** → each iteration is short → low, stable TPOT/ITL, but long prompts need more iterations → higher TTFT.
- **Larger budget** → faster prefill → lower TTFT, but longer iterations → higher TPOT for everyone else.

Chunking requires attention kernels that handle a chunk of queries attending to a KV cache that already contains the earlier chunks — the same kernel used for prefix caching below.

### Prefix caching and RadixAttention

Many requests share a **prefix**: the same system prompt, few-shot examples, tool definitions, document, or — in multi-turn chat and agents — the entire previous conversation. Recomputing that prefix's KV for every request is pure waste. **Prefix caching** keeps KV blocks after a request finishes and reuses them when a later request starts with the same tokens.

Two common implementations:

- **Hash-based blocks (vLLM).** Each *full* block is identified by a hash of its tokens together with the hash of all preceding blocks. A new request hashes its prompt block by block and reuses every leading block found in the table. Free cached blocks are evicted LRU when memory is needed.
- **Radix tree (SGLang's RadixAttention).** All cached token sequences are stored in a **radix tree** (a trie whose edges are labelled with token sequences, splitting edges where sequences diverge). Each node points at the KV for its edge's tokens. A new request walks the tree to find its **longest matching prefix**, reuses that KV, and prefills only the remainder; the new tokens are inserted as a branch. Leaves not used by running requests are evicted LRU. SGLang's scheduler is also **cache-aware**: it prefers requests with longer cached prefixes, to increase hit rates.

**Example.** A system prompt `S` (1,000 tokens) is shared by three requests.

```text
request 1: S + "Translate: hello"            ──▶ miss: prefill 1,000 + 4 tokens
request 2: S + "Translate: goodbye"          ──▶ hit S + "Translate:" ; prefill 1-2 tokens
request 3: S + "Summarize: ..." (200 tok)    ──▶ hit S ; prefill ~200 tokens

radix tree after the three requests
  root ── S(1000) ──┬── "Translate:" ──┬── "hello"
                    │                  └── "goodbye"
                    └── "Summarize: ..." (200)
```

Requests 2 and 3 skip ~1,000 tokens of prefill each: TTFT drops sharply and the GPU time is freed for other work. In an agent loop, where turn $k$ resends turns $1..k-1$ verbatim, hit rates can approach 100% of the prefix.

### Scheduling policies and preemption

A typical scheduler loop, run every iteration:

1. **Running decodes first** — keep the latency promise to admitted requests.
2. **Admit waiting requests** in **FCFS** order (first-come, first-served) if KV blocks are available, using prefix-cache hits to reduce their cost and chunked prefill to bound the iteration.
3. **If memory runs out** while running sequences grow, **preempt** a request (usually the most recently admitted, to keep FCFS fairness) and free its blocks. Two ways to bring it back later:
    - **Swap**: copy its KV blocks to CPU memory and back when resumed. Cost: PCIe transfer.
    - **Recompute**: drop the KV and redo prefill over prompt + generated tokens when resumed. Cost: GPU compute — often cheaper than it sounds, since prefill is fast and parallel.

Engines also offer priority scheduling, per-tenant fairness, and limits such as `max_num_seqs` (sequences per batch) and `max_num_batched_tokens` (the chunked-prefill budget). Frequent preemption is a sign the engine is over-admitting; it shows up as latency spikes and wasted compute.

## Code

A tiny simulation of a paged KV block allocator with reference counts and copy-on-write, followed by a comparison of static vs continuous batching utilization. Pure Python — the logic is the same as in real engines, minus the GPU tensors.

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

    def add_seq(self, sid, n_tokens):              # prefill: allocate ceil(n/bs) blocks
        self.tables[sid] = [self._alloc() for _ in range(-(-n_tokens // self.bs))]
        self.lens[sid] = n_tokens

    def append_token(self, sid):                   # decode: maybe a new block, maybe CoW
        table, n = self.tables[sid], self.lens[sid]
        if n % self.bs == 0:                       # last block full -> new block
            table.append(self._alloc())
        elif self.ref[table[-1]] > 1:              # writing into a shared block -> copy
            self.ref[table[-1]] -= 1
            table[-1] = self._alloc()              # (real engine: copy the KV data too)
        self.lens[sid] = n + 1

    def fork(self, parent, child):                 # parallel sampling: share all blocks
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
alloc.add_seq("p", 6)                  # prompt of 6 tokens -> 2 blocks (last half full)
alloc.fork("p", "s1"); alloc.fork("p", "s2")
for s in ("s1", "s2"):
    alloc.append_token(s)              # each sample writes token 7 -> CoW of block 2
print({s: alloc.tables[s] for s in ("p", "s1", "s2")}, "free:", len(alloc.free))
alloc.free_seq("p")
print("ref counts:", alloc.ref, "free:", len(alloc.free))

# Static vs continuous batching: useful slots / total slots, same request stream.
random.seed(0)
lens = [min(int(random.expovariate(1 / 200)) + 1, 2000) for _ in range(256)]
B = 16
static_steps = sum(max(lens[i:i + B]) for i in range(0, len(lens), B))
queue, running, cont_steps = list(lens), [], 0
while queue or running:
    while queue and len(running) < B:   # admit new requests every iteration
        running.append(queue.pop(0))
    running = [r - 1 for r in running if r > 1]
    cont_steps += 1
print(f"static: {static_steps} steps, util {sum(lens) / (B * static_steps):.0%}; "
      f"continuous: {cont_steps} steps, util {sum(lens) / (B * cont_steps):.0%}")
```

Expected output: the two forked samples get private copies of the shared last prompt block while the first block stays shared (ref count 2 after the parent is freed); continuous batching finishes the same work in far fewer iterations than static batching.

## Worked example

**Setup.** Llama-3-8B (KV = 128 KiB/token in BF16, from Chapter 3.2) on one H100 80 GB. Weights ≈ 16 GB; suppose the engine reserves 90% of memory (72 GB), leaving ≈ 54 GB ≈ 50 GiB for KV after weights and workspace.

**Capacity in blocks.** With 16-token blocks, one block is $16 \times 128\ \text{KiB} = 2\ \text{MiB}$, so the pool has $50\ \text{GiB} / 2\ \text{MiB} = 25{,}600$ blocks = **409,600 tokens**.

**Contiguous reservation vs paging.** Requests have prompt 1,000 tokens, average output 300, maximum output allowed 4,096.

- Contiguous: reserve $1{,}000 + 4{,}096 = 5{,}096$ slots per request → $409{,}600 / 5{,}096 \approx 80$ concurrent requests.
- Paged: a request at the end of its life uses $\lceil 1{,}300/16 \rceil = 82$ blocks = 1,312 slots; on average over its life even fewer. Roughly $409{,}600 / 1{,}312 \approx 312$ concurrent requests — about **4×** more sequences per step, which in the bandwidth-bound regime is close to 4× the throughput.

**Prefix caching.** If all requests share an 800-token system prompt, those 50 blocks are stored once instead of per request, and each request only prefills 200 new prompt tokens: the prefill FLOPs per request drop by 80%, and so does most of the TTFT.

**Chunked prefill budget.** From Chapter 3.1, a decode iteration for this model takes ~5–10 ms and prefill runs at very roughly 25–35K tokens/s on an H100. A budget of 2,048 tokens adds very roughly 60–80 ms to an iteration containing a prefill chunk; a budget of 512 adds ~15–20 ms. If your ITL SLO is 50 ms, a 512-token budget is the safer choice — at the cost of a slower TTFT for long prompts. (These timings are order-of-magnitude estimates; measure your own deployment.)

## In practice (2025–2026)

- **All major engines** (vLLM, SGLang, TensorRT-LLM, TGI) implement continuous batching and paged KV. TensorRT-LLM calls continuous batching "in-flight batching".
- **Chunked prefill is on by default** in vLLM's V1 engine, which unified prefill and decode into one token-budgeted scheduler step.
- **Prefix caching is on by default** in vLLM V1 (hash-based) and in SGLang (radix tree). API providers expose it to users as "prompt caching" with discounted cached-input pricing, which is why agent frameworks keep prompts append-only.
- Block size is a kernel-tuning parameter (vLLM defaults to 16 tokens on GPUs; some kernels prefer larger pages).
- Teaching implementations: [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) implements paged KV, prefix caching and continuous batching in ~1,200 lines and is the best way to see all of this in real code.

## Common pitfalls / misconceptions

- **"Continuous batching means larger batches."** It means *fuller* batches: the same maximum batch, kept busy. Its gains come from removing idle slots and queueing delay.
- **Over-sizing `max_num_seqs`.** More concurrent sequences than KV can hold leads to constant preemption and recompute — throughput falls and tail latency explodes.
- **Breaking prefix caches.** Putting a timestamp or a request ID at the *start* of the system prompt makes every prompt unique and the hit rate zero. Put variable content last.
- **Expecting prefix caching to speed up decode.** It only removes prefill work (TTFT, GPU time); decode still reads the full KV every step.
- **Benchmarking with random prompts** and concluding prefix caching does not help — or with identical prompts and concluding it gives 10× speedups. Use your real traffic's prefix structure.

## Key takeaways

- Iteration-level (continuous) batching admits and retires requests every forward pass; static batching wastes slots on finished sequences.
- PagedAttention stores KV in fixed-size blocks addressed through per-sequence block tables; waste is under one block per sequence, and blocks can be shared with reference counts and copy-on-write.
- Chunked prefill bounds iteration length with a token budget, trading TTFT (prefill speed) against TPOT/ITL (decode smoothness).
- Prefix caching (hash-based blocks or a radix tree) reuses KV for shared prefixes, cutting TTFT and cost — essential for chat and agents.
- When memory runs out, schedulers preempt by swapping to CPU or recomputing later.

## Check your understanding

??? question "Output lengths in a static batch of 4 are 100, 100, 100, 1,000. What is the utilization? What would continuous batching achieve if the queue is never empty?"
    $1{,}300/(4 \times 1{,}000) = 32.5\%$. With continuous batching and a non-empty queue, each finished slot is refilled the next iteration, so utilization approaches 100% (minus scheduling overhead).

??? question "Block size 16, sequence length 1,000. How many blocks, and how many wasted slots?"
    $\lceil 1000/16 \rceil = 63$ blocks = 1,008 slots, so 8 wasted slots.

??? question "Why is copy-on-write needed only for the last, partially filled prompt block?"
    Full blocks are never written again (K/V of past tokens never change), so they can be shared forever. Only the partially filled block will receive new tokens, and different samples write different tokens there.

??? question "You lowered the chunked-prefill token budget from 2,048 to 256. Which metric improves, which gets worse?"
    ITL/TPOT gets lower and more stable (shorter iterations). TTFT gets worse for long prompts (more iterations to finish prefill), and throughput may drop slightly because small chunks use the GPU less efficiently.

??? question "Why do agent workloads get very high prefix-cache hit rates?"
    Each turn resends the entire previous context (system prompt, tool definitions, history) and appends a little; everything except the new suffix matches a cached prefix.

## Further reading

| Paper | Why read it |
|---|---|
| :material-star: [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) — Yu et al., OSDI 2022 | Iteration-level scheduling (continuous batching) and selective batching; now universal. |
| :material-star: [Efficient Memory Management for LLM Serving with PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180) — Kwon et al., SOSP 2023 | Block-based KV with block tables and copy-on-write; near-zero fragmentation, 2–4× throughput. The paper behind vLLM. |
| :material-star: [SGLang: Efficient Execution of Structured LM Programs (RadixAttention)](https://arxiv.org/abs/2312.07104) — Zheng et al., 2023 | Radix-tree prefix caching with LRU eviction and cache-aware scheduling, plus fast constrained decoding. |
| :material-star: [Sarathi-Serve: Taming Throughput-Latency Tradeoff](https://arxiv.org/abs/2403.02310) — Agrawal et al., OSDI 2024 | Chunked prefill and stall-free scheduling; the basis of vLLM's default scheduler. |
| :material-book-open-variant: [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) | A ~1,200-line readable implementation of paged KV, prefix caching and continuous batching. |
