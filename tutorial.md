# Complete Guide to nano-vllm: A Deep Dive

## What is nano-vllm?

  Analogy: A High-Speed Restaurant Kitchen

  Imagine nano-vllm as a high-efficiency restaurant kitchen that can serve hundreds of customers simultaneously. Instead of cooking one meal at a time, the kitchen:
  - Takes multiple orders at once (batching)
  - Remembers what was already cooked (KV cache)
  - Uses specialized stations for different tasks (tensor parallelism)
  - Pre-heats common ingredients (prefix caching)
  - Uses optimized cooking workflows (CUDA graphs)

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                     NANO-VLLM ARCHITECTURE                          │
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                     │
  │   User Prompts                                                      │
  │       │                                                             │
  │       ▼                                                             │
  │  ┌─────────┐     ┌───────────┐     ┌─────────────┐                  │
  │  │   LLM   │────▶│ LLMEngine │────▶│  Scheduler  │                  │
  │  └─────────┘     └───────────┘     └──────┬──────┘                  │
  │                        │                  │                         │
  │                        │           ┌──────┴──────┐                  │
  │                        │           │BlockManager │                  │
  │                        │           └──────┬──────┘                  │
  │                        ▼                  │                         │
  │               ┌─────────────┐             │                         │
  │               │ ModelRunner │◀────────────┘                         │
  │               └──────┬──────┘                                       │
  │                      │                                              │
  │         ┌────────────┼────────────┐                                 │
  │         ▼            ▼            ▼                                 │
  │   ┌──────────┐ ┌──────────┐ ┌─────────┐                             │
  │   │Qwen3Model│ │ KV Cache │ │ Sampler │                             │
  │   └──────────┘ └──────────┘ └─────────┘                             │
  │                                                                     │
  │   Output Tokens                                                     │
  └─────────────────────────────────────────────────────────────────────┘
```

  ---
## Part 1: The Entry Point - LLMEngine

  File: nanovllm/engine/llm_engine.py:15-93

  Analogy: The Restaurant Manager

  The LLMEngine is like the restaurant manager who coordinates everything:
  - Takes customer orders (prompts)
  - Assigns them to the scheduling queue
  - Tells the kitchen (ModelRunner) what to cook
  - Delivers finished meals (tokens) back to customers

  How It Works

```
  ┌──────────────────────────────────────────────────────────────┐
  │                    LLMEngine.generate()                      │
  ├──────────────────────────────────────────────────────────────┤
  │                                                              │
  │  1. ADD REQUESTS                                             │
  │     ┌─────────┐  ┌─────────┐  ┌─────────┐                    │
  │     │Prompt 1 │  │Prompt 2 │  │Prompt 3 │                    │
  │     └────┬────┘  └────┬────┘  └────┬────┘                    │
  │          │            │            │                         │
  │          ▼            ▼            ▼                         │
  │     ┌────────────────────────────────────┐                   │
  │     │        Scheduler.waiting           │                   │
  │     └────────────────────────────────────┘                   │
  │                                                              │
  │  2. MAIN LOOP (while not finished):                          │
  │                                                              │
  │     ┌─────────────────┐    ┌──────────────┐                  │
  │     │ schedule()      │───▶│ Batch + Mode │                  │
  │     └─────────────────┘    │(prefill/decode)                 │
  │                            └───────┬──────┘                  │
  │                                    │                         │
  │                                    ▼                         │
  │     ┌─────────────────┐    ┌──────────────┐                  │
  │     │ model_runner.   │───▶│ New Tokens   │                  │
  │     │ run()           │    └───────┬──────┘                  │
  │     └─────────────────┘            │                         │
  │                                    ▼                         │
  │     ┌─────────────────┐    ┌──────────────┐                  │
  │     │ postprocess()   │───▶│Update states │                  │
  │     └─────────────────┘    └──────────────┘                  │
  │                                                              │
  │  3. RETURN: decoded text + token IDs                         │
  └──────────────────────────────────────────────────────────────┘
```

  Key Code Walkthrough

```python
  # llm_engine.py:17-34 - Initialization

  def __init__(self, model, **kwargs):
      # 1. Create configuration
      config = Config(model, **config_kwargs)

      # 2. Spawn worker processes for tensor parallelism
      # (like hiring additional chefs for a bigger kitchen)
      for i in range(1, config.tensor_parallel_size):
          process = ctx.Process(target=ModelRunner, args=(config, i, event))
          process.start()

      # 3. Create main ModelRunner (head chef)
      self.model_runner = ModelRunner(config, 0, self.events)

      # 4. Load tokenizer and create scheduler
      self.tokenizer = AutoTokenizer.from_pretrained(config.model)
      self.scheduler = Scheduler(config)

  # llm_engine.py:48-54 - The Step Function (one iteration)
  def step(self):
      # 1. Get next batch from scheduler
      seqs, is_prefill = self.scheduler.schedule()

      # 2. Run model and get new tokens
      token_ids = self.model_runner.call("run", seqs, is_prefill)

      # 3. Update sequence states
      self.scheduler.postprocess(seqs, token_ids)

      return outputs, num_tokens
```

  Gotcha: The num_tokens calculation is clever - positive for prefill (total tokens processed), negative for decode (negative batch size). This lets the progress bar distinguish between phases.

  ---
## Part 2: The Scheduler - Traffic Controller

  File: nanovllm/engine/scheduler.py:8-71

  Analogy: Airport Gate Controller

  The Scheduler is like an airport gate controller managing two runways:
  - Waiting queue: Planes waiting to take off (new prompts)
  - Running queue: Planes in the air (generating tokens)

  It decides which planes can take off (prefill) and manages the airspace (decode).

  Two-Phase Scheduling

```
  ┌──────────────────────────────────────────────────────────────┐
  │                    SCHEDULING PHASES                         │
  ├──────────────────────────────────────────────────────────────┤
  │                                                              │
  │  PHASE 1: PREFILL (Process new prompts)                      │
  │  ─────────────────────────────────────────────               │
  │                                                              │
  │  waiting: [Seq1: 100 tokens] [Seq2: 50 tokens] [Seq3: 200]   │
  │                 │                   │                        │
  │                 ▼                   ▼                        │
  │           ┌─────────────────────────────┐                    │
  │           │   Batch (if room):          │                    │
  │           │   Seq1 + Seq2 = 150 tokens  │                    │
  │           │   (Seq3 waits - over limit) │                    │
  │           └─────────────────────────────┘                    │
  │                                                              │
  │  PHASE 2: DECODE (Generate one token per sequence)           │
  │  ─────────────────────────────────────────────               │
  │                                                              │
  │  running: [Seq1] [Seq2] [Seq4] [Seq5]                        │
  │              │      │      │      │                          │
  │              ▼      ▼      ▼      ▼                          │
  │           ┌─────────────────────────────┐                    │
  │           │   Batch all running seqs    │                    │
  │           │   (1 token each = 4 tokens) │                    │
  │           └─────────────────────────────┘                    │
  │                                                              │
  │  Priority: PREFILL > DECODE                                  │
  │  (New prompts are processed before continuing generation)    │
  └──────────────────────────────────────────────────────────────┘
```

  Key Code Walkthrough

```python
  # scheduler.py:24-41 - Prefill Phase
  def schedule(self) -> tuple[list[Sequence], bool]:
      scheduled_seqs = []

      # Try to pack as many waiting sequences as possible
      while self.waiting and num_seqs < self.max_num_seqs:
          seq = self.waiting[0]

          # Check constraints:
          # 1. Would exceed max batch tokens?
          # 2. Enough KV cache blocks available?
          if num_batched_tokens + len(seq) > self.max_num_batched_tokens \
             or not self.block_manager.can_allocate(seq):
              break

          # Allocate KV cache and move to running
          self.block_manager.allocate(seq)
          num_batched_tokens += len(seq) - seq.num_cached_tokens  # Prefix cache!
          self.waiting.popleft()
          self.running.append(seq)
          scheduled_seqs.append(seq)

      if scheduled_seqs:
          return scheduled_seqs, True  # is_prefill = True

  # scheduler.py:43-58 - Decode Phase (when no waiting sequences)
      # decode
      while self.running and num_seqs < self.max_num_seqs:
          seq = self.running.popleft()

          # If KV cache is full, preempt (evict) sequences
          while not self.block_manager.can_append(seq):
              if self.running:
                  self.preempt(self.running.pop())  # Evict last-added
              else:
                  self.preempt(seq)  # Must evict ourselves
                  break
          else:
              self.block_manager.may_append(seq)  # Allocate new block if needed
              scheduled_seqs.append(seq)

      return scheduled_seqs, False  # is_prefill = False
```

  Gotcha: The else clause on the while loop is Python's little-known feature - it executes when the loop completes without a break. This is how the scheduler knows if eviction was successful.

  ---
## Part 3: Sequence - The Individual Request

  File: nanovllm/engine/sequence.py:14-83

  Analogy: A Food Order Ticket

  Each Sequence is like an order ticket in a restaurant:
  - Records what was ordered (prompt tokens)
  - Tracks what's been served (completion tokens)
  - Notes special instructions (temperature, max tokens)
  - Has a ticket number (seq_id)

```
  ┌────────────────────────────────────────────────────────────────┐
  │                        SEQUENCE                                │
  ├────────────────────────────────────────────────────────────────┤
  │                                                                │
  │  seq_id: 42                                                    │
  │  status: RUNNING                                               │
  │                                                                │
  │  Token IDs:                                                    │
  │  ┌────────────────────────────────────────────────────────┐    │
  │  │ [101, 2054, 2003, 1996, ... , 8021, 999]               │    │
  │  │  ◄──── prompt_tokens ────► ◄─ completion ─►            │    │
  │  │       (num_prompt = 10)    (num_completion = 5)        │    │
  │  └────────────────────────────────────────────────────────┘    │
  │                                                                │
  │  Block Table (KV cache locations):                             │
  │  ┌──────┬──────┬──────┐                                        │
  │  │ B:7  │ B:12 │ B:3  │  ◄── 3 blocks allocated                │
  │  └──────┴──────┴──────┘                                        │
  │                                                                │
  │  Block Size: 256 tokens per block                              │
  │  Cached Tokens: 512 (2 full blocks were prefix-cached)         │
  │                                                                │
  └────────────────────────────────────────────────────────────────┘
```

  Key Properties

```python
  # sequence.py:53-63 - Block calculations
  @property
  def num_cached_blocks(self):
      return self.num_cached_tokens // self.block_size  # How many blocks were cache hits

  @property
  def num_blocks(self):
      # Ceiling division: (tokens + block_size - 1) // block_size
      return (self.num_tokens + self.block_size - 1) // self.block_size

  @property
  def last_block_num_tokens(self):
      # How many tokens are in the last (partially filled) block
      return self.num_tokens - (self.num_blocks - 1) * self.block_size
```

  Gotcha: The __getstate__ and __setstate__ methods are for pickling (serialization). After prefill, it only stores last_token instead of the full token list to save memory when sending to worker processes.

  ---
## Part 4: BlockManager - Memory Tetris

  File: nanovllm/engine/block_manager.py:26-112

  Analogy: Hotel Room Manager

  The BlockManager is like a hotel manager allocating rooms:
  - Each "room" (block) holds 256 token "guests"
  - Rooms can be reused by new guests with the same "booking hash" (prefix caching)
  - When a guest checks out (sequence finishes), the room becomes available
  - Reference counting allows room sharing (multiple sequences sharing a prefix)

```
  ┌────────────────────────────────────────────────────────────────┐
  │                    KV CACHE BLOCK MANAGER                      │
  ├────────────────────────────────────────────────────────────────┤
  │                                                                │
  │  Physical Blocks (like hotel rooms):                           │
  │  ┌────────────────────────────────────────────────────────┐    │
  │  │ Block 0 │ Block 1 │ Block 2 │ Block 3 │ ... │ Block N  │    │
  │  │ ref=2   │ ref=1   │ ref=0   │ ref=1   │     │ ref=0    │    │
  │  │ hash=A  │ hash=B  │ (free)  │ hash=C  │     │ (free)   │    │
  │  └────────────────────────────────────────────────────────┘    │
  │                                                                │
  │  Sequence 1: [Block 0] -> [Block 1]                            │
  │  Sequence 2: [Block 0] -> [Block 3]  ◄── Shares Block 0!       │
  │                                                                │
  │  PREFIX CACHING FLOW:                                          │
  │  ───────────────────                                           │
  │                                                                │
  │  Prompt: "Hello, my name is"                                   │
  │           │                                                    │
  │           ▼                                                    │
  │  Hash = xxhash64("Hello, my name is")                          │
  │           │                                                    │
  │           ▼                                                    │
  │  Lookup: hash_to_block_id[hash] = Block 0?                     │
  │           │                                                    │
  │      ┌────┴────┐                                               │
  │      ▼         ▼                                               │
  │   CACHE HIT   CACHE MISS                                       │
  │   ref_count++  allocate new block                              │
  │   skip prefill copy tokens                                     │
  │                                                                │
  └────────────────────────────────────────────────────────────────┘
```

  Key Code Walkthrough

```python
  # block_manager.py:36-41 - Hash computation for prefix matching
  @classmethod
  def compute_hash(cls, token_ids: list[int], prefix: int = -1):
      h = xxhash.xxh64()
      if prefix != -1:
          h.update(prefix.to_bytes(8, "little"))  # Chain with previous block's hash
      h.update(np.array(token_ids).tobytes())
      return h.intdigest()
```

  The hash is chained - each block's hash includes the previous block's hash. This ensures that identical token sequences at different positions don't falsely match.

```python
  # block_manager.py:59-82 - Allocation with prefix caching
  def allocate(self, seq: Sequence):
      h = -1
      cache_miss = False

      for i in range(seq.num_blocks):
          token_ids = seq.block(i)  # Get tokens for block i

          # Only compute hash for full blocks
          h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1

          # Try to find existing block with same hash
          block_id = self.hash_to_block_id.get(h, -1)

          if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
              cache_miss = True  # Once we miss, all subsequent blocks miss too

          if cache_miss:
              # Allocate new block from free list
              block_id = self.free_block_ids[0]
              block = self._allocate_block(block_id)
          else:
              # CACHE HIT! Reuse existing block
              seq.num_cached_tokens += self.block_size
              block.ref_count += 1  # Increment reference count

          seq.block_table.append(block_id)
```

  Gotcha: The cache_miss flag is "sticky" - once you miss, you can't hit again. This prevents false matches where later blocks happen to have the same tokens but different prefixes.

  ---
## Part 5: ModelRunner - The GPU Execution Engine

  File: nanovllm/engine/model_runner.py:15-251

  Analogy: Formula 1 Pit Crew

  The ModelRunner is like an F1 pit crew optimizing every millisecond:
  - Pre-computes data layouts (prepare_prefill/prepare_decode)
  - Uses pre-recorded "pit stop routines" (CUDA graphs)
  - Coordinates multiple crew members (tensor parallelism)
  - Has pre-allocated space for parts (KV cache)

```
  ┌────────────────────────────────────────────────────────────────┐
  │                     MODEL RUNNER FLOW                          │
  ├────────────────────────────────────────────────────────────────┤
  │                                                                │
  │  INITIALIZATION:                                               │
  │  ┌────────────────┐                                            │
  │  │ 1. Init NCCL   │ ◄── Distributed communication              │
  │  │ 2. Load Model  │ ◄── Weights from safetensors               │
  │  │ 3. Warmup      │ ◄── Find peak memory usage                 │
  │  │ 4. Alloc KV    │ ◄── Pre-allocate cache based on free mem   │
  │  │ 5. Capture     │ ◄── Record CUDA graphs for decode          │
  │  │    Graphs      │                                            │
  │  └────────────────┘                                            │
  │                                                                │
  │  EXECUTION:                                                    │
  │  ┌────────────────────────────────────────────────────────┐    │
  │  │                                                        │    │
  │  │  is_prefill?                                           │    │
  │  │      │                                                 │    │
  │  │  ┌───┴───┐                                             │    │
  │  │  ▼       ▼                                             │    │
  │  │ YES      NO                                            │    │
  │  │  │        │                                            │    │
  │  │  ▼        ▼                                            │    │
  │  │ prepare  prepare                                       │    │
  │  │ _prefill _decode                                       │    │
  │  │  │        │                                            │    │
  │  │  ▼        ▼                                            │    │
  │  │ Eager    CUDA Graph                                    │    │
  │  │ Exec     Replay                                        │    │
  │  │  │        │                                            │    │
  │  │  └───┬────┘                                            │    │
  │  │      ▼                                                 │    │
  │  │  Sample Token                                          │    │
  │  │                                                        │    │
  │  └────────────────────────────────────────────────────────┘    │
  │                                                                │
  └────────────────────────────────────────────────────────────────┘
```

  KV Cache Allocation

```python
  # model_runner.py:100-118 - Smart memory allocation
  def allocate_kv_cache(self):
      # Get memory stats
      free, total = torch.cuda.mem_get_info()
      peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
      current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

      # Calculate bytes per block
      # 2 (K+V) * num_layers * block_size * num_kv_heads * head_dim * dtype_size
      block_bytes = 2 * num_layers * block_size * num_kv_heads * head_dim * dtype.itemsize

      # Allocate based on 90% GPU utilization
      available = total * 0.9 - used - peak + current
      num_blocks = available // block_bytes

      # Create single contiguous tensor: [2, layers, blocks, block_size, heads, dim]
      self.kv_cache = torch.empty(2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)

      # Assign views to each attention layer
      for module in self.model.modules():
          if hasattr(module, "k_cache"):
              module.k_cache = self.kv_cache[0, layer_id]
              module.v_cache = self.kv_cache[1, layer_id]
```

  CUDA Graph Capture

```
  ┌────────────────────────────────────────────────────────────────┐
  │                    CUDA GRAPHS                                 │
  ├────────────────────────────────────────────────────────────────┤
  │                                                                │
  │  Without CUDA Graphs:                                          │
  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                       │
  │  │CPU  │→│GPU  │→│CPU  │→│GPU  │→│CPU  │  ...                  │
  │  │setup│ │exec │ │setup│ │exec │ │setup│                       │
  │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘                       │
  │     ↑               ↑               ↑                          │
  │   Kernel launch overhead each time                             │
  │                                                                │
  │  With CUDA Graphs:                                             │
  │  ┌────────────────────────────────┐                            │
  │  │ Record once                    │                            │
  │  │ ┌────┬────┬────┬────┬────┐     │                            │
  │  │ │ K1 │ K2 │ K3 │ K4 │ K5 │     │                            │
  │  │ └────┴────┴────┴────┴────┘     │                            │
  │  └────────────────────────────────┘                            │
  │                 │                                              │
  │                 ▼                                              │
  │  ┌────────────────────────────────┐                            │
  │  │ Replay (single launch)         │                            │
  │  │ ┌────────────────────────────┐ │                            │
  │  │ │████████████████████████████│ │  ◄── All kernels fused     │
  │  │ └────────────────────────────┘ │                            │
  │  └────────────────────────────────┘                            │
  │                                                                │
  │  Captured batch sizes: [1, 2, 4, 8, 16, 32, ...]               │
  │  Decode picks smallest graph >= actual batch size              │
  │                                                                │
  └────────────────────────────────────────────────────────────────┘
```

```python
  # model_runner.py:217-241 - CUDA Graph capture
  def capture_cudagraph(self):
      # Pre-allocate input/output tensors
      self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))

      for bs in reversed(self.graph_bs):  # Larger first (to share memory pool)
          graph = torch.cuda.CUDAGraph()

          # Warmup run
          outputs[:bs] = self.model(input_ids[:bs], positions[:bs])

          # Capture
          with torch.cuda.graph(graph, self.graph_pool):
              outputs[:bs] = self.model(input_ids[:bs], positions[:bs])

          self.graphs[bs] = graph
```

  Gotcha: Graphs are captured in reverse order (largest first). This allows smaller graphs to reuse memory from the pool established by larger graphs, reducing total memory footprint.

  Tensor Parallelism Communication

```python
  # model_runner.py:41-48 - Worker process communication via shared memory
  if self.world_size > 1:
      if rank == 0:
          self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)  # 1MB
          dist.barrier()
      else:
          dist.barrier()
          self.shm = SharedMemory(name="nanovllm")
          self.loop()  # Workers enter infinite loop waiting for commands

  # model_runner.py:76-83 - Sending commands to workers
  def write_shm(self, method_name, *args):
      data = pickle.dumps([method_name, *args])  # Serialize
      self.shm.buf[0:4] = len(data).to_bytes(4, "little")  # Size header
      self.shm.buf[4:n+4] = data  # Payload
      for event in self.event:
          event.set()  # Wake up all workers
```

  ---
## Part 6: The Qwen3 Model Architecture

  File: nanovllm/models/qwen3.py:14-215

  Analogy: Assembly Line Factory

  The Qwen3 model is like a factory assembly line:
  1. Embedding: Raw materials (token IDs) → Initial shapes (vectors)
  2. Decoder Layers: Multiple processing stations, each adding refinement
  3. LM Head: Final inspection → Shipping labels (logits)

```
  ┌────────────────────────────────────────────────────────────────┐
  │                    QWEN3 DECODER LAYER                         │
  ├────────────────────────────────────────────────────────────────┤
  │                                                                │
  │  Input: hidden_states                                          │
  │              │                                                 │
  │              ▼                                                 │
  │     ┌────────────────┐                                         │
  │     │  RMSNorm       │ ◄── Pre-normalize                       │
  │     └───────┬────────┘                                         │
  │             │                                                  │
  │             ▼                                                  │
  │     ┌────────────────┐      ┌─────────────────────────────┐    │
  │     │  Self-Attn     │      │  INSIDE SELF-ATTENTION:     │    │
  │     │                │      │                             │    │
  │     │  Q = Wq(x)     │      │  ┌───┐ ┌───┐ ┌───┐          │    │
  │     │  K = Wk(x)     │      │  │ Q │ │ K │ │ V │          │    │
  │     │  V = Wv(x)     │      │  └─┬─┘ └─┬─┘ └─┬─┘          │    │
  │     │                │      │    │     │     │            │    │
  │     │  Q,K = RoPE    │      │  RoPE  RoPE    │            │    │
  │     │  O = Attn(Q,K,V)      │    │     │     │            │    │
  │     │  out = Wo(O)   │      │    └──┬──┘     │            │    │
  │     └───────┬────────┘      │       ▼        │            │    │
  │             │               │   ┌───────┐    │            │    │
  │             │               │   │ Attn  │◄───┘            │    │
  │             │               │   │Score  │                 │    │
  │             │               │   └───┬───┘                 │    │
  │             │               │       ▼                     │    │
  │             │               │   ┌───────┐                 │    │
  │             │               │   │Softmax│                 │    │
  │             │               │   └───────┘                 │    │
  │             │               └─────────────────────────────┘    │
  │             │                                                  │
  │             ├───────────────────┐                              │
  │             ▼                   ▼                              │
  │     ┌────────────┐     + ◄── Residual connection               │
  │     │            │              │                              │
  │     └────────────┘              │                              │
  │             │                   │                              │
  │             ▼                   │                              │
  │     ┌────────────────┐          │                              │
  │     │  RMSNorm       │          │                              │
  │     └───────┬────────┘          │                              │
  │             │                   │                              │
  │             ▼                   │                              │
  │     ┌────────────────┐          │                              │
  │     │     MLP        │          │                              │
  │     │                │          │                              │
  │     │ gate = σ(Wg*x) │          │                              │
  │     │ up = Wu*x      │          │                              │
  │     │ out = Wd(gate*up)         │                              │
  │     └───────┬────────┘          │                              │
  │             │                   │                              │
  │             ├───────────────────┘                              │
  │             ▼                                                  │
  │     + ◄── Residual connection                                  │
  │             │                                                  │
  │             ▼                                                  │
  │  Output: hidden_states                                         │
  │                                                                │
  └────────────────────────────────────────────────────────────────┘
```

  Weight Packing for Efficiency

```python
  # qwen3.py:186-192 - Maps HuggingFace weight names to fused operations
  packed_modules_mapping = {
      "q_proj": ("qkv_proj", "q"),    # q_proj → qkv_proj, shard "q"
      "k_proj": ("qkv_proj", "k"),    # k_proj → qkv_proj, shard "k"
      "v_proj": ("qkv_proj", "v"),    # v_proj → qkv_proj, shard "v"
      "gate_proj": ("gate_up_proj", 0),  # gate → gate_up, index 0
      "up_proj": ("gate_up_proj", 1),    # up → gate_up, index 1
  }
```

  This means instead of 3 separate matrix multiplications for Q, K, V, we do one larger multiply and split the result. Same for gate/up projections.

  ---
## Part 7: Attention with KV Cache

  File: nanovllm/layers/attention.py:10-75

  Analogy: A Court Stenographer

  Attention with KV cache is like a court stenographer:
  - Prefill: Record the entire testimony (process all prompt tokens)
  - Decode: For each new question, reference the existing transcript (KV cache) plus the new question

```
  ┌────────────────────────────────────────────────────────────────┐
  │                 ATTENTION WITH KV CACHE                        │
  ├────────────────────────────────────────────────────────────────┤
  │                                                                │
  │  PREFILL PHASE:                                                │
  │  ═══════════════                                               │
  │  Tokens: [T1, T2, T3, T4, T5]                                  │
  │                                                                │
  │  Q: ████████████████████  (all tokens)                         │
  │  K: ████████████████████  (all tokens) ──┐                     │
  │  V: ████████████████████  (all tokens) ──┼─► Store in cache    │
  │                                          │                     │
  │  Attention: Full causal matrix           │                     │
  │  ┌─────────────┐                         │                     │
  │  │ Q1 Q2 Q3 Q4 Q5                        │                     │
  │  │ ──────────────                        │                     │
  │  │ █ . . . .  K1                         │                     │
  │  │ █ █ . . .  K2                         │                     │
  │  │ █ █ █ . .  K3                         │                     │
  │  │ █ █ █ █ .  K4                         │                     │
  │  │ █ █ █ █ █  K5                         │                     │
  │  └─────────────┘                         │                     │
  │                                                                │
  │  DECODE PHASE:                                                 │
  │  ═════════════                                                 │
  │  New token: [T6]                                               │
  │                                                                │
  │  Q: █  (just T6)                                               │
  │  K: ████████████████████ + █  ◄── From cache + new             │
  │  V: ████████████████████ + █                                   │
  │                                                                │
  │  Attention: Single row                                         │
  │  ┌─────────────────┐                                           │
  │  │ Q6                                                          │
  │  │ ──────────────────                                          │
  │  │ █ █ █ █ █ █  K1-K6   ◄── Only one query token!              │
  │  └─────────────────┘                                           │
  │                                                                │
  └────────────────────────────────────────────────────────────────┘
```

  Triton Kernel for KV Cache Storage

```python
  # attention.py:10-30 - Custom GPU kernel to store K,V in cache
  @triton.jit
  def store_kvcache_kernel(
      key_ptr, key_stride,
      value_ptr, value_stride,
      k_cache_ptr, v_cache_ptr,
      slot_mapping_ptr,
      D: tl.constexpr,  # num_heads * head_dim
  ):
      idx = tl.program_id(0)  # Token index
      slot = tl.load(slot_mapping_ptr + idx)  # Where to store in cache

      if slot == -1: return  # Skip invalid slots (padding)

      # Load from temporary K,V
      key = tl.load(key_ptr + idx * key_stride + tl.arange(0, D))
      value = tl.load(value_ptr + idx * value_stride + tl.arange(0, D))

      # Store to cache
      cache_offsets = slot * D + tl.arange(0, D)
      tl.store(k_cache_ptr + cache_offsets, key)
      tl.store(v_cache_ptr + cache_offsets, value)
```

  Flash Attention Integration

```python
  # attention.py:59-75 - Forward pass
  def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
      # Store K,V in cache (non-blocking)
      store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping)

      if context.is_prefill:
          # Variable-length sequences packed together
          # Uses prefix caching: if block_tables provided, read K,V from cache
          o = flash_attn_varlen_func(q, k, v,
              cu_seqlens_q=context.cu_seqlens_q,  # Cumulative lengths
              cu_seqlens_k=context.cu_seqlens_k,
              block_table=context.block_tables,   # For prefix cache hits
              causal=True)
      else:
          # Single token per sequence, read all K,V from cache
          o = flash_attn_with_kvcache(q.unsqueeze(1),
              self.k_cache, self.v_cache,
              cache_seqlens=context.context_lens,
              block_table=context.block_tables,
              causal=True)
      return o
```

  Gotcha: The block_tables parameter enables non-contiguous KV cache access. Tokens from different physical blocks can be accessed in the correct logical order, which is essential for prefix caching.

  ---
## Part 8: Tensor Parallelism - Splitting Work

  File: nanovllm/layers/linear.py:1-153

  Analogy: Parallel Assembly Lines

  Tensor parallelism splits the neural network across GPUs like parallel assembly lines:
  - Column Parallel: Each GPU makes part of the output, then combines
  - Row Parallel: Each GPU processes part of the input, then combines

```
  ┌────────────────────────────────────────────────────────────────┐
  │               TENSOR PARALLELISM STRATEGIES                    │
  ├────────────────────────────────────────────────────────────────┤
  │                                                                │
  │  COLUMN PARALLEL (split output dimension):                     │
  │  ════════════════════════════════════════                      │
  │                                                                │
  │  Input X (replicated on all GPUs)                              │
  │       │                                                        │
  │       ├──────────────┬──────────────┐                          │
  │       ▼              ▼              ▼                          │
  │  ┌─────────┐    ┌─────────┐    ┌─────────┐                     │
  │  │  GPU 0  │    │  GPU 1  │    │  GPU 2  │                     │
  │  │ W[:,:n] │    │W[:,n:2n]│    │W[:,2n:] │                     │
  │  │         │    │         │    │         │                     │
  │  │  Y_0    │    │  Y_1    │    │  Y_2    │                     │
  │  └────┬────┘    └────┬────┘    └────┬────┘                     │
  │       │              │              │                          │
  │       └──────────────┼──────────────┘                          │
  │                      ▼                                         │
  │               AllReduce / Concat                               │
  │                      │                                         │
  │                      ▼                                         │
  │                  Y = [Y_0, Y_1, Y_2]                           │
  │                                                                │
  │  ROW PARALLEL (split input dimension):                         │
  │  ══════════════════════════════════════                        │
  │                                                                │
  │  Input X (split across GPUs)                                   │
  │  [X_0]        [X_1]        [X_2]                               │
  │    │            │            │                                 │
  │    ▼            ▼            ▼                                 │
  │  ┌─────────┐ ┌─────────┐ ┌─────────┐                           │
  │  │  GPU 0  │ │  GPU 1  │ │  GPU 2  │                           │
  │  │ W[:n,:] │ │W[n:2n,:]│ │W[2n:,:] │                           │
  │  │         │ │         │ │         │                           │
  │  │  Y_0    │ │  Y_1    │ │  Y_2    │                           │
  │  └────┬────┘ └────┬────┘ └────┬────┘                           │
  │       │            │            │                              │
  │       └────────────┼────────────┘                              │
  │                    ▼                                           │
  │               AllReduce                                        │
  │                    │                                           │
  │                    ▼                                           │
  │              Y = Y_0 + Y_1 + Y_2                               │
  │                                                                │
  └────────────────────────────────────────────────────────────────┘
```

  QKV Parallel Linear (Fused Q, K, V Projections)

```python
  # linear.py:96-128 - Special handling for Q, K, V weights
  class QKVParallelLinear(ColumnParallelLinear):
      def __init__(self, hidden_size, head_size, total_num_heads, total_num_kv_heads):
          # Output size = Q + K + V sizes
          output_size = (total_num_heads + 2 * total_num_kv_heads) * head_size
          super().__init__(hidden_size, output_size)

      def weight_loader(self, param, loaded_weight, loaded_shard_id: str):
          # loaded_shard_id is "q", "k", or "v"
          # Calculate offset into the fused weight matrix
          if loaded_shard_id == "q":
              shard_offset = 0
              shard_size = self.num_heads * self.head_size
          elif loaded_shard_id == "k":
              shard_offset = self.num_heads * self.head_size
              shard_size = self.num_kv_heads * self.head_size
          else:  # "v"
              shard_offset = (self.num_heads + self.num_kv_heads) * self.head_size
              shard_size = self.num_kv_heads * self.head_size

          # Load the shard into the correct position
          param.narrow(0, shard_offset, shard_size).copy_(loaded_weight)
```

  Gotcha: The number of KV heads can be different from Q heads (Group Query Attention / GQA). In Qwen3, this reduces memory while maintaining quality.

  ---
## Part 9: Rotary Position Embeddings (RoPE)

  File: nanovllm/layers/rotary_embedding.py:6-61

  Analogy: A Rotating Clock Hand

  RoPE encodes position by rotating vector components like clock hands:
  - Position 0: No rotation
  - Position 1: Small rotation
  - Position 100: Large rotation

```
  ┌────────────────────────────────────────────────────────────────┐
  │                  ROTARY EMBEDDINGS (RoPE)                      │
  ├────────────────────────────────────────────────────────────────┤
  │                                                                │
  │  Each dimension pair rotates at different frequencies:         │
  │                                                                │
  │  Dim 0-1: Fast rotation (high frequency)                       │
  │  ┌─────────────────────────────────────────────────────────┐   │
  │  │    Pos 0      Pos 1      Pos 2      Pos 3               │   │
  │  │      │          │          │          │                 │   │
  │  │     ──►       ⬈          ⬆          ⬉                   │   │
  │  │      0°       45°        90°       135°                 │   │
  │  └─────────────────────────────────────────────────────────┘   │
  │                                                                │
  │  Dim 2-3: Medium rotation                                      │
  │  ┌─────────────────────────────────────────────────────────┐   │
  │  │    Pos 0      Pos 1      Pos 2      Pos 3               │   │
  │  │      │          │          │          │                 │   │
  │  │     ──►       ⬈          ⬈          ⬆                   │   │
  │  │      0°       22°        45°        67°                 │   │
  │  └─────────────────────────────────────────────────────────┘   │
  │                                                                │
  │  Dim 126-127: Slow rotation (low frequency)                    │
  │  ┌─────────────────────────────────────────────────────────┐   │
  │  │    Pos 0      Pos 1      Pos 2      Pos 3               │   │
  │  │      │          │          │          │                 │   │
  │  │     ──►       ──►        ──►        ⬈                   │   │
  │  │      0°       0.1°       0.2°       0.3°                │   │
  │  └─────────────────────────────────────────────────────────┘   │
  │                                                                │
  │  WHY IT WORKS:                                                 │
  │  When computing attention between positions i and j:           │
  │  Q_i · K_j depends on (θ_i - θ_j) = relative position!         │
  │                                                                │
  └────────────────────────────────────────────────────────────────┘
```

  Implementation

```python
  # rotary_embedding.py:17-35 - Pre-compute rotation matrices
  class RotaryEmbedding(nn.Module):
      def __init__(self, head_size, rotary_dim, max_position_embeddings, base):
          # Inverse frequencies for each dimension pair
          # Lower dimensions rotate faster (higher frequency)
          inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2) / rotary_dim))

          # Compute angles for all positions
          t = torch.arange(max_position_embeddings)
          freqs = torch.einsum("i,j -> ij", t, inv_freq)  # Outer product

          # Cache cos and sin values
          cos = freqs.cos()
          sin = freqs.sin()
          self.register_buffer("cos_sin_cache", torch.cat((cos, sin), dim=-1))

  # rotary_embedding.py:6-14 - Apply rotation
  def apply_rotary_emb(x, cos, sin):
      x1, x2 = torch.chunk(x, 2, dim=-1)  # Split into pairs
      y1 = x1 * cos - x2 * sin  # Rotation formula
      y2 = x2 * cos + x1 * sin
      return torch.cat((y1, y2), dim=-1)
```

  Gotcha: The base (default 10000) controls the "wavelength" of position encoding. Larger bases allow longer context lengths. Some models use rope_scaling to extend beyond training length, but nano-vllm currently disables this (line 59 comment).

  ---
## Part 10: The Sampler - Token Selection

  File: nanovllm/layers/sampler.py:5-15

  Analogy: A Weighted Lottery

  The sampler is like a lottery where:
  - Each token has a ticket proportional to its probability
  - Temperature controls how "fair" the lottery is (higher = more random)

```
  ┌────────────────────────────────────────────────────────────────┐
  │                     TEMPERATURE SAMPLING                       │
  ├────────────────────────────────────────────────────────────────┤
  │                                                                │
  │  Temperature = 0 (Greedy):                                     │
  │  ┌─────────────────────────────────────────────────────┐       │
  │  │  "the" │  "a"  │"cat"│ "dog"│  ...                  │       │
  │  │  ████  │  ██   │  █  │  █   │                       │       │
  │  │  99%   │  0.5% │ 0.3%│ 0.2% │                       │       │
  │  └─────────────────────────────────────────────────────┘       │
  │  → Always picks "the"                                          │
  │                                                                │
  │  Temperature = 1.0:                                            │
  │  ┌─────────────────────────────────────────────────────┐       │
  │  │  "the" │  "a"  │"cat"│ "dog"│  ...                  │       │
  │  │  ████  │  ███  │  ██ │  █   │                       │       │
  │  │  40%   │  30%  │ 20% │ 10%  │                       │       │
  │  └─────────────────────────────────────────────────────┘       │
  │  → Usually "the", sometimes others                             │
  │                                                                │
  │  Temperature = 2.0:                                            │
  │  ┌─────────────────────────────────────────────────────┐       │
  │  │  "the" │  "a"  │"cat"│ "dog"│  ...                  │       │
  │  │  ███   │  ███  │  ██ │  ██  │                       │       │
  │  │  28%   │  26%  │ 24% │ 22%  │                       │       │
  │  └─────────────────────────────────────────────────────┘       │
  │  → Nearly uniform, very creative/random                        │
  │                                                                │
  └────────────────────────────────────────────────────────────────┘
```

  Gumbel-Max Trick

```python
  # sampler.py:10-15 - Efficient sampling without explicit multinomial
  @torch.compile
  def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
      # Scale logits by temperature
      logits = logits.float().div_(temperatures.unsqueeze(dim=1))

      # Convert to probabilities
      probs = torch.softmax(logits, dim=-1)

      # GUMBEL-MAX TRICK: Instead of multinomial sampling,
      # divide by exponential noise and take argmax
      # Equivalent to: torch.multinomial(probs, 1)
      # But faster because argmax is cheaper than multinomial
      sample_tokens = probs.div_(
          torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
      ).argmax(dim=-1)

      return sample_tokens
```

  Gotcha: The Gumbel-Max trick (p / Exp(1) then argmax) is mathematically equivalent to multinomial sampling but much faster on GPU. The clamp_min_(1e-10) prevents division by zero.

  ---
## Part 11: RMSNorm - Efficient Normalization

  File: nanovllm/layers/layernorm.py:5-50

  Analogy: Adjusting Photo Brightness

  RMSNorm is like auto-adjusting photo brightness based on the average pixel intensity, but simpler than full histogram equalization (LayerNorm).

```
  ┌────────────────────────────────────────────────────────────────┐
  │              RMSNORM vs LAYERNORM                              │
  ├────────────────────────────────────────────────────────────────┤
  │                                                                │
  │  LayerNorm:                                                    │
  │  ┌────────────────────────────────────────────┐                │
  │  │  x_norm = (x - mean(x)) / sqrt(var(x) + ε) │                │
  │  │  output = x_norm * γ + β                   │                │
  │  └────────────────────────────────────────────┘                │
  │  → Requires mean AND variance, plus bias term                  │
  │                                                                │
  │  RMSNorm (Root Mean Square):                                   │
  │  ┌────────────────────────────────────────────┐                │
  │  │  rms = sqrt(mean(x²) + ε)                  │                │
  │  │  output = (x / rms) * γ                    │                │
  │  └────────────────────────────────────────────┘                │
  │  → Only needs mean of squares, no bias                         │
  │  → ~25% faster than LayerNorm                                  │
  │                                                                │
  └────────────────────────────────────────────────────────────────┘
```

  Fused Add + RMSNorm

```python
  # layernorm.py:28-40 - Combine residual addition with normalization
  @torch.compile
  def add_rms_forward(self, x: torch.Tensor, residual: torch.Tensor):
      # Fuse: residual = residual + x, then normalize x
      # This saves one memory round-trip
      x = x.float().add_(residual.float())  # In-place add
      residual = x.to(orig_dtype)  # Save for next layer
      var = x.pow(2).mean(dim=-1, keepdim=True)
      x.mul_(torch.rsqrt(var + self.eps))  # In-place normalize
      x = x.to(orig_dtype).mul_(self.weight)
      return x, residual
```

  Gotcha: The function returns TWO values: the normalized output AND the updated residual. This is because the decoder layer uses a "residual stream" pattern where we need to carry the residual forward.

  ---
## Part 12: Context Management

  File: nanovllm/utils/context.py:1-27

  Analogy: Shared Clipboard

  The Context is like a shared clipboard that all attention layers can read from. Instead of passing parameters through many function calls, we set them once and everyone reads them.

```python
  # context.py:5-14 - The context dataclass
  @dataclass
  class Context:
      is_prefill: bool = False           # Prefill or decode phase?
      cu_seqlens_q: Tensor = None        # Cumulative query lengths [0, 10, 25, 40]
      cu_seqlens_k: Tensor = None        # Cumulative key lengths
      max_seqlen_q: int = 0              # Longest query sequence
      max_seqlen_k: int = 0              # Longest key sequence
      slot_mapping: Tensor = None        # Token → cache slot mapping
      context_lens: Tensor = None        # KV cache lengths per sequence
      block_tables: Tensor = None        # Block IDs per sequence

  # Global singleton pattern
  _CONTEXT = Context()

  def set_context(...):
      global _CONTEXT
      _CONTEXT = Context(...)

  def get_context():
      return _CONTEXT
```

  Gotcha: This global state pattern works because inference is single-threaded. In multi-threaded scenarios, you'd need thread-local storage.

  ---
## Part 13: Model Weight Loading

  File: nanovllm/utils/loader.py:1-28

  Analogy: Furniture Assembly with Custom Instructions

  Loading weights is like assembling furniture where some parts need special handling (fused/packed weights) and others just slot in directly.

```python
  # loader.py:12-28 - Load model with packed weight support
  def load_model(model: nn.Module, path: str):
      # Map of original names to fused names
      packed_modules_mapping = model.packed_modules_mapping

      for file in glob(os.path.join(path, "*.safetensors")):
          with safe_open(file, "pt", "cpu") as f:
              for weight_name in f.keys():
                  # Check if this weight should be packed
                  for original, (fused_name, shard_id) in packed_modules_mapping.items():
                      if original in weight_name:
                          # Load into fused weight with shard ID
                          param = model.get_parameter(weight_name.replace(original, fused_name))
                          param.weight_loader(param, f.get_tensor(weight_name), shard_id)
                          break
                  else:
                      # Default loading - just copy
                      param = model.get_parameter(weight_name)
                      param.weight_loader(param, f.get_tensor(weight_name))
```

  Gotcha: Each parameter has a custom weight_loader method (set in linear.py). This allows tensor-parallel layers to shard weights correctly during loading.

  ---
## Summary: The Complete Data Flow

```
  ┌────────────────────────────────────────────────────────────────┐
  │                   COMPLETE INFERENCE FLOW                      │
  ├────────────────────────────────────────────────────────────────┤
  │                                                                │
  │  1. User calls LLM.generate(["Hello world"])                   │
  │                      │                                         │
  │                      ▼                                         │
  │  2. Tokenize: [101, 7592, 2088]                                │
  │                      │                                         │
  │                      ▼                                         │
  │  3. Create Sequence object, add to scheduler.waiting           │
  │                      │                                         │
  │                      ▼                                         │
  │  4. MAIN LOOP:                                                 │
  │     ┌────────────────────────────────────────────────────┐     │
  │     │ scheduler.schedule()                               │     │
  │     │   → Check waiting queue (prefill) or running (decode)    │
  │     │   → Allocate KV cache blocks via BlockManager      │     │
  │     │   → Return batch of sequences                      │     │
  │     └────────────────────────────────────────────────────┘     │
  │                      │                                         │
  │                      ▼                                         │
  │     ┌────────────────────────────────────────────────────┐     │
  │     │ model_runner.run(seqs, is_prefill)                 │     │
  │     │   → Prepare input tensors (positions, slot_mapping)│     │
  │     │   → Set context for attention layers               │     │
  │     │   → Run model (CUDA graph or eager)                │     │
  │     │   → Sample next token                              │     │
  │     └────────────────────────────────────────────────────┘     │
  │                      │                                         │
  │                      ▼                                         │
  │     ┌────────────────────────────────────────────────────┐     │
  │     │ MODEL FORWARD PASS:                                │     │
  │     │                                                    │     │
  │     │ tokens → Embedding → hidden                        │     │
  │     │                                                    │     │
  │     │ for each layer:                                    │     │
  │     │   hidden = RMSNorm(hidden)                         │     │
  │     │   Q,K,V = QKVProj(hidden)                          │     │
  │     │   Q,K = RoPE(Q,K, positions)                       │     │
  │     │   Store K,V in cache                               │     │
  │     │   hidden = Attention(Q,K,V) + residual             │     │
  │     │   hidden = RMSNorm(hidden)                         │     │
  │     │   hidden = MLP(hidden) + residual                  │     │
  │     │                                                    │     │
  │     │ logits = LMHead(hidden[-1])  # Last token only     │     │
  │     │ next_token = Sample(logits, temperature)           │     │
  │     └────────────────────────────────────────────────────┘     │
  │                      │                                         │
  │                      ▼                                         │
  │     ┌────────────────────────────────────────────────────┐     │
  │     │ scheduler.postprocess(seqs, token_ids)             │     │
  │     │   → Append new token to each sequence              │     │
  │     │   → Check for EOS or max_tokens                    │     │
  │     │   → Mark finished sequences                        │     │
  │     │   → Deallocate KV cache for finished               │     │
  │     └────────────────────────────────────────────────────┘     │
  │                      │                                         │
  │                      ▼                                         │
  │  5. Repeat until all sequences finished                        │
  │                      │                                         │
  │                      ▼                                         │
  │  6. Detokenize and return text                                 │
  │                                                                │
  └────────────────────────────────────────────────────────────────┘
```

  ---
  Key Gotchas Summary
  ┌──────────────┬───────────────────────────────────────────────────────────────┐
  │  Component   │                            Gotcha                             │
  ├──────────────┼───────────────────────────────────────────────────────────────┤
  │ Scheduler    │ else on while executes only if loop didn't break              │
  ├──────────────┼───────────────────────────────────────────────────────────────┤
  │ Sequence     │ Custom pickling excludes tokens after prefill to save memory  │
  ├──────────────┼───────────────────────────────────────────────────────────────┤
  │ BlockManager │ Hash chaining prevents false prefix cache matches             │
  ├──────────────┼───────────────────────────────────────────────────────────────┤
  │ ModelRunner  │ CUDA graphs captured in reverse order for memory sharing      │
  ├──────────────┼───────────────────────────────────────────────────────────────┤
  │ Attention    │ slot_mapping=-1 signals padding (skip in Triton kernel)       │
  ├──────────────┼───────────────────────────────────────────────────────────────┤
  │ Linear       │ Each weight has custom weight_loader for TP sharding          │
  ├──────────────┼───────────────────────────────────────────────────────────────┤
  │ Sampler      │ Gumbel-Max trick is equivalent to but faster than multinomial │
  ├──────────────┼───────────────────────────────────────────────────────────────┤
  │ RMSNorm      │ Returns both normalized output AND residual                   │
  ├──────────────┼───────────────────────────────────────────────────────────────┤
  │ Context      │ Global state works because inference is single-threaded       │
  └──────────────┴───────────────────────────────────────────────────────────────┘

  ---
  This is nano-vllm - approximately 1,200 lines of Python that achieves production-level LLM inference performance through careful engineering of batching, caching, and GPU optimization. Would you like me to dive deeper into any specific component?

✻ Cogitated for 6m 17s
