# 流程说明

每个阶段读上游的 jsonl、写自己的 jsonl，阶段之间只通过文件耦合。要改某个阶段，只看它对应的一个 `src/sN_*.py` 即可。

```
cc3m-tsv/_shards/*.tsv
        │  path + 原始 caption
        ▼
[1] s1_caption.py       gemma4 两级 caption          out/caption/shard{N}.jsonl
        │                                                   │ 约 8% 请求失败
        │                          [1b] s1b_caption_retry.py│ scan → run → merge（原地补齐）
        ▼
[2] s2_grounding.py     Florence-2 短语定位          out/ground/ground_shard{N}.jsonl
        ▼
[3] s3_clean.py         过滤噪声短语与低质框          out/clean/clean_shard{N}.jsonl
        ▼
[4] s4_verify.py        抽样校验 → report.py         out/verify_clean.jsonl + docs/RESULT.md
```

## 阶段 1 · caption

`run/1_caption.sh` → `src/s1_caption.py`，8 进程各 16 并发，轮询 8 个 gemma4 端点。

两级 prompt（`src/common.py`）：

- `SHORT` 一句概括，产出中位 20 词。
- `DENSE` **要求逐个点名具体名词**，产出中位 178 词。这是整条链路的源头 —— grounding 只能定位 caption 里出现过的短语，所以 prompt 刻意反常识地要求"穷举名词"而不是"写通顺"。

| 参数 | 默认 | 说明 |
|---|---|---|
| `--concurrency` | 16 | 每进程并发请求；调高吃满 GPU，过高会触发端点排队 |
| `--limit-per-tsv` | 0 | >0 时每个 tsv 只取前 N 行，用于冒烟 |

输出字段：`{id, shard, path, gt_caption, gemma_short, gemma_dense, dt_s}`。失败条目带 `error`，缺 `gemma_dense`。

## 阶段 1b · 补齐

sglang 实例在长时批处理中偶发 watchdog 重启，期间的请求报 `APIConnectionError`。全量实测约 24 万条（8%）。

```bash
bash run/1b_retry.sh scan     # 全扫 3.7G，提取缺 dense 的条目 → retry_worklist.jsonl
bash run/1b_retry.sh run      # 8 进程并行补跑 → retry_shard{N}.jsonl
bash run/1b_retry.sh merge    # 成功项按 path 覆盖回原 shard
bash run/1b_retry.sh          # 以上三步 + 等待，一条命令搞定
```

`merge` 原地改写 `shard*.jsonl`（先写 `.tmp` 再 `rename`，中断不会留半截文件），保持原行序，所以补齐后每行仍与原图一一对应。

## 阶段 2 · grounding

`run/2_grounding.sh` → `src/s2_grounding.py`，8 进程各占一张卡。

三个优化叠加得到 3.85× 加速，任一条破坏都会掉精度或掉速度：

1. **batch=8**。F2 的 decode 是 memory-bound，batch 增大能把算力吃满。
2. **禁 EOS**（`eos_token_id=-1`）。generate 在 batch 下某序列先到 EOS 会被剔除，batch 维度 N→N-1 改变其余序列的 attention 上下文，输出漂移。禁 EOS 让全 batch 同步生成到 `max_new_tokens`，实测与单图逐 token 一致。
3. **首个 `</s>` 处截断再解析**。禁 EOS 的代价：模型在自然结束点吐 `</s>`，之后是重复碎片（`- -`、`materialy`）。不截断的话垃圾短语占 32%，截断后降到 4.6% 且与自然 EOS 结果完全一致。
4. **长度分桶**。F2 的 SDPA 实现不支持 padding（会形状冲突），所以同 batch 必须等长 —— 按 caption 词数排序后相邻成组，再截到组内最短，截断损失最小。

| 参数 | 默认 | 说明 |
|---|---|---|
| `--batch` | 8 | 实测吞吐拐点；再大收益递减且延迟明显上升 |
| `--max-new-tokens` | 420 | 自然长度实测 p50=199 / p90=360 / p99≈965。420 覆盖约 92%，长尾图会丢尾部短语；要全覆盖设 1024，吞吐降约一半 |

输出字段：`{id, shard, path, img_wh, grounding, n_phrase, n_box}`，`grounding` 为 `{短语: [[x0,y0,x1,y1], ...]}`，坐标是原图像素。带 `img_wh` 是为了让下游算框面积占比时不必重新读图。

batch 内任一图异常（坏图、显存抖动）会退化为逐图重试，不会整组丢失。

## 阶段 3 · 清洗

`run/3_clean.sh` → `src/s3_clean.py`，纯 CPU 单进程。五条规则各自可关：

| 规则 | 过滤什么 | 关闭参数 |
|---|---|---|
| `vague` | 整体指代类短语（`the entire image`），ground 到全图无区域价值 | `--no-vague` |
| `garbled` | 归一化后无法在源 caption 中找到 —— 解码漂移产物 | `--no-garbled` |
| `words` | 实词数不足（虚词、标点） | `--min-words 0` |
| `dup` | 同短语下 IoU>0.9 的重复框 | `--no-dup` |
| `area` | 框面积占比过小 | 默认关，`--min-area 0.02` 开 |

默认只开前四条（无损清洗，实测保留约 93%）。

**要做训练数据**用 `TRAIN=1 bash run/3_clean.sh`，等价于额外加 `--min-area 0.02 --min-words 2`。该档实测精度 80.3%、保留 66%，折合约 16 个有效短语/图。各档权衡见 `docs/RESULT.md` 的「过滤规则权衡」表。

## 阶段 4 · 校验 + 报告

`run/4_verify.sh` → `src/s4_verify.py` → `src/report.py`。

把每个框外扩 12% 裁出来，问 gemma4「这个裁剪里能清楚看到 `<短语>` 吗」。判定器只看裁剪图、不看 caption，依据与产出模型不同，所以能同时抓到「框位错」和「概念根本不存在」两类错误。

| 变量 | 默认 | 说明 |
|---|---|---|
| `SAMPLE` | 400 | 抽多少张图 |
| `IN` | clean | `clean` 校验清洗后，`ground` 校验清洗前（做对照用） |

**结果读作精度下界，不是真值** —— 判定器与 caption 出自同一模型族，存在确认偏误。可比的是「同判定器、同样本、同 prompt」下的相对差异。

`report.py` 汇总各阶段产出规模、短语密度、精度随框面积/短语长度的变化、过滤规则权衡，写到 `docs/RESULT.md`。
