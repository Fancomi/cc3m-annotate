# cc3m-annotate

用 VLM + Florence-2 为 [CC3M](https://ai.google.com/research/ConceptualCaptions/) 全量 289 万张图片自动生成**区域级短语标注**（短语 + 边界框），无需人工标注、无需预设类别表。

```
图片 ──gemma4──▶ dense caption ──Florence-2──▶ {短语: [框]} ──清洗──▶ 训练数据
        「有什么」          178 词        「在哪」      30 框/图     去噪
```

## 为什么这样设计

单个模型做不好这件事，所以拆成三方各司其职：

| 谁 | 干什么 | 为什么不能换 |
|---|---|---|
| **gemma4** | 看图写出所有可见物体的名字 | 决定了后续能定位哪些短语。对照实验中它比 Qwen3.6 的 caption 驱动出的 grounding 精度高 5.3 个点（71.4% vs 66.1%），且更少脑补 |
| **Florence-2** | 把 caption 里的短语逐个定位成框 | 它是纯定位器：caption 没写的物体不会被 ground 出来（实测 88.5% 的短语可逐字回溯到 caption）。检测头方案（YOLOE 等）受词表上限约束，开放词汇分类器（NOVIC 等）不出框 |
| **gemma4**（再次） | 裁出每个框，独立判断短语是否真在里面 | 只看裁剪图、不看 caption，依据与产出模型不同，所以能抓到「框位错」和「概念不存在」两类错误 |

**关键认知**：产出的丰富程度由 caption 决定，不由 Florence-2 决定。同一张图，10 词 caption 只 ground 出 3 个框，178 词 dense caption 出 30 个。想提升产出质量，改 prompt 或换 caption 模型，而不是调 Florence-2。

## 快速开始

```bash
bash install.sh                  # 建环境、下权重、自检（幂等）
bash run/run_all.sh --smoke      # 冒烟：约 1150 张，20 分钟，验证链路
bash run/run_all.sh              # 全量：289 万张，约 60 小时
```

单独跑某个阶段：

```bash
bash run/1_caption.sh            # 阶段1  caption            约 30 小时
bash run/1b_compact.sh           # 阶段1b 压实 + 完整性检查    约 3 分钟
bash run/sgl.sh down             # 释放显存给 Florence-2
bash run/2_grounding.sh          # 阶段2  grounding          约 29 小时
bash run/3_clean.sh              # 阶段3  清洗               约 20 分钟
bash run/4_verify.sh             # 阶段4  抽样校验 + 报告     约 10 分钟
```

所有阶段都断点续传，且**只跳过成功条目** —— 失败的会在重跑时自动重试。中断后重跑同一条命令即可。

## 产出

```
out/
├── caption/shard{0..7}.jsonl        {id, shard, path, gt_caption, gemma_short, gemma_dense}
├── ground/ground_shard{0..7}.jsonl  {id, shard, path, img_wh, grounding, n_phrase, n_box}
├── clean/clean_shard{0..7}.jsonl    同上，grounding 已过滤
└── verify_clean.jsonl               {path, phrase, box, crop_frac, verdict}
docs/RESULT.md                       统计报告（由阶段4 自动生成）
```

`grounding` 形如 `{"a black dog": [[12.4, 88.1, 402.7, 511.0]], "dorsal fin": [...]}`，坐标是原图像素。

**主键是 `path`**。`id` 只是图片在所属 tsv 内的行号，576 个 tsv 之间会撞号；`(shard, id)` 才唯一，而 `path` 天然全局唯一，所以去重、对齐、断点续传统一用 `path`。

## 性能

8×H800 实测：

| 阶段 | 吞吐 | 全量耗时 | 瓶颈 |
|---|---|---|---|
| caption | ~80k 图/时 | ~30 h | gemma4 decode |
| grounding | ~100k 图/时 | ~29 h | Florence-2 decode |
| 清洗 | — | ~20 min | 单进程 CPU |

grounding 阶段用了三个优化叠加，合计 **3.85× 加速**（3150ms → 817ms/图），细节见 `src/s2_grounding.py` 的模块注释：batch 推理 + 禁 EOS 保精度 + 长度分桶。

## 目录

```
install.sh          环境一键配置
run/                入口脚本，按阶段编号；改路径只需改 run/env.sh
src/                实现
  common.py         图像编码、VLM 客户端（含重试）、jsonl 读写、分片、断点续传、共用正则
  batch.py          并发/串行批处理骨架（分片 + 续传 + 进度打点）
  s1_caption.py     阶段1  gemma4 两级 caption
  compact.py        阶段1b 分片去重收口（append 续传产生的重复行）
  s2_grounding.py   阶段2  Florence-2 batch grounding
  s3_clean.py       阶段3  短语清洗
  s4_verify.py      阶段4  抽样校验
  report.py         统计报告生成
docs/
  INSTALL.md        环境细节、版本约束原因、常见故障
  PIPELINE.md       各阶段输入输出与参数含义
  DECISIONS.md      选型对照数据与被淘汰的方案
legacy/             早期三方对照实验（Florence-2 三级caption / NOVIC / Qwen），仅供复现
```

**新接手这个项目**：先读本文件 → `docs/PIPELINE.md` → 要动哪个阶段就只看对应的 `src/sN_*.py`。各阶段之间只通过 jsonl 文件耦合，改一个不影响其他。`legacy/` 是已淘汰路线，不要在上面改代码。

## 已知限制

- **校验精度是下界不是真值**。判定器与 caption 出自同一模型族，存在确认偏误。可靠的是「同判定器、同样本」下的相对比较，不是绝对数值。
- **CC3M 原 caption 不能当真值**。它是网页 alt-text，含大量视觉不可见信息（人名、事件、地点），所以「与原标注的词重合率」衡量的是文本相似度，不是正确性。
- **约 4.6% 短语是解码漂移产物**（拼写偏移、词组截断），阶段3 的 `garbled` 规则按「能否在源 caption 中逐字找到」过滤。
- **grounding 长尾被截断**。禁 EOS 后 batch 内统一生成 420 token，覆盖约 92% 的图；更长的图会丢尾部短语。要全覆盖需 `--max-new-tokens 1024`，代价是吞吐降约一半。
