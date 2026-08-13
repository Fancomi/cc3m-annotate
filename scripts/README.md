# scripts/ —— 人工裁决工具

从 `legacy/stage3_verify/` 适配到当前 pipeline 的版本（输入字段、文档路径已更新，逻辑不变）。
用途：**用人工复核校准阶段4 的自动精度下界**（gemma4 判 gemma4 有确认偏误，只能算下界）。

## 数据流

```
out/verify_clean.jsonl（阶段4 产出，自带 path 主键）
        │  make_adjudicate_html.py --verify ... --out adjudicate_clean.html
        ▼
adjudicate_clean.html（分层抽样 YES/NO 各 100，人工逐题判 Y/N/U）
        │  人工答题；serve_review.py 提供 POST /save 落盘到开发机磁盘
        ▼
out/human_adjudication_clean.json（每答一题全量写盘，可跨端口/浏览器续答）
        │  apply_human_adj.py --json ... --dir <repo>
        ▼
docs/RESULT.md 追加「第 6 节 人工裁决校准」
```

## 用法（在开发机上）

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/vision_encoder/cc3m-annotate

# 1. 生成裁决页面（分层抽样：YES/NO 各 100，seed 可改）
python3 scripts/make_adjudicate_html.py \
  --verify out/verify_clean.jsonl \
  --out adjudicate_clean.html \
  --per-stratum 100 --seed 11 \
  [--auto-precision 80.3]   # 可选：RESULT.md 里的自动下界（%），默认取 verify 的 YES 比例

# 2. 起服务（带落盘，替代 python -m http.server）
python3 scripts/serve_review.py --dir . --port 8899 \
  --save out/human_adjudication_clean.json \
  [--allowedip 你的IP]      # 若被 403 用这个追加白名单

# 3. 本地浏览器打开 http://<开发机>:8899/adjudicate_clean.html 逐题作答
#    （页面启动自动 GET /load 恢复已存裁决；键盘 Y/N/U，← 回退）

# 4. 答完写回统计（追加到 docs/RESULT.md 第 6 节）
python3 scripts/apply_human_adj.py --json out/human_adjudication_clean.json \
  --dir . --auto-precision 80.3
```

## 关键设计（别改坏）

- **分层抽样不可换成简单随机**：NO 层只占总体 ~25%，随机抽样本里 NO 太少，估不准「判定器误否率」。两层的 a/b 按总体权重还原：`真实精度 = (N_YES·a + N_NO·b)/(N_YES+N_NO)`。
- **页面刻意不显示自动裁决**，避免锚定人工判断；判完每题可展开对照。
- **UNSURE 不计入统计**。
- 落盘是「先写 tmp 再 rename」，每 20 条留带计数快照（`.0020`、`.0040`…），防误点重置。

## 与 legacy 版的差异

| 项 | legacy/stage3_verify | scripts/（本目录） |
|---|---|---|
| 输入 | `out/verify_crossfeed.jsonl` + `crossfeed_shard*.jsonl`（id 主键） | `--verify out/verify_clean.jsonl`（path 主键，自带图路径） |
| 判定模型 | Qwen | gemma4 |
| 写回目标 | `ANALYSIS.md` | `docs/RESULT.md` |
| 自动下界 | 硬编码 74.5% | `--auto-precision`，默认取 verify 的 YES 比例 |

`serve_review.py` 与 legacy 版一致（无需改动）。`make_review_html.py` / `verify_crossfeed.py` 属于旧链路，未迁移。
