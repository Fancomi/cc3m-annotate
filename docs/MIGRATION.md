# cc3m-annotate · 迁移与交接文档

> 适用：从当前机器迁移到**另一台完全同配置、同目录结构**的开发机。
> 代码走 git，数据手动拉取。全部命令在**新机器**上执行。

## 迁移状态

**10.52.101.139 → 10.52.101.140 已完成**（数据、rec 档全量重跑、全量校验、审核页面）。
详见第 9 节。本文件第 1~5 节保留为「下次换机怎么做」的操作手册。

## 0. 迁移前：本机最后的 git 提交

在旧机器（本机）执行（已完成）：

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/vision_encoder/cc3m-annotate
git add -A && git commit -m "清洗规则演进：词边界+abstract+xdup+消融，推荐档取代面积过滤"
git push origin main
```

## 1. 代码（git 拉取）

```bash
# 新机器，目录结构必须与旧机器完全一致
mkdir -p /root/paddlejob/workspace/env_run/penghaotian/vision_encoder
cd /root/paddlejob/workspace/env_run/penghaotian/vision_encoder
git clone https://github.com/Fancomi/cc3m-annotate.git
cd cc3m-annotate
git checkout main
```

## 2. 数据（手动拉取，约 11G）

代码与数据分离：`out/` 被 `.gitignore` 排除，必须用 rsync/scp 单独拉。

```bash
# 在旧机器上打包（可选，减少连接数），或直接 rsync 目录
cd /root/paddlejob/workspace/env_run/penghaotian/vision_encoder/cc3m-annotate

# 方案 A：rsync 直接同步（推荐，断点续传）
rsync -avP --progress out/caption/  <新机器>:/root/paddlejob/workspace/env_run/penghaotian/vision_encoder/cc3m-annotate/out/caption/
rsync -avP --progress out/ground/   <新机器>:.../out/ground/
rsync -avP --progress out/clean/    <新机器>:.../out/clean/
rsync -avP --progress out/ab/       <新机器>:.../out/ab/
rsync -avP --progress out/verify_clean.jsonl out/verify_clean_s400.jsonl <新机器>:.../out/

# 方案 B：tar 打包传输（适合一次性全量）
tar czf cc3m_out.tar.gz out/caption out/ground out/clean out/ab \
  out/verify_clean.jsonl out/verify_clean_s400.jsonl
# 传过去后解压到 cc3m-annotate/out/
```

**务必同步这些**（按优先级）：
| 数据 | 体积 | 为什么必须 | 能否重建 |
|---|---|---|---|
| `out/caption/` | 4.4G | 阶段1 全量产出（289 万图 dense caption） | 重建需 30 小时 + 8×H800 |
| `out/ground/` | 4.5G | 阶段2 全量产出（Florence-2 grounding） | 重建需 29 小时 |
| `out/clean/` | 2.0G | **最终训练数据**（rec 档需全量重跑后覆盖） | 重跑清洗 20 分钟 |
| `out/ab/` | 165M | 消融实验全部档位 + 各档 verify | 重建约 1 小时 |
| `out/verify_clean.jsonl` | 1.8M | 阶段4 抽样校验结果（1000 图） | 重跑 2 分钟 |
| `out/verify_clean_s400.jsonl` | 708K | 400 图版备份 | 可重建 |

**可选**：`logs/`（各阶段日志，方便查历史）、`rule_ablation.html`（3.1M，消融审核页面）。

## 3. 环境依赖（必须与旧机器一致）

代码运行时依赖以下外部资源，**不在仓库里**，需确认新机器已有：

| 依赖 | 路径 | 用途 |
|---|---|---|
| `envs/sglang__0.5.12` | `~/envs/sglang__0.5.12` | 阶段1/4/5：gemma4 sglang + openai 客户端 |
| `envs/dam` | `~/envs/dam` | 阶段2：Florence-2 推理（transformers 4.46.3） |
| `models/Florence-2-large` | `~/models/Florence-2-large` | 阶段2 权重 |
| `models/gemma-4-26B-A4B-it` | `~/models/gemma-4-26B-A4B-it` | 阶段1/4 权重（49G） |
| `/dev/shm/models/gemma-4-26B-A4B-it` | /dev/shm | sglang 用（首次自动拷贝） |
| `datas/cc3m-tsv/_shards/` | `~/datas/cc3m-tsv/_shards` | 576 个 tsv 分片（阶段1 输入） |
| `datas/cc3m-tsv/images/` | `~/datas/cc3m-tsv/images` | 图片（全量约 300G，校验/清洗要读） |

> 若新机器缺 gemma4 权重，跑 `bash install.sh`（幂等，会自动拉权重+建环境）。
> `/dev/shm/models` 在重启后清空，但 sgl.sh 首次启动会自动从 `~/models` 拷贝（约 3 分钟）。

## 4. 新机器上的验证（迁移后必做）

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/vision_encoder/cc3m-annotate

# 1) 数据完整性
wc -l out/caption/shard*.jsonl | tail -1   # 应 ≈ 2,894,191
wc -l out/ground/ground_shard*.jsonl | tail -1  # 应 ≈ 2,894,191
wc -l out/clean/clean_shard*.jsonl | tail -1    # 应 ≈ 2,894,189（旧档；rec 档全量重跑后更新）
head -1 out/verify_clean.jsonl                 # 应有内容

# 2) 依赖可用
bash run/sgl.sh status        # 应显示就绪 0/8（未启动，属正常）
python3 -c "import sys; sys.path.insert(0,'src'); from common import is_abstract; print(is_abstract('the lighting'))"  # True

# 3) 冒烟：小批量清洗 + 校验（不占 GPU 太久）
#    清洗用 python3（CPU），校验需 gemma4（见下）
```

## 5. 下一步工作流（新机器）

### 5.1 全量重跑清洗（rec 档 = 推荐规则）

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/vision_encoder/cc3m-annotate
TRAIN=1 bash run/3_clean.sh
```

rec 档参数（消融结论固化，已写进 `run/3_clean.sh` 的 TRAIN 分支）：
```bash
--min-words 1 --word-boundary --abstract --xdup --max-cover 0.95 --edge
```

### 5.2 全量校验（需要 gemma4）

```bash
# 本机 8001-8008 常驻 8 个 gemma4 sglang 实例（参数与 run/sgl.sh 一致），直接复用：
PORT_BASE=8001 SAMPLE=1000 bash run/4_verify.sh
```

> 别用默认 PORT_BASE=8101：8 卡显存已被那批实例占满（每卡仅剩约 12G），
> `sgl.sh up` 会 OOM。也**别跑 `sgl.sh down` 或 `run_all.sh`** —— 里面的
> `pkill -f sglang.launch_server` 会把那批实例一起杀掉。

### 5.3 gemma 审核（消融审核页面）

```bash
python3 scripts/make_rule_ablation_html.py --n 40 --variants "c rec"   # -> rule_ablation.html
python3 scripts/serve_review.py --dir . --port 8900                    # 8899 已被 /tmp/hdr_capture.py 占用
# 浏览器开 http://10.52.101.140:8900/rule_ablation.html
```

### 5.4 人工裁决（校准精度下界）

```bash
python3 scripts/make_adjudicate_html.py --verify out/verify_clean.jsonl \
  --out adjudicate_clean.html --per-stratum 100
python3 scripts/serve_review.py --dir . --port 8900 --save out/human_adjudication_clean.json
# 浏览器开 http://10.52.101.140:8900/adjudicate_clean.html（答一题即落盘，可换端口/浏览器续答）
# 答完 → python3 scripts/apply_human_adj.py --json out/human_adjudication_clean.json --dir . \
#   --auto-precision 70.1   # 用 RESULT.md 里 rec 档的精度
```

## 6. 目录结构（交接后）

```
cc3m-annotate/
├── install.sh            # 环境一键配置（幂等）
├── README.md             # 项目总览
├── run/                  # 阶段入口脚本（env.sh 集中配置路径）
│   ├── 1_caption.sh      # 阶段1 gemma4 dense caption
│   ├── 1b_compact.sh     # 阶段1b 压实收口
│   ├── 2_grounding.sh    # 阶段2 Florence-2 grounding
│   ├── 3_clean.sh        # 阶段3 清洗（TRAIN=1 为训练档）
│   ├── 4_verify.sh       # 阶段4 抽样校验 + 报告
│   ├── ab_clean.sh       # A/B 清洗对照（多档）
│   ├── ab_verify.sh      # 对照校验
│   ├── auto_verify.sh    # 自动接力守护（等 GPU 释放后跑阶段4）
│   └── sgl.sh            # gemma4 sglang 起停
├── src/                  # 实现
│   ├── s1_caption.py s2_grounding.py s3_clean.py s4_verify.py
│   ├── common.py         # 公共：正则/客户端/工具
│   ├── batch.py          # 并发批处理骨架
│   ├── report.py         # 统计报告
│   ├── ab_report.py      # 消融汇总
├── scripts/              # 工具
│   ├── ablate_rules.py   # 规则消融（一次性跑出全部档位）
│   ├── make_rule_ablation_html.py + rule_ablation_tpl.html   # 消融审核页
│   ├── make_adjudicate_html.py + serve_review.py + apply_human_adj.py  # 人工裁决
│   ├── explain_sampling.py # 复现 7942 抽样数字
├── docs/
│   ├── PIPELINE.md       # 各阶段输入输出
│   ├── DECISIONS.md      # 选型记录
│   ├── INSTALL.md        # 环境细节
│   └── RESULT.md         # 统计报告（阶段4 生成）
├── legacy/               # 早期对照实验（勿改）
└── out/                  # 产出（git 排除，手动拉取）
```

## 7. 关键结论速查（给接手的人）

- **rec 档全量实测（289 万图）**：19.4 短语/图、25.7 框/图、精度下界 **70.1%**（1000 图 / 10590 对），
  有效信号 ≈ 13.6 短语/图。对比旧 C 档：8.7 短语/图、71.3%、有效信号 6.2 ——
  **rec 档用 1.2pt 精度换来两倍多的有效信号**，这是选它的理由
- **消融那批数字是子集数字，别当全量用**：消融取 `ground_shard0` 前 20000 条，
  该区段短语密度只有 11.5/图，而全量均值 23.1/图（同文件尾部 35.5/图，密度沿文件递增）。
  所以旧文档里的「9.14 短语/图、有效信号 6.4」是子集口径；精度 69.9%（子集）与
  70.1%（全量）能对上，说明规则的相对结论没问题，但绝对的量级数字要用全量的
- **清洗规则演进**：`--min-words 2`（删 45% 只为滤 0.5% 碎片，**已证伪**）→ 整词 garbled + min-words 1（B 档）→ +abstract +xdup（C 档）→ **去掉面积过滤 + cover + edge（rec 档，推荐）**
  - 全量证据支持放宽 min-words：1 词短语精度 69.2%（n=2027），与总体 70.1% 基本齐平
- **小物体不该删（仍待人工裁决确认）**：全量看小框确实低分（<0.5% 面积 → 47.8%，
  0.5-2% → 62.8%），但先前观察是判定器在 20px 尺度看不清（`nose` 大框 5/5 全对、小框屡判 NO）。
  这是 5.4 人工裁决要回答的第一个问题
- **消融结论**：`ratio`(-0.7pt) 与 `nbox`(-0.4pt) 精度反降被否；`cover`(+0.8pt)、`edge`(+0.1pt) 纳入
- 全部数字来自同判定器对照，精度是同判定器下的相对值，不是绝对真值（确认偏误下界）

## 8. 已知限制（迁移前须知）

- `out/` 不进 git，**数据必须手动拉**（见第 2 节）
- `/dev/shm/models` 重启后清空，sglang 首次启动自动从 `~/models` 拷贝（3 分钟）
- 校验精度是下界：判定器 gemma4 与 caption 同模型族，存在确认偏误；可靠的是同判定器下的相对比较
- CC3M 原 caption 不能当真值（网页 alt-text，含视觉不可见信息）
- `out/verify_*.jsonl` 是**追加+按 (path, phrase) 续跑**的（`src/batch.py`）。
  换了清洗档位重跑校验前必须把旧文件挪走，否则新旧档判定会混进同一个文件、
  `report.py` 会算出混合口径的精度

## 9. 本次迁移记录（139 → 140）

- **数据**：旧机 `bdhttp3.py` 文件服务（`http://10.52.101.139:8555`，根目录 = `penghaotian/`），
  aria2c 多连接拉取 `out/` 58 个文件共 10.83 GiB + `logs/` 34 个文件 36.6 MiB，
  逐文件比对 Content-Length 与本地字节数，全部一致；行数复核
  caption/ground 各 2,894,191、clean 2,894,189
- **环境**：`envs/sglang__0.5.12`、`models/gemma-4-26B-A4B-it`、`datas/cc3m-tsv`（images 250G）
  新机原本就有；`/dev/shm/models/gemma-4-26B-A4B-it` 已在。
  **缺 `envs/dam` 与 `models/Florence-2-large`**（只有阶段2 grounding 要用，产出已全量迁完）。
  要重跑阶段2 就跑 `bash install.sh`，它会建 dam 环境并拉 Florence-2-large ——
  别用 HTTP 拷 venv，符号链接和权限过不去
- **rec 档全量重跑**：旧 C 档产出保留为 `out/clean_c_old/`（2.0G），
  新 `out/clean/` 3.6G。旧校验基线保留为 `out/verify_clean_c_old.jsonl`（71.3%）
- **全量校验**：复用 8001-8008，10590 对 1.9 分钟跑完，YES=7425 / NO=3165，
  无 ERROR / UNPARSED / TOO_SMALL
- **审核页面**：`rule_ablation.html`（40 图，c vs rec）、`adjudicate_clean.html`（YES/NO 各抽 100），
  服务在 `10.52.101.140:8900`（复用本机 http.server 的 IP 白名单 ACL，非白名单 403）
- 顺手修的两处：`run/3_clean.sh` 的 TRAIN 分支（旧档 → rec 档，交接文档里留的待办）、
  `scripts/make_adjudicate_html.py` 的 `--help` 崩溃（help 文本里裸 `%`）

