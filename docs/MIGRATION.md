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
| `out/human_adjudication_clean.json` | 21K | **人工裁决 100 条**（第 5 节精度校准的唯一依据） | **重建不了**，得重新人工标 |

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
# 8900 服务（8899 已被 /tmp/hdr_capture.py 占用）。本机 http.server 带 IP 白名单
# （只有 GPU 机器池的 BNS + 127.0.0.1），办公机 IP 不在里面会 403，必须显式放行：
NETS=$(python3 -c "print(','.join(f'172.24.{o}.0/24' for o in range(16,32)))")
python3 scripts/serve_review.py --dir . --port 8900 \
  --save out/human_adjudication_clean.json --allowednet "$NETS"
# 浏览器开 http://10.52.101.140:8900/rule_ablation.html
```

> ACL 只接受 /24~/32 掩码，给 /8 或 /16 会让它在后台线程里 `sys.exit`，
> 结果白名单变空、**所有请求包括 127.0.0.1 都 403**。要放宽一段就逐个 /24 列出来。
> 别用别的文件服务（如 `bdhttp3.py`）打开裁决页 —— 那种服务没有 `/save` 端点，
> 裁决只会留在浏览器 localStorage 里。页面现在会在页首弹红色告警条提示这种情况。

### 5.4 人工裁决（校准精度下界）

```bash
python3 scripts/make_adjudicate_html.py --verify out/verify_clean.jsonl \
  --out adjudicate_clean.html --per-stratum 50 \
  --save-url http://10.52.101.140:8900/save
# 服务同 5.3（要带 --allowednet 放行办公机网段）
# 浏览器开 http://10.52.101.140:8900/adjudicate_clean.html（答一题即落盘，可换端口/浏览器续答）
# 答完 → python3 scripts/apply_human_adj.py --json out/human_adjudication_clean.json --dir . \
#   --auto-precision 70.1   # 用 RESULT.md 里 rec 档的精度
```

`--save-url` 写绝对地址是为了页面从别的 origin 打开时也能落盘（`text/plain` 请求头
避开 serve_review.py 不处理的 CORS 预检）。落盘失败时页首会弹红条，不再是页脚一行小字。

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
- **小物体不该删（人工裁决已确认）**：自动判定看到的「小框低分」（<0.5% 面积 → 47.8%，
  0.5-2% → 62.8%）是判定器在 20px 尺度看不清造成的，不是标注错。人工按同样分桶重新
  加权后 <0.5% 是 81.0%、0.5-2% 是 85.1%，与大框（80.0%）齐平；判定器误否率在小框上
  60~64%、大框只有 19%。详见 `docs/RESULT.md` 5.1。**不加面积过滤的决定成立。**
- **人工裁决结论（100 个样本，YES/NO 各 50）**：a=P(真存在|判YES)=96.0%、
  b=P(真存在|判NO)=42.0%，还原真实精度 **79.9%**（95% 区间 69.5–86.0%），
  比自动下界 70.1% 高 9.8pt —— 判定器偏保守，70.1% 可放心当下界用。
  判定器自身准确率 84.6%
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

## 10. 全量 gemma 校验（已完成）

把阶段4 那个抽样估计器的样本量拉到**全部图**，得到 3030 万条 YES/NO。
注意它**不是**逐对标签文件 —— 覆盖口径见下面「审核覆盖多少」一节，别当训练权重直接用。

```bash
# 注意：不能用 run/4_verify.sh —— 它会写 out/verify_clean.jsonl（1000 图基线，
# 人工裁决就挂在这个文件上）并重生成 docs/RESULT.md（会抹掉第 5 节）
URLS8=$(python3 -c "print(','.join(f'http://127.0.0.1:{p}/v1' for p in range(8001,8009)))")
source run/env.sh          # 取 $PY_SGL；别写 ~/envs/...，这台机器的环境在 env_run/penghaotian/envs/ 下
setsid nohup "$PY_SGL" -u src/s4_verify.py \
  --in-dir out/clean --pattern 'clean_shard*.jsonl' --out out/verify_full.jsonl \
  --urls "$URLS8" --model /dev/shm/models/gemma-4-26B-A4B-it \
  --sample 3000000 --max-boxes 12 --concurrency 128 > logs/verify_full.log 2>&1 &
```

- 规模 30,303,579 对（286.9 万图 × 平均 10.56 个短语，`--max-boxes 12` 截断后）
- **实测 65.1 小时跑完**（3906 分钟，8ms/对 ≈ 129 对/秒），8 卡 GPU 利用率 32~89%，
  客户端单核 94%、RSS 37 GiB。产出 `out/verify_full.jsonl` 6.4G
- 断点续传：中断后重跑同一条命令即可，按 (path, phrase) 跳过已完成
- 判定条件必须与已校准的那批一致（pad 0.12、maxside 512、同 prompt），
  否则 70.1% / 79.9% 这两个数字对不上，别为了提速改这些

### 审核覆盖多少（容易误读，务必先看这节）

`s4_verify.py` 有三处截断，两处真正生效：`if r.get("grounding")` 跳过空图、
`k >= max_boxes` 每图只取前 12 个短语、`boxes[0]` 每短语只取首框
（VAGUE 那条在 clean 输入上实测删 0 条，阶段3 已删净）。全量重算：

| | 数量 | 覆盖 |
|---|---|---|
| 图 | 2,868,984 / 2,894,189 | 99.13%（25205 张清洗后短语为空，无可审） |
| 短语 | 30,303,579 / 56,007,139 | **54.1%** |
| (短语,框) 对 | 30,303,579 / 74,289,118 | **40.8%** |

损失拆解：`max-boxes 12` 砍掉 25,703,560 个短语（占框总数 34.6pt），
`boxes[0]` 再砍掉 9,846,050 个框（13.3pt）。

**为什么这么设计**：这是估计量而非标注器。同一短语挂 5 个框若全审，它在均值里的权重
就是别人的 5 倍；同一张图内的短语高度相关（同次 caption、同次 F2 解码），预算花在更多图上
比花在同一张图的第 35 个短语上更能降方差。另外续传主键是 `(path, phrase)`，
**文件结构上装不下同一短语的多个 verdict** —— 想要逐框标签得改主键，不是加参数就行。

**别拿子集分母算覆盖率**：密度沿文件递增，`clean_shard0` 前 20000 行只有 9.14 短语/图、
11.50 框/图，`clean_shard7` 末 20000 行是 29.02 / 41.86，全量均值 19.35 / 25.67。
用头部分母会算出「覆盖 85% 短语」这种错的结论（真实 54.1%）。同第 7 节那个坑。

### 全量结果

| verdict | 数量 | 占比 |
|---|---|---|
| YES | 21,175,099 | 69.88% |
| NO | 9,127,032 | 30.12% |
| ERROR | 1,195 | 0.004% |
| TOO_SMALL | 252 | 0.001% |
| UNPARSED | 1 | — |

**全量精度下界 69.88%**（n=30,302,131），与 1000 图抽样基线 70.1% 只差 0.2pt ——
说明当初那次抽样没偏，`docs/RESULT.md` 里的数字在全量上站得住。
ERROR 那 1195 条重跑同一命令会自动重试，占比 0.004%，不重跑也不影响任何统计。

分层（全量，取代 RESULT.md 里的抽样版）：

| 框面积 | 精度 | n | | 短语词数 | 精度 | n |
|---|---|---|---|---|---|---|
| <0.5% | 44.96% | 3.82M | | 1 | 68.61% | 5.81M |
| 0.5-2% | 62.80% | 5.58M | | 2 | 66.87% | 11.31M |
| 2-10% | 70.83% | 8.41M | | 3 | 69.71% | 7.41M |
| ≥10% | 80.02% | 12.49M | | 4+ | 77.27% | 5.77M |

两条趋势与抽样一致：框越小判定越低（但第 5.1 节已证明这是判定器看不清，不是标注错）；
短语越长精度越高。1 词短语 68.61% 与总体 69.88% 基本齐平，再次支持 rec 档的 `--min-words 1`。

按短语名下框数分层（反直觉）：1 框 69.93%、2 框 70.70%、3 框 70.95%、≥4 框 61.78% ——
多框短语的首框并不差，只有框数失控（≥4）才掉。

### 69.88% 描述的是「前 12 短语首框」这个总体，不能外推到全部 7429 万对

按短语在图内的序号 k 拆开（`clean_shard0` 的图，各档 n 均 >26 万）：

| k | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 精度 | 82.44% | 79.42% | 74.65% | 72.14% | 69.87% | 67.73% | 66.18% | 64.99% | 64.27% | 63.64% | 63.32% | 62.89% |

单调降 20pt —— 靠前的短语是主体（大、显著、caption 先点名），靠后是细节和从属物。
曲线在 k=11 已趋平（末三档只降 0.3~0.4pt），按 ~60% 外推被截掉的 2570 万个短语，
**全短语精度约 65%**。

这不影响已有结论的可比性：1000 图基线 70.1%、人工裁决 79.9%、全量 69.88%
用的是同一套抽样规则，三者能互相校准，0.2pt 的吻合依然说明当初抽样没偏。
但**不能**把 69.88% 当成全部 (短语,框) 对的精度。
（消融那批用 `--max-boxes 0`，是另一个总体，别与这些数字混着比。）

逐图 YES 分布（校验覆盖 10.56 对/图）：平均 **7.38 个 YES/图**，
99.2% 的图至少有 1 个 YES、96.8% ≥2 个、93.4% ≥3 个、83.8% ≥5 个；
全图 0 个 YES 的只有 22851 张（0.80%）—— 这批可以直接丢掉。

**这批标签能干什么、不能干什么**（依据第 5 节的人工裁决）：
按 YES 筛，在**被审到的那 40.8% 框**里保留 69.88%、精度升到 96.0%，
但会误删「判 NO 实际正确」的那 12.6%（≈2.4 短语/图）。小框上判定器误否率 60~64%，
所以**小框的 NO 不该直接删** —— 更稳的用法是把 verdict 当置信度分层，而不是当硬过滤。
剩下 59.2% 的框**没有 verdict**，硬过滤会把它们全部误伤成「未标注」。

若真要逐对标签，得把 `run_pool` 主键改成 `(path, phrase, box_idx)` 并去掉
`--max-boxes` 上限，补跑剩下约 4400 万对。吞吐实测（`--by-image` 变体、多进程分片）：
单进程 128 并发 82/s、4 进程 120/s、8 进程 157/s，按图打包一次解码**无收益**，
GPU 均值约 60% —— 相对生产的 129/s 只有 1.2 倍余量，补跑仍需约 78 小时。
所以「太耗时」的结论成立，但**不必补跑**，见第 11 节。

## 11. verdict 边车（消除「NO」与「未审」的歧义）

`verify_full.jsonl` 主键是 `(path, phrase)`，下游拿它过滤时无法区分
「判 NO」与「根本没审」—— 这是这批数据唯一的歧义来源。边车把它摊平成与 clean
**逐框对齐**的文件，每个框位显式给一个槽，未审写 `null`：

```bash
python3 scripts/make_verdict_sidecar.py     # 默认 out/clean + out/verify_full.jsonl → out/verdict/
```

产出 `out/verdict/verdict_shard{0-7}.jsonl`，1.9G，**2,894,189 行与 clean 一一对应**
（含 grounding 为空的图）。按 `path` 关联，框按下标对齐：

```json
{"path": ".../001478524.jpg", "n_pair": 3, "n_audited": 2,
 "verdict": {"The text": ["YES"], "The characters": ["NO", null]}}
```

纯 CPU 两趟、4 分钟（索引 1.9min + 改写 2.2min），峰值内存约 12G。

自检结果（脚本末尾会打印，可复现）：

- 框位 74,289,118，有 verdict 30,303,579（**40.79%**），`null` 43,985,539
- **verify 里对不上 clean 的 (path,phrase) = 0**，且 30,303,579 行 → 30,303,579 个唯一主键
  （无重复主键，说明那次全量跑没有因续传产生重复行）
- 被审短语的 k 分布恰好落在 0~11、被审框的 `box_idx` 分布恰好只有 `{0}` ——
  从数据侧独立验证了第 10 节说的两处截断，不是靠读代码推断的
- verdict 计数与第 10 节完全一致（YES 21,175,099 / NO 9,127,032 / ERROR 1,195 /
  TOO_SMALL 252 / UNPARSED 1）

k 分布同时给出「有第 k+1 个短语的图数」：k=0 是 2,868,984，单调降到 k=11 的 2,104,554。

**为什么这就够了，不需要补抽样**：`k`（短语在图内序号）和 `box_idx` 是从 clean
直接数出来的确定性特征，零成本，而且已经在被审的那 40.8% 上标定出精度曲线
（k=0 82.44% → k=11 62.89%，见第 10 节）。下游按 (k, box_idx, 面积桶) 给全部
7429 万对加权即可，不需要任何额外推理，也不需要对未审区间做点估计。

## 12. 下游能怎么用（结论写在 docs/MANIFEST.md）

`docs/MANIFEST.md` 是随数据一起交付的数据卡（数据根目录下有同步副本），
新增「能做什么训练、不能做什么」一节，要点：

- **「每短语只取首框」是审核口径，不是数据口径。** `clean_rec` 里每短语的全部框都在，
  转 detect 样本格式在数据层面没障碍。受限的是 verdict：多框短语占 20.92%（名下 40.38%
  的框），其中 **18,281,979 个非首框永远没有 verdict**，所以 **verdict 不能用来在同一
  短语的多个框之间取舍**
- **更要紧的限制是没有 recall 保证，而且补跑审核也解决不了**：框来自 F2 对 dense caption
  做短语定位，caption 没点名的物体就没有框；短语是自由文本，跨图无类别对齐。
  所以未标注区域是**漏标不是背景**，不能当负样本
- **适合** phrase grounding / REC / region-text 对齐预训练（短语 → 首框，与 verdict 同口径）；
  **不适合**闭集 detection，尤其不适合做 mAP / recall 评测（漏标会系统性压低分数）
- 真要做 detect 式训练：partial-annotation 类损失（未标注区域不进负样本）+
  只用首框 verdict 加权 + 多框短语只取首框或整条丢掉

边车与 clean **严格逐行对齐**（已逐行核对 path，shard0/shard7 各 36 万行 0 处错位），
关联不必建索引，直接 `zip` 两个文件即可，MANIFEST 里有片段。
