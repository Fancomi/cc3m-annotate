# 备份清单 · 数据与产物

> 机器 `10.52.101.140` · 仓库 `/root/paddlejob/workspace/env_run/penghaotian/vision_encoder/cc3m-annotate`
> 生成于 2024-09-04（cc12m 完成清洗后）。所有大小均为实测字节数。
> 代码走 git（已推送到 `github.com/Fancomi/cc3m-annotate`），本节只列 `out/`、`out_cc12m/`、`logs/` 等被 `.gitignore` 排除、必须单独备份的数据。

## 总览

| 数据集 | 体积（GiB） | 阶段覆盖 | 图数 |
|---|---|---|---|
| **cc3m** 核心产物 | **~20.3** | 1 caption / 2 grounding / 3 clean / 4 verify+verdict | 2,894,191 |
| **cc12m** 产物 | **~45.0** | 1 caption / 2 grounding / 3 clean（阶段4 按手册只抽样，未跑全量） | 10,968,539 |
| 运行日志 logs/ | ~0.31 | — | — |
| **小计（必须备份）** | **~65.6** | | |
| cc3m 可选/可重建项 | ~2.3 | clean_c_old、caption retry、旧抽样基线 | — |
| **合计（含可选）** | **~67.9** | | |

---

## 1. cc3m 核心产物（out/，必须备份，约 20.3 GiB）

| 备份项 | 路径 | 体积 | 内容 | 行数 | 可否重建 |
|---|---|---|---|---|---|
| gemma4 caption | `out/caption/shard{0-7}.jsonl` | 3.96 GiB | gemma4 两级 caption：`gemma_short`(约20词) + `gemma_dense`(约178词) | 2,894,191 | 需 8×H800 约 30h |
| Florence-2 grounding | `out/ground/ground_shard{0-7}.jsonl` | 4.42 GiB | 短语→像素框，`grounding` dict + `img_wh` | 2,894,191 | 需 8×H800 约 29h |
| **最终训练数据(rec)** | `out/clean/clean_shard{0-7}.jsonl` | 3.58 GiB | 清洗后的短语/框（19.35 短语、25.67 框/图） | 2,894,189 | 清洗 20 分钟（依赖上游） |
| verdict 边车 | `out/verdict/verdict_shard{0-7}.jsonl` | 1.87 GiB | 逐框对齐 verdict（未审=null），与 clean 严格逐行对齐 | 2,894,189 | 纯 CPU 4 分钟（依赖 verify_full） |
| gemma4 全量审核 | `out/verify_full.jsonl` | 6.34 GiB | 30,303,579 条 YES/NO（覆盖 40.79% 框位） | — | 需 8×H800 约 65h |
| 抽样校验基线 | `out/verify_clean.jsonl` | 2.3 MiB | 1000 图抽样（10,590 对），人工裁决挂载于此 | 10,590 | 重跑 2 分钟 |
| 清洗规则消融 | `out/ab/` | 165 MiB | 12 档位 clean_shard + 11 份对照 verify | — | 重建约 1h |
| 人工裁决（唯一） | `out/human_adjudication_clean.json*` | ~150 KiB | 人工 100 条（YES/NO 各 50）——**精度校准唯一依据，不可重建，最高优先级** | 100 | ❌ 不可重建 |

## 2. cc3m 可选 / 可重建项（约 2.3 GiB）

| 备份项 | 路径 | 体积 | 说明 |
|---|---|---|---|
| 旧 C 档 clean | `out/clean_c_old/` | 1.94 GiB | 已被 rec 档取代，仅做档位对比需要（结论已在 docs） |
| caption 补跑中间文件 | `out/caption/retry_*.jsonl` | 0.34 GiB | 阶段1 补跑临时文件，结果已压实进 shard |
| 旧抽样基线 | `out/verify_clean_s400.jsonl`（0.7M）/ `out/verify_clean_c_old.jsonl`（1.8M） | 2.5 MiB | 早期版本，备份 |
| 审核页面 | `rule_ablation.html`（3.2M）/ `adjudicate_clean.html`（3.0M） | 6.2 MiB | scripts 可随时重新生成（git 已跟踪） |

## 3. cc12m 产物（out_cc12m/，必须备份，约 45.0 GiB）

| 备份项 | 路径 | 体积 | 内容 | 行数 | 可否重建 |
|---|---|---|---|---|---|
| gemma4 caption | `out_cc12m/caption/shard{0-7}.jsonl` | 15.65 GiB | 同 cc3m 结构（short + dense），10,968,539 唯一 path、零重复 | 10,968,539 | 需 8×H800 约 145h |
| Florence-2 grounding | `out_cc12m/ground/ground_shard{0-7}.jsonl` | 16.22 GiB | 短语→像素框（阶段2 已跑完） | 10,968,532 | 需 8×H800 约 160h |
| **最终训练数据(rec)** | `out_cc12m/clean/clean_shard{0-7}.jsonl` | 13.16 GiB | 清洗后短语/框 | 10,968,515 | 清洗约 76 分钟 |

> 说明：cc12m 阶段4 按 `docs/CC12M.md` 建议**不跑全量审核**（只抽样 1000~5000 图），
> 故没有 `verify_full.jsonl` / `verdict` 级产物；密度口径对比见该文档（cc3m 全量 23.1/33.8，cc12m 待阶段2 终值后复测）。

## 4. 运行日志（logs/，可选但建议留，约 0.31 GiB）

| 备份项 | 体积 | 内容 |
|---|---|---|
| `logs/cc12m/` | 273 MiB | cc12m 阶段1/2/3 各卡日志（cap_*、gnd_*、clean.log） |
| `logs/*.log` | 43 MiB | cc3m 各阶段日志（verify_full、gnd_*、sgl_* 等） |

## 5. 不需要备份（明确排除）

| 项 | 体积 | 原因 |
|---|---|---|
| cc3m 原图 `datas/cc3m-tsv/images/` | ~250 GiB | 公开数据集，产物里 `path` 字段指向它 |
| cc12m 原图 `datas/cc12m/images/` | ~1.1 TiB | 公开数据集 |
| 模型权重 / conda 环境 | — | `install.sh` 一键重建 |
| 兄弟项目 `../AtomUnitLM`、`../open_clip` | 5.3 GiB / 29 MiB | 独立项目，是否备份另议 |

## 6. 备份建议

- **最高优先级**：`out/human_adjudication_clean.json*`（不可重建）→ 之后 cc3m 的
  `caption`/`ground`/`clean`/`verify_full`/`verdict`（重建成本按小时计）→ cc12m 的
  `caption`/`ground`/`clean`（重建成本按上百小时计）。
- 用 rsync 断点续传（同 MANIFEST/MIGRATION 方案），或 tar 后一次传输。
- 传输后按本节行数做完整性校验（每文件 `wc -l` 应等于上表行数）。
