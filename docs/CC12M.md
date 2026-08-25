# 在 CC12M 上跑同一套 pipeline

同一份代码跑两个数据集，靠 `DATASET` 环境变量分派（`run/env.sh`）。
`DATASET` 不设时行为与从前逐字节相同，cc3m 的产出路径没动过。

| | cc3m | cc12m |
|---|---|---|
| 图数 | 2,894,191 | **10,968,539**（3.79×） |
| 输入 | `datas/cc3m-tsv/_shards`（576 个 tsv） | `datas/cc12m/_shards`（2176 个 tsv） |
| 产出 | `out/` | `out_cc12m/` |
| 日志 | `logs/` | `logs/cc12m/` |
| sglang 端口 | 8101-8108（自己拉起） | 8001-8008（**复用常驻实例**） |
| 报告 | `docs/RESULT.md` | `docs/RESULT_cc12m.md` |

## 0. 先解包：wds tar → 图片文件 + tsv 清单

CC12M 原始格式是 webdataset（2176 个 tar，1.1T，每条 `KEY.jpg` + `KEY.json` + `KEY.txt`），
与 cc3m 的「图片文件 + tsv 清单」不同。**解包成同构布局**，这样阶段 1~4 一行都不用改：

```bash
python3 scripts/wds_extract.py \
  --src /root/paddlejob/gpfsspace/cc12m-wds \
  --dst $WORK_ROOT/datas/cc12m --workers 48
```

产出 `datas/cc12m/images/cc12m-train-NNNN/KEY.jpg` + `datas/cc12m/_shards/cc12m-train-NNNN.tsv`。
实测 **4.3 分钟 / 1020G**，10,968,539 图 + 2176 个 tsv，与 `_info.json` 的 `num_samples` 完全一致。

**为什么解包而不直接读 tar**：阶段 2/3/4 都按 `path` 随机打开图片。tar 内随机访问要么
全表扫描，要么把 tar 偏移塞进 `path` 字符串 —— 后者会让产出里的 `path` 变成非标准形式，
下游拿到数据还得先学一套解析规则。用磁盘换无歧义。
坐标语义因此与 cc3m 完全相同：**原图像素，解包不做任何缩放**。

断点续传：tsv 最后才原子落盘，所以「tsv 存在」= 该 tar 已完整解完，重跑直接跳过。

`suspect_jpg 1163`（占 0.011%）是尾部缺 FFD9 的图，**全部保留不丢**：实测这类文件
PIL 仍能完整解码（有的多一个填充字节，有的截断在 100KB 边界但扫描数据是全的）。
真读不出来的图阶段1 会记 error 并在重跑时重试，不该在解包时静默丢。

## 1. 各阶段

```bash
DATASET=cc12m bash run/1_caption.sh      # 阶段1  caption
DATASET=cc12m bash run/1b_compact.sh     # 阶段1b 压实 + 完整性检查
DATASET=cc12m bash run/2_grounding.sh    # 阶段2  grounding
DATASET=cc12m TRAIN=1 bash run/3_clean.sh   # 阶段3  清洗（rec 档）
DATASET=cc12m SAMPLE=1000 bash run/4_verify.sh   # 阶段4  抽样校验 + 报告
```

**不要跑 `bash run/sgl.sh down`**（cc12m 分支下它只会杀 8001-8008 段 ——
也就是要复用的那批常驻实例）。阶段2 不需要腾显存：每卡还剩约 12G，
F2-large fp16 只要约 1.6G，共卡即可。

### 实测吞吐（8×H800，与常驻 sglang 共卡）

| 阶段 | 实测 | 全量 | 对比 cc3m |
|---|---|---|---|
| 1 caption | 341 ms/it/进程 × 8 → **23.5 图/s** | **约 130 小时** | cc3m 26.8 图/s / 30h |
| 2 grounding | 未跑 | 约 110 小时（按 3.79× 外推） | 29h |
| 3 清洗 | 未跑 | 约 76 分钟 | 20min |
| 4 校验（全量） | 未跑 | 约 247 小时 | 65h |

阶段1 比 cc3m 慢约 12%：cc12m 图更大（93 KB/图 vs 86 KB/图）。

**阶段4 在 cc12m 上不要跑全量**。cc3m 那次全量 65 小时换来的结论是
「全量 69.88% 与 1000 图抽样 70.1% 只差 0.2pt」—— 抽样本来就没偏。
在 cc12m 上重复一次要 247 小时，买不到新信息。抽 1000~5000 图即可。

### 磁盘

解包 1020G + 产出约 50G（阶段1 约 16G、阶段2 约 17G、阶段3 约 14G）。
两者都在 `/dev/nvme2n1`（3.5T），解包后剩 1.4T，够。

## 2. 一次跑起来会踩到的四个坑（已在代码里修掉）

按暴露顺序，都是「只跑一个数据集时永远不会发作」的：

1. **`sgl.sh down` 不看端口**，无条件 `pkill -f sglang.launch_server`，
   一次 down 把别人常驻的 8001-8008 全带走。已改成按 `PORT_BASE` 段逐个 pgrep。
2. **`4_verify.sh` 把报告写死在 `docs/RESULT.md`**，跑一次 cc12m 就覆盖 cc3m 那份 ——
   而它第 5 节的人工裁决（100 条，唯一的精度校准依据）重建不了。已按数据集分流。
3. **`install.sh` 里 torch 的 cu124 索引一直没生效**：`uvpip` 硬塞
   `--index-url $PIP_INDEX`，调用方再传一个，uv 不允许该参数出现两次、整条命令报错退出；
   而 `run_step` 写的是 `"step_$1" && mark "$1"`，`&&` 在整个函数体里关掉了 `set -e`，
   所以失败没有中断，torch 被当 timm/accelerate 的传递依赖从镜像源拉了最新版（cu13 wheel），
   最后按自检返回码打上 `.done`。两处都已修。
4. **`huggingface-cli` 已弃用**（huggingface_hub 0.35 起只打印一句提示、不下载任何东西，
   还以 0 退出，所以 `|| die` 也没触发，看着像成功）。新入口是 `hf`，已改成两个名字都探测。

网络：这台机器上 **hf-mirror 直连超时，走代理直连 huggingface.co 可以**：

```bash
HF_ENDPOINT=https://huggingface.co http_proxy=http://agent.baidu.com:8188 \
  https_proxy=$http_proxy bash install.sh weights
```

## 3. 与 cc3m 结论的关系

cc3m 那套结论（rec 档参数、不按面积过滤小物体、verdict 当置信度分层而非硬过滤、
未标注区域是漏标不是背景）都是关于**方法**的，不依赖数据集，直接沿用。
但**所有数字要在 cc12m 上重测**：短语/框密度、精度下界、k 曲线都由 caption 分布决定，
而 cc12m 的原始 alt-text 比 cc3m 更长更噪（未过滤的网页文本），不能假定一致。

`docs/MANIFEST.md` 那份数据卡的口径说明（审核覆盖率、`null` ≠ `NO`、
「每短语只取首框」是审核口径不是数据口径）对 cc12m 同样成立，
但**具体数字必须换成 cc12m 自己的**，别直接复制。
