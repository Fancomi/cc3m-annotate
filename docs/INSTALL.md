# 环境配置

目标硬件：8×H800 80G、CUDA 12.9 工具链、NVIDIA 驱动 550+、Ubuntu 22.04、`/dev/shm` ≥ 50G 空闲。

```bash
bash install.sh            # 全部步骤（幂等，已完成的跳过）
bash install.sh --list     # 列出步骤
bash install.sh env_f2     # 只跑某一步
```

完成标记写在 `$WORK_ROOT/.install_cc3m/*.done`，删掉对应文件即可重跑该步。

## 两个 venv，不能合并

| 环境 | python | torch | transformers | 用途 |
|---|---|---|---|---|
| `envs/dam` | 3.10 | 2.6.0+**cu124** | **4.46.3** | 阶段2 本地跑 Florence-2 |
| `envs/sglang__0.5.12` | 3.12 | 2.11.0+**cu129** | 5.x | 阶段1/1b/4 起 gemma4 服务 |

**为什么必须隔离**：

- **Florence-2 锁死 transformers 4.46.x**。它是 `trust_remote_code` 的自定义架构（权重目录里的 `modeling_florence2.py`），引用了 4.46 的私有 API（`_prepare_4d_attention_mask` 等），transformers 5.x 已删除，升级直接 `ImportError`。
- **sglang 0.5.12 要 transformers 5.x + torch cu129**，与上面互斥。
- **F2 侧的 torch 锁 cu124**：驱动 550 不支持 cu129 wheel。sglang 侧能用 cu129 是因为它自带 kernel 不依赖驱动侧新特性。

不要试图"统一版本"，这两条约束是硬的。

## sglang 装不上时

`install.sh` 走 `pip install sglang[all]==0.5.12`，代理环境下常因拉不到 wheel 失败。改用源码编译：

```bash
# 源码编译需要预取 6 个第三方仓（cutlass / triton / flashinfer / sgl-attn / fmt / mscclpp，共约 1.5G）
# 关键点：CMake 的 FetchContent 会 fork 内层 git 进程，它们读不到命令行 --config，
# 只继承环境变量，所以代理下必须用 GIT_CONFIG_COUNT/KEY/VALUE 注入 http.version=HTTP/1.1
export GIT_CONFIG_COUNT=2
export GIT_CONFIG_KEY_0=http.version GIT_CONFIG_VALUE_0=HTTP/1.1
export GIT_CONFIG_KEY_1=http.lowSpeedTime GIT_CONFIG_VALUE_1=600
```

预取后用 `FETCHCONTENT_SOURCE_DIR_<NAME>` 指向本地副本，任一仓中断不会拖垮整个 configure。
sgl-attn 的两个子模块（`csrc/cutlass`、`csrc/composable_kernel`）从未被 CMakeLists 引用，一律不递归克隆。

## 权重

| 模型 | 大小 | 用途 |
|---|---|---|
| `models/Florence-2-large` | 1.5G | 阶段2 grounding |
| `models/gemma-4-26B-A4B-it` | 49G | 阶段1/1b/4 服务 |

gemma4 需 HuggingFace 授权，`install.sh` 下载失败时手动放到 `models/gemma-4-26B-A4B-it`。

`run/sgl.sh up` 首次会把 gemma4 拷到 `/dev/shm/models/`（约 49G，一次性），后续启动直接命中，加载快很多。

## 数据集

```
datas/cc3m-tsv/
├── _shards/cc3m-train-{0000..0575}.tsv     每行 `图片绝对路径 \t cc3m 原始 caption`
├── images/cc3m-train-{0000..0575}/*.jpg    249G，共 2894191 张
└── annotations/_stats.json                 {"num_shards":576, "num_rows":2894191}
```

pipeline 只读 `_shards/*.tsv`，图片路径从 tsv 里取，不扫目录。tsv 编号即记录的 `shard` 字段。

## 常见故障

| 现象 | 原因 | 处理 |
|---|---|---|
| 请求全部 `APIConnectionError` | 代理拦截 127.0.0.1 | `src/common.py:clear_proxy()` 已自动清；手工调试时先 `unset http_proxy https_proxy` |
| sglang 某实例静默退出，该卡显存空出 | 长时批处理触发 watchdog | 单独重启该实例：`CUDA_VISIBLE_DEVICES=N ... --port 810N`；阶段1 的客户端会自动路由到存活实例，产出不受影响，失败条目由阶段1b 补齐 |
| F2 `ImportError: _prepare_4d_attention_mask` | transformers 被升到 5.x | 回滚 `uv pip install --python envs/dam/bin/python transformers==4.46.3` |
| F2 batch 报形状冲突（850 vs 273） | 同 batch 内输入不等长 | F2 的 SDPA 不支持 padding，`s2_grounding.py` 已按长度分桶并截到 min_len |
| 阶段2 OOM | sglang 仍占着显存 | `bash run/sgl.sh down` 后再跑；或降 `GND_BATCH` |
| `/dev/shm` 空间不足 | 权重副本累积 | `rm -rf /dev/shm/models/<不用的模型>` |
