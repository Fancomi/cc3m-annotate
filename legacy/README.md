# legacy · 早期实验代码

这里的代码**不在生产链路上**，只用于复现 `docs/DECISIONS.md` 里的选型结论。不要在其上改代码或作为新功能的起点。

各脚本的路径、环境、字段名都是当时的样子，与当前 `src/` 不兼容（它们读的是 `merged.jsonl`、`qwen_dense` 这类旧字段）。要复现需自备当时的中间产物。

| 目录 | 内容 | 支撑了什么结论 |
|---|---|---|
| `stage1_threeway/` | Florence-2 三级 caption、NOVIC 开放词汇分类、Qwen caption（sglang 版 + transformers 本地版）、三方合并与定量分析 | NOVIC 只分类不出框故淘汰；检测头词表是硬天花板 |
| `stage2_crossfeed/` | 用外部 dense caption 驱动 F2 grounding | grounding 的产出丰富度由输入 caption 决定（框数 9.8 → 39.4） |
| `stage3_verify/` | 裁框问 VLM 的幻觉校验；分层抽样人工裁决界面 + Wilson 区间还原；带服务端落盘的静态服务器 | 精度下界估计方法；「框数翻 3.4 倍精度只掉 3.2 点」 |
| `stage4_select/` | gemma4 vs Qwen 同图同 prompt 对照 | gemma4 胜出 5.3 个点，成为生产链路的 caption 模型 |

有两样东西仍值得复用：

- `stage3_verify/make_adjudicate_html.py` + `apply_human_adj.py` + `serve_review.py` —— 分层抽样人工裁决的完整闭环（界面、落盘、Wilson 区间还原）。当前所有精度数字都是「VLM 判 VLM」的下界，要拿到可引用的真实精度就靠这套。
- `stage3_verify/make_review_html.py` —— 自包含 HTML 抽查页（图片 base64 内嵌 + 框叠加），换一下字段名就能看新产出。
