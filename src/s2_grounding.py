#!/usr/bin/env python3
"""阶段 2 · grounding：用 Florence-2 把 caption 里的短语定位成框。

输入  <cap-dir>/shard*.jsonl        阶段 1 的 gemma_dense caption
输出  <out>/ground_shard{N}.jsonl   每行 {id, shard, path, img_wh, grounding, n_phrase, n_box}
        grounding = {短语: [[x0,y0,x1,y1], ...]}，坐标为原图像素

机制：F2 的 <CAPTION_TO_PHRASE_GROUNDING> 是 seq2seq —— 输入整段 caption，
模型自己从文本里摘短语并逐个配框。它只负责"在哪"，不负责"有什么"：
caption 没写的物体不会被 ground 出来（实测 88.5% 的短语可逐字回溯到 caption）。

三个性能/正确性要点，改代码前务必理解：

1. batch 推理提速 3.85×，但必须禁 EOS。
   generate 在 batch 模式下，某序列先到 EOS 会被剔除，batch 维度 N->N-1
   改变了其余序列的 attention 上下文，输出随之漂移。禁 EOS（eos_token_id=-1）
   让全 batch 同步生成到 max_new_tokens，实测与单图逐 token 一致。

2. 禁 EOS 后必须在首个 </s> 处截断再解析。
   模型在自然结束点吐 </s>，之后是重复碎片（`- -`、`materialy`）。
   不截断的话垃圾短语占 32%；截断后降到 4.6%，且与自然 EOS 的结果完全一致。

3. 长度分桶 + 截到 min_len。
   F2 的 SDPA 实现不支持 padding（会形状冲突），所以同 batch 必须等长。
   按 caption 词数排序后相邻成组，截断损失最小。
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TASK_GROUND, iter_shards, load_done, open_append, take_shard, write_jsonl

# 自然生成长度实测 p50=199 p90=360 p99≈965。420 覆盖约 92%，
# 再往上是拿全 batch 的时间去换少数长尾图的尾部短语，不划算。
DEFAULT_MAX_NEW = 420


def main():
    ap = argparse.ArgumentParser(description="阶段2：Florence-2 batch grounding")
    ap.add_argument("--cap-dir", required=True, help="阶段1 输出目录")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--model", default="/root/paddlejob/workspace/env_run/penghaotian/models/Florence-2-large")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8, help="batch 大小；8 是实测吞吐拐点")
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW)
    ap.add_argument("--limit", type=int, default=0, help=">0 时只跑前 N 条（冒烟）")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor

    items = [{"id": r["id"], "shard": r["shard"], "path": r["path"], "dense": r["gemma_dense"]}
             for r in iter_shards(args.cap_dir) if r.get("gemma_dense")]
    items = take_shard(items, args.shard, args.num_shards)
    if args.limit:
        items = items[:args.limit]

    out_f = os.path.join(args.out, f"ground_shard{args.shard}.jsonl")
    done = load_done(out_f)
    items = [it for it in items if it["path"] not in done]
    print(f"[gnd{args.shard}] 待跑 {len(items)}（已完成 {len(done)}）", flush=True)
    if not items:
        return

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float16).cuda().eval()
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    print(f"[gnd{args.shard}] gpu{torch.cuda.current_device()} 模型已加载", flush=True)

    def to_cuda(d):
        return {k: (v.cuda().half() if v.dtype == torch.float32 else v.cuda()) for k, v in d.items()}

    def encode(it):
        img = Image.open(it["path"]).convert("RGB")
        return proc(text=TASK_GROUND + it["dense"], images=img, return_tensors="pt"), img.size

    def generate(inps):
        """inps 为单元素时走单图，多元素时截到 min_len 后 stack 成 batch。"""
        if len(inps) == 1:
            batch = to_cuda(inps[0])
        else:
            n = min(x["input_ids"].shape[1] for x in inps)
            batch = to_cuda({k: torch.stack([x[k][0, :n] for x in inps]) for k in inps[0]})
        with torch.no_grad():
            return model.generate(**batch, max_new_tokens=args.max_new_tokens,
                                  num_beams=1, do_sample=False, eos_token_id=-1)

    def parse(ids_row, size):
        txt = proc.batch_decode(ids_row, skip_special_tokens=False)[0]
        i = txt.find("</s>", 5)          # 前缀是 `</s><s><s><s>`，从 5 开始找真结束符
        if i > 0:
            txt = txt[:i + 4]
        try:
            v = proc.post_process_generation(txt, task=TASK_GROUND, image_size=size)[TASK_GROUND]
        except Exception:
            return {}
        agg = {}
        for lb, bb in zip(v["labels"], v["bboxes"]):
            lb = lb.strip()
            if lb:
                agg.setdefault(lb, []).append([round(x, 1) for x in bb])
        return agg

    def rec_of(it, agg, size):
        return {"id": it["id"], "shard": it["shard"], "path": it["path"], "img_wh": list(size),
                "grounding": agg, "n_phrase": len(agg),
                "n_box": sum(len(v) for v in agg.values())}

    items.sort(key=lambda it: len(it["dense"].split()))     # 长度分桶
    fo = open_append(out_f)
    t0 = time.perf_counter()
    n = 0
    try:
        for s in range(0, len(items), args.batch):
            chunk = items[s:s + args.batch]
            try:
                encs = [encode(it) for it in chunk]
                out = generate([e[0] for e in encs])
                for j, it in enumerate(chunk):
                    write_jsonl(fo, rec_of(it, parse(out[j:j + 1], encs[j][1]), encs[j][1]))
            except Exception:
                # batch 内任一图异常（坏图 / 显存抖动）时退化为逐图，避免整组丢失
                for it in chunk:
                    try:
                        inp, size = encode(it)
                        write_jsonl(fo, rec_of(it, parse(generate([inp]), size), size))
                    except Exception as e:
                        write_jsonl(fo, {"id": it["id"], "shard": it["shard"], "path": it["path"],
                                         "error": f"{type(e).__name__}: {e}"})
            n += len(chunk)
            if n % (args.batch * 10) == 0:
                el = time.perf_counter() - t0
                print(f"[gnd{args.shard}] {n}/{len(items)} {el/n*1000:.0f}ms/img "
                      f"eta {(len(items)-n)*el/n/60:.0f}min", flush=True)
    finally:
        fo.close()
    print(f"[gnd{args.shard}] DONE {n} in {(time.perf_counter()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
