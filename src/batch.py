"""并发批处理骨架：分片 + 断点续传 + 进度打点。

阶段脚本只需提供「取任务列表」和「处理单条」两个函数，其余交给这里。
续传只跳过成功条目（`common.is_ok`），error 占位行会被重试。
"""
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import islice

from common import load_done, open_append, rec_key, write_jsonl


def report(tag, n, total, t0, unit="s/it"):
    el = time.perf_counter() - t0
    per = el / max(1, n)
    eta = (total - n) * per / 60
    v = f"{per*1000:.0f}ms" if per < 1 else f"{per:.2f}s"
    print(f"[{tag}] {n}/{total} {v}/it eta {eta:.1f}min", flush=True)


def run_pool(items, work, out_path, tag, workers=16, every=200, key=rec_key):
    """并发跑 work(item)->rec，结果追加写 out_path。返回实际处理条数。

    已完成的条目在此过滤，所以重跑同一条命令即为续跑。

    有界提交（别改回 ex.map）：`ex.map` 会先把全部任务 submit 完才产出第一个结果。
    任务量小时无所谓，但全量校验有 3000 万条 —— 实测跑 12 分钟，磁盘上一行都没有、
    结果全攒在 Future 里，内存以 4 GiB/min 涨，中途挂掉全部白跑。改成只保留
    workers*8 个在飞的任务，完成即落盘，内存恒定、进度可见、随时可续。
    代价是输出顺序变成完成顺序而非输入顺序 —— 下游一律按主键索引，不依赖顺序。
    """
    done = load_done(out_path, key)
    todo = [it for it in items if key(it) not in done]
    print(f"[{tag}] 总 {len(items)} 已完成 {len(done)} 待跑 {len(todo)}", flush=True)
    if not todo:
        return 0
    fo = open_append(out_path)
    t0 = time.perf_counter()
    n = 0
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            src = iter(todo)
            fs = {ex.submit(work, it) for it in islice(src, max(workers * 8, 256))}
            while fs:
                ready, fs = wait(fs, return_when=FIRST_COMPLETED)
                for f in ready:
                    write_jsonl(fo, f.result())
                    n += 1
                    if n % every == 0:
                        report(tag, n, len(todo), t0)
                fs |= {ex.submit(work, it) for it in islice(src, len(ready))}
    finally:
        fo.close()
    print(f"[{tag}] DONE {n} in {(time.perf_counter()-t0)/60:.1f}min", flush=True)
    return n


def run_serial(items, work, out_path, tag, every=50, key=rec_key):
    """串行版（GPU 独占型任务用，如 Florence-2 推理）。work 可返回单条或列表。"""
    done = load_done(out_path, key)
    todo = [it for it in items if key(it) not in done]
    print(f"[{tag}] 总 {len(items)} 已完成 {len(done)} 待跑 {len(todo)}", flush=True)
    if not todo:
        return 0
    fo = open_append(out_path)
    t0 = time.perf_counter()
    n = 0
    try:
        for it in todo:
            out = work(it)
            for rec in (out if isinstance(out, list) else [out]):
                write_jsonl(fo, rec)
                n += 1
            if n % every < len(out if isinstance(out, list) else [out]):
                report(tag, n, len(todo), t0)
    finally:
        fo.close()
    print(f"[{tag}] DONE {n} in {(time.perf_counter()-t0)/60:.1f}min", flush=True)
    return n
