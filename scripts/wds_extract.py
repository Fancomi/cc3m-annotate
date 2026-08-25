#!/usr/bin/env python3
"""把 webdataset tar 解包成「图片文件 + tsv 清单」，喂给阶段1。

产出布局刻意与 cc3m-tsv 一致，这样阶段 1~4 一行代码都不用改：

    <dst>/images/<tar 名>/<key>.jpg
    <dst>/_shards/<tar 名>.tsv        每行 `图片绝对路径 \t 原始 caption`

为什么解包而不是直接读 tar：阶段 2/3/4 都按 `path` 随机打开图片，
tar 内随机访问要么全表扫描要么把偏移塞进 path 字符串 —— 后者会让产出里的
`path` 变成非标准形式，下游拿到数据还得先学一套解析规则。磁盘换无歧义。

坐标语义因此与 cc3m 完全相同：**原图像素**，不做任何缩放。

断点续传：tsv 最后才原子落盘，所以「tsv 存在」= 该 tar 已完整解完，直接跳过。
"""
import argparse, io, json, os, sys, tarfile, time
from multiprocessing import Pool


def one_tar(job):
    """解一个 tar。返回 (tar 名, 写出图数, 各类跳过计数)。"""
    tar_path, dst, verify_jpg = job
    name = os.path.basename(tar_path)[:-4]          # cc12m-train-0000
    img_dir = os.path.join(dst, "images", name)
    tsv_fin = os.path.join(dst, "_shards", f"{name}.tsv")
    if os.path.exists(tsv_fin):
        return name, -1, {}                          # 已完成
    os.makedirs(img_dir, exist_ok=True)
    tsv_tmp = tsv_fin + ".tmp"

    skip = {"no_txt": 0, "not_success": 0, "empty_caption": 0, "suspect_jpg": 0}
    n = 0
    cur = {"key": None, "jpg": None, "ok": True, "cap": None}
    lines = []

    def flush():
        nonlocal n
        k = cur["key"]
        if k is None:
            return
        if cur["jpg"] is None or cur["cap"] is None:
            skip["no_txt"] += 1
        elif not cur["ok"]:
            skip["not_success"] += 1
        elif not cur["cap"].strip():
            skip["empty_caption"] += 1
        else:
            # 尾部没有 FFD9 只计数不丢弃：实测这类文件 PIL 仍能完整解码
            # （有的多一个填充字节，有的截断在 100KB 边界但扫描数据是全的）。
            # 真正读不出来的图，阶段1 会记 error 并在重跑时重试，不该在这里静默丢。
            if verify_jpg and b"\xff\xd9" not in cur["jpg"][-16:]:
                skip["suspect_jpg"] += 1
            p = os.path.join(img_dir, k + ".jpg")
            with open(p, "wb") as fo:
                fo.write(cur["jpg"])
            # caption 里的 tab / 换行会破坏 tsv，压成空格
            cap = " ".join(cur["cap"].split())
            lines.append(f"{p}\t{cap}\n")
            n += 1
        cur.update(key=None, jpg=None, ok=True, cap=None)

    # "r|" 是纯流式：不 seek、不建成员表，比 "r:" 快很多也省内存
    with tarfile.open(tar_path, "r|") as tf:
        for m in tf:
            if not m.isfile():
                continue
            key, _, ext = m.name.rpartition(".")
            key = os.path.basename(key)
            if key != cur["key"]:
                flush()
                cur["key"] = key
            if ext == "jpg":
                cur["jpg"] = tf.extractfile(m).read()
            elif ext == "json":
                try:
                    cur["ok"] = json.load(tf.extractfile(m)).get("status") == "success"
                except Exception:
                    cur["ok"] = False
            elif ext == "txt":
                cur["cap"] = tf.extractfile(m).read().decode("utf-8", "replace")
        flush()

    with open(tsv_tmp, "w") as fo:
        fo.writelines(lines)
    os.replace(tsv_tmp, tsv_fin)                     # 原子：此后该 tar 算已完成
    return name, n, {k: v for k, v in skip.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="wds 目录（含 *.tar）")
    ap.add_argument("--dst", required=True, help="输出根目录")
    ap.add_argument("--glob", default="*-train-*.tar")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help=">0 只解前 N 个 tar（冒烟）")
    ap.add_argument("--no-verify-jpg", action="store_true",
                    help="不统计 jpg 尾部 FFD9 异常（只计数，从不丢弃）")
    args = ap.parse_args()

    import glob as _g
    tars = sorted(_g.glob(os.path.join(args.src, args.glob)))
    if args.limit:
        tars = tars[:args.limit]
    os.makedirs(os.path.join(args.dst, "_shards"), exist_ok=True)
    print(f"[wds] {len(tars)} 个 tar -> {args.dst}  workers={args.workers}", flush=True)

    jobs = [(t, args.dst, not args.no_verify_jpg) for t in tars]
    total = skipped = 0
    agg = {}
    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        for i, (name, n, skip) in enumerate(pool.imap_unordered(one_tar, jobs), 1):
            if n < 0:
                skipped += 1
            else:
                total += n
            for k, v in skip.items():
                agg[k] = agg.get(k, 0) + v
            if i % 50 == 0 or i == len(tars):
                el = time.perf_counter() - t0
                eta = (len(tars) - i) * el / i / 60
                print(f"[wds] {i}/{len(tars)} tar  图 {total:,}  已跳过 tar {skipped}  "
                      f"{el/60:.1f}min  eta {eta:.1f}min  可疑/丢弃 {agg}", flush=True)
    print(f"[wds] DONE 图 {total:,}  跳过 tar {skipped}  可疑/丢弃 {agg}  "
          f"{(time.perf_counter()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
