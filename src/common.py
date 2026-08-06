"""公共模块：图像编码、VLM 客户端、断点续传、分片、prompt 常量。

所有阶段脚本共用。改这里会影响全部阶段，改完跑 bash run/run_all.sh --smoke 验证。
"""
import base64, glob, io, json, os, re, time

# ---------------- caption prompt ----------------
# 两级 prompt。DENSE 是整条链路的源头：它决定 grounding 能定位到哪些短语，
# 所以刻意要求"逐个点名具体名词"，而不是写通顺的描述。
SHORT = "Describe this image in one concise sentence."
DENSE = ("Describe this image in one dense paragraph. Explicitly name every distinct visible "
         "object, material and body part using concrete nouns. Be exhaustive and factual.")

# Florence-2 短语定位任务标记
TASK_GROUND = "<CAPTION_TO_PHRASE_GROUNDING>"

# 整体指代类短语：ground 到全图，对区域标注无价值。
# 阶段3（清洗）与阶段4（校验）共用这一份 —— 否则校验会去烧算力验清洗本就要删的短语。
VAGUE = re.compile(
    r"^(the|this|a|an)?\s*(entire|whole|overall)?\s*"
    r"(image|photo|photograph|picture|scene|view|frame|composition|background|foreground)s?$",
    re.I)

# 判断短语是否忠实摘抄自 caption 时用的停用词表
STOP = frozenset("a an the of in on at to for with and or its his her their this that these those".split())


def norm_phrase(s):
    """小写、去标点连字符、压空格 —— 使 `black-framed` 能匹配 caption 里的 `black framed`。"""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def content_words(s):
    return [w for w in norm_phrase(s).split() if w not in STOP]


def b64(img, maxside=768, quality=90):
    """PIL 图 -> base64 JPEG。长边缩到 maxside 以控制请求体积。

    重采样核固定用 LANCZOS：全量 caption 是用它跑的，换核会让补跑的样本与
    历史产出不一致。
    """
    from PIL import Image
    im = img.copy()
    if max(im.size) > maxside:
        s = maxside / max(im.size)
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def clear_proxy():
    """清代理环境变量 —— 代理会拦截 127.0.0.1 上的 sglang 端点。"""
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(k, None)


def make_clients(urls, timeout=600):
    """建 OpenAI 兼容客户端列表，配合 round_robin 轮询多个 sglang 实例。"""
    clear_proxy()
    from openai import OpenAI
    if isinstance(urls, str):
        urls = [u.strip() for u in urls.split(",") if u.strip()]
    return [OpenAI(api_key="x", base_url=u, timeout=timeout) for u in urls]


def round_robin(items):
    """无锁轮询迭代器（itertools.count 的 next 是原子的，多线程安全）。"""
    import itertools
    c = itertools.count()
    return lambda: items[next(c) % len(items)]


def ask_vlm(clients, model, img_b64, prompt, max_tokens=320, retries=4, pick=None):
    """单轮图文问答，失败换端点重试。

    enable_thinking=False 必须带：这些模型默认开推理链，会把"The user wants..."
    当正文吐出来并吃满 token 预算（实测拿不到 caption）。

    重试+换端点必须带：8 实例长跑时个别 sglang 实例会被 watchdog 重启，
    期间路由到它的请求直接失败。首轮全量 caption 没有重试，8.3% 的图
    （238874/2894191）落成了 APIConnectionError。

    clients 可以是客户端列表，也可以是单个客户端（自动包成单元素列表）。
    """
    if not isinstance(clients, (list, tuple)):
        clients = [clients]
    if pick is None:
        pick = round_robin(clients)
    last = None
    for k in range(retries):
        try:
            r = pick().chat.completions.create(
                model=model, max_tokens=max_tokens, temperature=0.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": prompt}]}])
            txt = r.choices[0].message.content or ""
            if "</think>" in txt:
                txt = txt.split("</think>", 1)[1]
            return txt.strip()
        except Exception as e:
            last = e
            time.sleep(2 ** k)
    raise last


# ---------------- jsonl 读写 ----------------
def iter_jsonl(path):
    """逐行读 jsonl，跳过坏行（批处理中途被 kill 可能留半截行）。"""
    with open(path) as f:
        for l in f:
            try:
                yield json.loads(l)
            except Exception:
                continue


def iter_shards(d, pattern="shard*.jsonl"):
    """按文件名排序遍历某目录下所有分片的所有记录。"""
    for f in sorted(glob.glob(os.path.join(d, pattern))):
        yield from iter_jsonl(f)


def write_jsonl(fo, rec):
    fo.write(json.dumps(rec, ensure_ascii=False) + "\n")


def open_append(path):
    """行缓冲追加写 —— 进程被 kill 时已写的行不会丢，配合 load_done 续跑。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return open(path, "a", buffering=1)


# ---------------- 分片 / 断点续传 ----------------
# 记录主键是 path：id 只是 tsv 内的行号，576 个 tsv 之间必然撞号，
# 单用 id 做去重会把几百张不同的图误判成同一张。
def rec_key(r):
    return r["path"]


def is_ok(r):
    """记录是否算「成功完成」。error 占位行不算，重跑时会被重试。

    首轮全量 caption 把 error 行也当已完成，导致 8.3% 的图永久缺 dense。
    """
    return "error" not in r


def load_done(path, key=rec_key, ok=is_ok):
    """读已有输出，返回已成功完成的键集合。文件不存在返回空集。"""
    if not os.path.exists(path):
        return set()
    return {key(r) for r in iter_jsonl(path) if ok(r)}


def take_shard(records, shard, num_shards, key=lambda r: r["id"]):
    """确定性分片：按 key 取模。id 在各 tsv 内连续，故分片天然均衡。"""
    return [r for r in records if key(r) % num_shards == shard]
