#!/usr/bin/env python3
"""生成人工裁决 HTML —— 用于校准"gemma4 判 gemma4"那个精度下界的可信度。

与 legacy/stage3_verify/ 版本的区别：输入改为当前 pipeline 阶段4 产出的
verify_clean.jsonl（记录自带 path 主键，不再需要 crossfeed_shard 路径表），
判定模型文案从 Qwen 改为 gemma4。

统计设计（关键，别改成简单随机抽样）：
  verify 文件里的条目按 verdict 分 YES / NO 两层。若直接随机抽，
  NO 层占比低时样本里 NO 太少，估不准"判定器误否"的比例。故改为**分层抽样**：
  从 YES 层和 NO 层各独立抽 n 个，人工判断每个框里短语是否真的存在，得到
      a = P(真存在 | 自动判 YES)
      b = P(真存在 | 自动判 NO)     ← 判定器的误否率 = a 的对偶
  再按总体权重还原：
      真实精度 = (N_YES·a + N_NO·b) / (N_YES + N_NO)
  同时可算出判定器自身的准确率，回答"这个下界能不能信"。

页面内嵌裁剪图（就是喂给判定器的那张，含 12% padding），人工只需看图判
"短语在不在"，不显示判定器的裁决以免锚定；判完可展开对比。
结果存 localStorage 防丢，可导出 JSON 交回脚本做统计。

用法: python make_adjudicate_html.py --verify out/verify_clean.jsonl --out adjudicate_clean.html
"""
import argparse, base64, glob, html, io, json, os, random


def crop_b64(path, box, pad=0.12, maxside=320):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    W, H = im.size
    x0, y0, x1, y1 = box
    pw, ph = (x1 - x0) * pad, (y1 - y0) * pad
    c = im.crop((max(0, x0 - pw), max(0, y0 - ph), min(W, x1 + pw), min(H, y1 + ph)))
    if c.width < 4 or c.height < 4:
        return None, None
    # 同时给一张全图缩略，标出框位置 —— 人工判断"框位对不对"需要上下文
    thumb = im.copy()
    ts = 260 / max(thumb.size)
    thumb = thumb.resize((max(1, int(thumb.width * ts)), max(1, int(thumb.height * ts))))
    from PIL import ImageDraw
    d = ImageDraw.Draw(thumb)
    d.rectangle([x0 * ts, y0 * ts, x1 * ts, y1 * ts], outline=(255, 60, 60), width=3)
    if max(c.size) > maxside:
        s = maxside / max(c.size)
        c = c.resize((max(1, int(c.width * s)), max(1, int(c.height * s))))
    def enc(i):
        b = io.BytesIO(); i.save(b, format="JPEG", quality=85)
        return base64.b64encode(b.getvalue()).decode()
    return enc(c), enc(thumb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", required=True,
                    help="阶段4 产出的 verify jsonl，如 out/verify_clean.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-stratum", type=int, default=100, help="YES / NO 各抽多少")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--auto-precision", type=float, default=None,
                    help="自动判定给出的精度下界（%%），默认取 verify 里 YES 比例")
    ap.add_argument("--save-url", default="/save",
                    help="落盘端点。默认同源 /save，要求页面必须从 serve_review.py 打开；"
                         "若可能从别的文件服务（如 bdhttp3.py）打开，填绝对地址 "
                         "http://<开发机>:8900/save，跨源也能存")
    args = ap.parse_args()
    save_url = args.save_url
    load_url = save_url[:-len("/save")] + "/load" if save_url.endswith("/save") else "/load"

    rows = [json.loads(l) for l in open(args.verify)]
    if args.auto_precision is None:
        ys = sum(1 for r in rows if r.get("verdict") == "YES")
        args.auto_precision = ys / len(rows) * 100 if rows else 0

    yes = [r for r in rows if r["verdict"] == "YES"]
    no = [r for r in rows if r["verdict"] == "NO"]
    N_YES, N_NO = len(yes), len(no)
    random.seed(args.seed)
    random.shuffle(yes); random.shuffle(no)
    samp = yes[:args.per_stratum] + no[:args.per_stratum]
    random.shuffle(samp)   # 打乱顺序，避免人工按块判断产生偏差

    items = []
    for r in samp:
        p = r.get("path")
        if not p:
            continue
        try:
            cb, tb = crop_b64(p, r["box"])
        except Exception:
            continue
        if not cb:
            continue
        items.append({"path": r["path"], "phrase": r["phrase"], "verdict": r["verdict"],
                      "crop_frac": r.get("crop_frac"), "crop": cb, "thumb": tb})
    print(f"抽样 {len(items)} 个（YES 层 {sum(1 for i in items if i['verdict']=='YES')} / "
          f"NO 层 {sum(1 for i in items if i['verdict']=='NO')}）")

    payload = json.dumps({"N_YES": N_YES, "N_NO": N_NO, "items": items}, ensure_ascii=False)

    H = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>人工裁决 · 校准判定器</title>
<style>
:root{{--bg:#0f1115;--fg:#e8e8e8;--dim:#98a0aa;--card:#181b21;--line:#2b3038;
      --ok:#4ade80;--bad:#ff6b6b;--acc:#2ec4ff;--warn:#fbbf24}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);
  font:14px/1.55 -apple-system,"Segoe UI",Roboto,sans-serif}}
header{{padding:14px 22px;border-bottom:1px solid var(--line);position:sticky;top:0;
  background:var(--bg);z-index:20}}
h1{{margin:0 0 4px;font-size:16px}}
.note{{background:#2a2213;border-left:3px solid var(--warn);padding:10px 14px;margin:12px 22px;
  border-radius:4px;font-size:13px;color:#fde68a}}
.note b{{color:var(--warn)}}
#stats{{display:flex;gap:22px;flex-wrap:wrap;font-size:13px;color:var(--dim);margin-top:6px}}
#stats b{{color:var(--fg);font-size:15px}}
.big{{color:var(--acc)!important}}
.wrap{{max-width:900px;margin:0 auto;padding:0 22px 90px}}
.q{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px;
  margin:16px 0;display:none}}
.q.on{{display:block}}
.ph{{font-size:19px;font-weight:600;margin:4px 0 12px}}
.ph span{{color:var(--dim);font-size:13px;font-weight:400}}
.imgs{{display:flex;gap:14px;align-items:flex-start;flex-wrap:wrap}}
.imgs figure{{margin:0}}
.imgs img{{display:block;border-radius:4px;border:1px solid var(--line)}}
.imgs figcaption{{color:var(--dim);font-size:11px;margin-top:4px;text-align:center}}
.btns{{display:flex;gap:10px;margin-top:16px;flex-wrap:wrap}}
button{{font:inherit;padding:9px 20px;border-radius:6px;border:1px solid var(--line);
  background:#20242c;color:var(--fg);cursor:pointer}}
button:hover{{border-color:var(--acc)}}
button.y{{background:#14351f;border-color:#2f6b42}} button.n{{background:#3a1518;border-color:#7a2b30}}
button.s{{background:#26282d}}
.kb{{color:var(--dim);font-size:12px;margin-top:10px}}
.rev{{margin-top:12px;padding-top:10px;border-top:1px dotted var(--line);font-size:13px;
  color:var(--dim);display:none}}
.rev.on{{display:block}}
.v{{font-weight:700}} .v.YES{{color:var(--ok)}} .v.NO{{color:var(--bad)}}
#done{{display:none;background:var(--card);border:1px solid var(--line);border-radius:8px;
  padding:20px;margin:16px 0}}
#done.on{{display:block}}
table{{border-collapse:collapse;width:100%;margin:10px 0;font-size:13px}}
th,td{{border:1px solid var(--line);padding:7px 10px;text-align:left}}
th{{background:#20242c;color:var(--dim);font-weight:600}}
.foot{{position:fixed;bottom:0;left:0;right:0;background:#12151a;border-top:1px solid var(--line);
  padding:10px 22px;display:flex;gap:12px;align-items:center;z-index:20}}
.bar{{flex:1;height:6px;background:#242830;border-radius:3px;overflow:hidden}}
.bar i{{display:block;height:100%;background:var(--acc);width:0}}
code{{background:#20242c;padding:1px 5px;border-radius:3px;font-size:12px}}
</style></head><body>
<header><h1>人工裁决 · 校准自动判定器</h1>
<div id="stats"></div></header>
<div class="note">
<b>这个页面在做什么：</b>之前 docs/RESULT.md 里的精度数字是<b>用 gemma4 判 gemma4 自己产出的短语</b>，
存在确认偏误，只能算下界。这里请你对同样的 (短语, 框) 对做人工裁决，用来估出真实精度。<br>
<b>抽样方式是分层的</b>：从 gemma4 判 YES 的 {N_YES} 对里抽 {args.per_stratum} 个，
判 NO 的 {N_NO} 对里也抽 {args.per_stratum} 个 —— 因为 NO 只占总体
{N_NO/(N_YES+N_NO)*100:.1f}%，随机抽会导致 NO 样本太少估不准。统计时会按总体权重还原。<br>
<b>判断标准</b>：红框圈出的区域里，能否清楚看到这个短语所指的东西？框大致对上即可，不必像素级精确。
<b>页面刻意不显示 gemma4 的裁决</b>，避免锚定；判完每题可展开对照。<br>
<b>保存</b>：每答一题自动 POST 到 <code>{save_url}</code> 写进开发机磁盘（页脚显示"已存盘 N"）。
关页面、换端口、换浏览器都能续答。若页脚变红、页首出现红色告警条，说明<b>没存上</b>，
此时只有 localStorage —— 换成 <code>scripts/serve_review.py</code> 提供的地址重开本页即可补存。
</div>
<div class="wrap"><div id="qs"></div>
<div id="done"><h2 style="margin-top:0">裁决完成</h2><div id="result"></div>
<div class="btns"><button class="s" onclick="dl()">导出 JSON</button>
<button class="s" onclick="if(confirm('清空所有裁决重来? 磁盘上的备份快照仍保留')){{localStorage.removeItem(KEY);ans={{}};push();setTimeout(()=>location.reload(),400)}}">重置</button></div>
</div></div>
<div class="foot"><span id="prog" style="color:var(--dim);font-size:13px"></span>
<div class="bar"><i id="fill"></i></div>
<span id="sv" style="font-size:12px;color:var(--dim)">—</span>
<button class="s" onclick="jump(-1)">← 上一题</button></div>
<script>
const DATA = {payload};
const SAVE_URL = {save_url!r}, LOAD_URL = {load_url!r};
const KEY = "cc3m_human_adj_v1";
let ans = JSON.parse(localStorage.getItem(KEY) || "{{}}");
let cur = 0;
const items = DATA.items, qs = document.getElementById('qs');

items.forEach((it,i)=>{{
  const d=document.createElement('div'); d.className='q'; d.id='q'+i;
  d.innerHTML=`<div class="ph">${{esc(it.phrase)}} <span>· 框占全图 ${{(it.crop_frac*100).toFixed(2)}}%</span></div>
  <div class="imgs">
    <figure><img src="data:image/jpeg;base64,${{it.crop}}"><figcaption>裁剪（喂给判定器的就是这张）</figcaption></figure>
    <figure><img src="data:image/jpeg;base64,${{it.thumb}}"><figcaption>全图 · 红框为该区域</figcaption></figure>
  </div>
  <div class="btns">
    <button class="y" onclick="rec(${{i}},'YES')">能看到 (Y)</button>
    <button class="n" onclick="rec(${{i}},'NO')">看不到 (N)</button>
    <button class="s" onclick="rec(${{i}},'UNSURE')">说不清 (U)</button>
  </div>
  <div class="kb">键盘：Y / N / U · ← 回退</div>
  <div class="rev" id="r${{i}}"></div>`;
  qs.appendChild(d);
}});

function esc(s){{return s.replace(/[&<>]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;'}})[c])}}
function key(it){{return it.path+"|"+it.phrase}}

function rec(i,v){{
  ans[key(items[i])]=v; localStorage.setItem(KEY,JSON.stringify(ans));
  push();
  const r=document.getElementById('r'+i);
  const agree = (v===items[i].verdict);
  r.className='rev on';
  r.innerHTML=`你判：<span class="v ${{v}}">${{v}}</span> · 判定器(gemma4)判：<span class="v ${{items[i].verdict}}">${{items[i].verdict}}</span>`
    + (v==='UNSURE' ? ' · 该项不计入统计' : (agree?' · <span style="color:var(--ok)">一致</span>':' · <span style="color:var(--bad)">不一致</span>'));
  setTimeout(()=>{{jump(1)}},420);
}}

function show(){{
  document.querySelectorAll('.q').forEach(e=>e.classList.remove('on'));
  const done = Object.keys(ans).length;
  if(cur>=items.length){{
    document.getElementById('done').classList.add('on'); render();
  }} else {{
    document.getElementById('done').classList.remove('on');
    document.getElementById('q'+cur).classList.add('on');
    window.scrollTo({{top:0}});
  }}
  document.getElementById('prog').textContent=`${{Math.min(cur+1,items.length)}} / ${{items.length}} · 已判 ${{done}}`;
  document.getElementById('fill').style.width=(done/items.length*100)+'%';
  stats();
}}
function jump(d){{ cur=Math.max(0,Math.min(items.length,cur+d)); show(); }}

function calc(){{
  let ay=[0,0], an=[0,0];   // [真存在数, 已判数]
  items.forEach(it=>{{
    const v=ans[key(it)];
    if(!v||v==='UNSURE') return;
    const t=(v==='YES')?1:0;
    if(it.verdict==='YES'){{ ay[0]+=t; ay[1]++; }} else {{ an[0]+=t; an[1]++; }}
  }});
  const a = ay[1]? ay[0]/ay[1] : null;
  const b = an[1]? an[0]/an[1] : null;
  let truePrec=null;
  if(a!==null && b!==null)
    truePrec = (DATA.N_YES*a + DATA.N_NO*b)/(DATA.N_YES+DATA.N_NO);
  // 判定器自身准确率（按总体加权）
  let acc=null;
  if(a!==null && b!==null)
    acc = (DATA.N_YES*a + DATA.N_NO*(1-b))/(DATA.N_YES+DATA.N_NO);
  return {{a,b,ay,an,truePrec,acc}};
}}
function stats(){{
  const c=calc(), s=document.getElementById('stats');
  const pct=v=>v===null?'—':(v*100).toFixed(1)+'%';
  s.innerHTML=`<span>YES 层已判 <b>${{c.ay[1]}}</b>/${{DATA.items.filter(i=>i.verdict==='YES').length}}</span>
    <span>NO 层已判 <b>${{c.an[1]}}</b>/${{DATA.items.filter(i=>i.verdict==='NO').length}}</span>
    <span>a=P(真存在|判YES) <b>${{pct(c.a)}}</b></span>
    <span>b=P(真存在|判NO) <b>${{pct(c.b)}}</b></span>
    <span>还原真实精度 <b class="big">${{pct(c.truePrec)}}</b></span>
    <span>判定器准确率 <b>${{pct(c.acc)}}</b></span>`;
}}
function render(){{
  const c=calc(), pct=v=>v===null?'—':(v*100).toFixed(1)+'%';
  document.getElementById('result').innerHTML=`
  <table>
  <tr><th>量</th><th>值</th><th>含义</th></tr>
  <tr><td>a = P(真存在 | gemma4 判 YES)</td><td><b>${{pct(c.a)}}</b></td><td>n=${{c.ay[1]}}，判定器说有、确实有的比例</td></tr>
  <tr><td>b = P(真存在 | gemma4 判 NO)</td><td><b>${{pct(c.b)}}</b></td><td>n=${{c.an[1]}}，判定器误否的比例</td></tr>
  <tr><td>还原真实精度</td><td><b class="big">${{pct(c.truePrec)}}</b></td>
      <td>(${{DATA.N_YES}}·a + ${{DATA.N_NO}}·b) / ${{DATA.N_YES+DATA.N_NO}}</td></tr>
  <tr><td>自动判定给出的值</td><td>{args.auto_precision:.1f}%</td><td>docs/RESULT.md 里那个下界</td></tr>
  <tr><td>判定器自身准确率</td><td><b>${{pct(c.acc)}}</b></td><td>与人工一致的比例（总体加权）</td></tr>
  </table>
  <p style="color:var(--dim);font-size:13px">
  若"还原真实精度" &gt; {args.auto_precision:.1f}%，说明自动判定偏保守（确认偏误没有想象的严重）；
  若明显 &lt; {args.auto_precision:.1f}%，说明 gemma4 在给自己放水，那个下界不可用。<br>
  导出的 JSON 交给 <code>scripts/apply_human_adj.py</code> 可写回 docs/RESULT.md。</p>`;
}}
let svTimer=null, svState='';
function setSv(t,c){{ const e=document.getElementById('sv'); if(e){{e.textContent=t; e.style.color=c;}} }}
function payload(){{
  return {{N_YES:DATA.N_YES,N_NO:DATA.N_NO,calc:calc(),
    answers:items.map(it=>({{path:it.path,phrase:it.phrase,auto:it.verdict,
      human:ans[key(it)]||null,crop_frac:it.crop_frac}}))}};
}}
// 落盘到开发机磁盘。localStorage 绑定 origin，换端口/浏览器就丢；
// 服务端落盘才能真正保住，且免去把 JSON 从本地机传回开发机。
// Content-Type 用 text/plain 是有意的：这样跨源 POST 属于"简单请求"，不触发
// OPTIONS 预检（serve_review.py 不处理 OPTIONS）。服务端只做 json.loads，不看类型。
function push(){{
  clearTimeout(svTimer);
  svTimer=setTimeout(()=>{{
    setSv('保存中…','var(--dim)');
    fetch(SAVE_URL,{{method:'POST',headers:{{'Content-Type':'text/plain'}},
      body:JSON.stringify(payload())}})
      .then(r=>r.ok?r.json():Promise.reject(r.status))
      .then(j=>{{ setSv(j.ok?('已存盘 '+j.count):'存盘失败','var(--ok)'); warn(false); }})
      .catch(e=>{{ setSv('未落盘('+e+')','var(--bad)'); warn(true); }});
  }},250);
}}
// 存盘失败必须显眼 —— 之前只在页脚显示一行小字，标了几十题才发现没存上
function warn(on){{
  let b=document.getElementById('svwarn');
  if(!on){{ if(b) b.remove(); return; }}
  if(b) return;
  b=document.createElement('div'); b.id='svwarn'; b.className='note';
  b.style.cssText='background:#3a1518;border-left-color:var(--bad);color:#ffc9c9;'
    +'position:sticky;top:0;z-index:30;margin:0';
  b.innerHTML='<b>裁决没有存到开发机磁盘</b>（只在这个浏览器的 localStorage 里）。'
    +'落盘地址 <code>'+SAVE_URL+'</code> 不可达 —— 通常是页面不是从 '
    +'<code>scripts/serve_review.py</code> 打开的。请换成它提供的地址重开本页，'
    +'已答的题会自动带过去。';
  document.body.insertBefore(b, document.body.firstChild);
}}
function pull(){{
  return fetch(LOAD_URL).then(r=>r.json()).then(j=>{{
    let n=0;
    (j.answers||[]).forEach(x=>{{
      if(x.human){{ ans[x.path+"|"+x.phrase]=x.human; n++; }}
    }});
    if(n){{ localStorage.setItem(KEY,JSON.stringify(ans)); setSv('已从磁盘恢复 '+n,'var(--ok)'); }}
  }}).catch(()=>{{}});
}}
function dl(){{
  const out={{N_YES:DATA.N_YES,N_NO:DATA.N_NO,calc:calc(),
    answers:items.map(it=>({{path:it.path,phrase:it.phrase,auto:it.verdict,
      human:ans[key(it)]||null,crop_frac:it.crop_frac}}))}};
  const b=new Blob([JSON.stringify(out,null,1)],{{type:'application/json'}});
  const a=document.createElement('a'); a.href=URL.createObjectURL(b);
  a.download='human_adjudication.json'; a.click();
}}
document.addEventListener('keydown',e=>{{
  if(cur>=items.length) return;
  const k=e.key.toLowerCase();
  if(k==='y') rec(cur,'YES'); else if(k==='n') rec(cur,'NO');
  else if(k==='u') rec(cur,'UNSURE'); else if(e.key==='ArrowLeft') jump(-1);
}});
// 启动：先从磁盘拉已存裁决（跨端口/浏览器续答），再跳到第一个未判的
pull().then(()=>{{
  cur=0;
  while(cur<items.length && ans[key(items[cur])]) cur++;
  show();
}});
</script></body></html>"""

    with open(args.out, "w") as f:
        f.write(H)
    print(f"-> {args.out}  ({os.path.getsize(args.out)/2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
