"""Generate M11 CSV+plots (pure stdlib). Three labeled PNGs + block sweep."""
import json
import struct
import zlib
from pathlib import Path

OUT = Path("/opt/nano-vllm/benchmarks/results")
data = json.loads((OUT / "m11_engine_sparse.json").read_text())
methods = data["methods"]
names = list(methods.keys())
sweep = json.loads((OUT / "block_sweep.json").read_text())["block_size_sweep"]

# ---- tiny 5x7 font for labels --------------------------------------------
FONT = {
 "0":["01110","10011","10111","11001","10011","10011","01110"],
 "1":["01100","00100","00100","00100","00100","00100","01110"],
 "2":["01110","10001","00001","00010","00100","01000","11111"],
 "3":["11111","00010","00100","00010","00001","10001","01110"],
 "4":["00010","00110","01010","10010","11111","00010","00010"],
 "5":["11111","10000","11110","00001","00001","10001","01110"],
 "6":["00110","01000","10000","11110","10001","10001","01110"],
 "7":["11111","00001","00010","00100","01000","01000","01000"],
 "8":["01110","10001","10001","01110","10001","10001","01110"],
 "9":["01110","10001","10001","01111","00001","00010","01100"],
 ".":["00000","00000","00000","00000","00000","00110","00110"],
 "%":["11001","11010","00010","00100","01011","10011","00000"],
 "m":["00000","00000","01100","10010","11010","10010","10010"],
 "s":["00000","00000","01111","10000","01110","00001","11110"],
 "e":["00000","00000","01110","10001","11111","10000","01110"],
 "a":["00000","00000","01110","00001","01111","10001","01111"],
 "r":["00000","00000","01000","10110","11000","10000","10111"],
 "t":["00000","00000","00100","11111","00100","00100","00011"],
 "u":["00000","00000","10001","10001","10001","10011","01101"],
}


def text_w(px, w, h, x, y, s, color=(0, 0, 0), scale=2):
    cx = x
    for ch in s:
        glyph = FONT.get(ch)
        if glyph:
            for ry, row in enumerate(glyph):
                for rx, c in enumerate(row):
                    if c == "1":
                        rect(px, w, h, cx + rx*scale, y + ry*scale,
                             cx + (rx+1)*scale, y + (ry+1)*scale, color)
            cx += 6 * scale
        elif ch == " ":
            cx += 3 * scale


def write_png(path, width, height, pixels):
    raw = b"".join(b"\x00" + pixels[y*width*3:(y+1)*width*3] for y in range(height))
    def chunk(tag, body):
        return struct.pack(">I", len(body)) + tag + body + \
               struct.pack(">I", zlib.crc32(tag + body) & 0xffffffff)
    px = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) \
         + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    Path(path).write_bytes(px)


def canvas(w, h, bg=(255, 255, 255)):
    return bytearray(bg * (w * h))


def rect(px, w, h, x0, y0, x1, y1, color):
    for y in range(max(0, y0), min(h, y1)):
        b = y*w*3
        for x in range(max(0, x0), min(w, x1)):
            px[b+x*3:b+x*3+3] = bytes(color)


BLUE=(70,110,180); RED=(180,40,40); GRAY=(150,150,150); ORANGE=(230,150,50)


def chart(path, title, bars, labels, maxv, colors, value_suffix=""):
    W,H = 820,520
    px = canvas(W,H)
    base = H-70; top = 50
    rect(px,W,H,60,top,60,base,(0,0,0))      # y axis
    rect(px,W,H,60,base,W-10,base,(0,0,0))   # x axis
    n=len(labels); bw=max(18,(W-120)//(n*2)-10)
    for i,vals in enumerate(bars):
        x0 = 70 + i*(W-120)//n
        for j,v in enumerate(vals):
            bh=int((base-top)*(v/maxv if maxv else 0))
            rect(px,W,H,x0+j*(bw+2),base-bh,x0+j*(bw+2)+bw,base,colors[j])
            text_w(px,W,H,x0+j*(bw+2),base-bh-16,f"{v:.2f}{value_suffix}",colors[j],1)
    text_w(px,W,H,70,15,title,(20,20,20),2)
    for i,lab in enumerate(labels):
        x0 = 70 + i*(W-120)//n
        text_w(px,W,H,x0,base+8,lab,(40,40,40),1)
    write_png(path,W,H,px)


ratios=[(methods[n].get("selection_sample_layer14") or {}).get("selected_token_ratio",0) for n in names]
recall=[(methods[n].get("selection_sample_layer14") or {}).get("work",{}).get("pre_window_recall",0) for n in names]
chart(OUT/"m11_quality_tradeoff.png","selected ratio & recall vs oracle",
      list(zip(ratios,recall)),names,1.0,[BLUE,RED])
ttft=[methods[n]["ttft_ms"]/1000 for n in names]
tp=[methods[n]["decode_p50_ms"]/1000 for n in names]
chart(OUT/"m11_latency_breakdown.png","TTFT(s) & p50 TPOT(s)",list(zip(ttft,tp)),
      names,max(ttft+tp)*1.15,[GRAY,ORANGE])
mem=data["memory"]; hist=methods["query_guided"]["cpu_history_bytes"]/1e6
chart(OUT/"m11_memory_comparison.png","MiB",[[mem["dense_paged_kv_bytes"]/1e6],
      [mem["sparse_paged_kv_bytes"]/1e6],[hist]],["dense_paged","sparse_paged","cpu_history"],
      max(mem["dense_paged_kv_bytes"]/1e6,hist)*1.15,[(40,90,140)])
print("plots ok")
