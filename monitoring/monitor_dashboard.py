#!/usr/bin/env python3
"""GPU PC リソースモニタリング ダッシュボード
CPU/GPU使用率・温度・ネットワーク使用率を収集し、CSVに記録しつつWebダッシュボードを提供する。
標準ライブラリのみ。 ポート 43350。
"""
import json, os, time, threading, subprocess, glob
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from collections import deque

PORT = 43350
INTERVAL = 5                      # サンプリング間隔(秒)
MAXLEN = 2160                     # メモリ保持: 3時間分(5s刻み)
BASE = os.path.expanduser("~/monitor")
CSV = os.path.join(BASE, "metrics.csv")
HISTORY = deque(maxlen=MAXLEN)
LOCK = threading.Lock()


def read_cpu():
    with open("/proc/stat") as f:
        vals = list(map(int, f.readline().split()[1:]))
    idle = vals[3] + vals[4]
    return idle, sum(vals)


def read_net():
    rx = tx = 0
    with open("/proc/net/dev") as f:
        for line in f.readlines()[2:]:
            iface, data = line.split(":")
            if iface.strip() == "lo":
                continue
            d = data.split()
            rx += int(d[0]); tx += int(d[8])
    return rx, tx


def read_cpu_temp():
    temps = []
    for f in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            temps.append(int(open(f).read().strip()) / 1000.0)
        except Exception:
            pass
    return round(max(temps), 1) if temps else None


def read_gpu():
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], timeout=5).decode().strip()
        u, t, mu, mt = [x.strip() for x in out.split(",")]
        return float(u), float(t), float(mu), float(mt)
    except Exception:
        return None, None, None, None


def load_csv():
    """再起動後に直近データを復元"""
    if not os.path.exists(CSV):
        return
    try:
        lines = open(CSV).read().splitlines()[1:][-MAXLEN:]
        for ln in lines:
            p = ln.split(",")
            if len(p) < 9:
                continue
            def fv(x):
                return None if x in ("", "None") else float(x)
            HISTORY.append({"t": int(p[0]), "cpu": fv(p[1]), "cpu_temp": fv(p[2]),
                            "gpu": fv(p[3]), "gpu_temp": fv(p[4]),
                            "gpu_mem": fv(p[5]), "gpu_mem_total": fv(p[6]),
                            "net_rx": fv(p[7]), "net_tx": fv(p[8])})
    except Exception as e:
        print("csv load skip:", e)


def collector():
    last_cpu = read_cpu()
    last_net = read_net()
    last_t = time.time()
    if not os.path.exists(CSV):
        with open(CSV, "w") as f:
            f.write("timestamp,cpu_pct,cpu_temp,gpu_pct,gpu_temp,"
                    "gpu_mem_used,gpu_mem_total,net_rx_kbps,net_tx_kbps\n")
    while True:
        time.sleep(INTERVAL)
        now = time.time()
        dt = max(now - last_t, 0.001)
        idle, total = read_cpu()
        di, dtot = idle - last_cpu[0], total - last_cpu[1]
        cpu_pct = round(100 * (1 - di / dtot), 1) if dtot > 0 else 0.0
        last_cpu = (idle, total)
        rx, tx = read_net()
        net_rx = round((rx - last_net[0]) / dt / 1024, 1)
        net_tx = round((tx - last_net[1]) / dt / 1024, 1)
        last_net = (rx, tx)
        last_t = now
        cpu_temp = read_cpu_temp()
        gpu, gpu_temp, gmu, gmt = read_gpu()
        s = {"t": int(now), "cpu": cpu_pct, "cpu_temp": cpu_temp,
             "gpu": gpu, "gpu_temp": gpu_temp, "gpu_mem": gmu,
             "gpu_mem_total": gmt, "net_rx": net_rx, "net_tx": net_tx}
        with LOCK:
            HISTORY.append(s)
        try:
            with open(CSV, "a") as f:
                f.write("{t},{cpu},{cpu_temp},{gpu},{gpu_temp},{gpu_mem},"
                        "{gpu_mem_total},{net_rx},{net_tx}\n".format(**s))
        except Exception as e:
            print("csv write err:", e)


HTML = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GPU PC モニタリング</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
 body{margin:0;background:#0f1117;color:#e5e7eb;font-family:-apple-system,"Segoe UI",sans-serif}
 header{padding:16px 24px;border-bottom:1px solid #232634}
 h1{margin:0;font-size:18px}
 #updated{color:#9ca3af;font-size:12px;margin-top:4px}
 .cards{display:flex;flex-wrap:wrap;gap:12px;padding:20px 24px}
 .card{background:#1a1d29;border:1px solid #232634;border-radius:10px;padding:14px 18px;min-width:140px;flex:1}
 .card .label{color:#9ca3af;font-size:12px}
 .card .value{font-size:26px;font-weight:700;margin-top:4px}
 .card .sub{color:#6b7280;font-size:11px;margin-top:2px}
 .charts{padding:0 24px 24px}
 .chartbox{background:#1a1d29;border:1px solid #232634;border-radius:10px;padding:14px;margin-bottom:16px}
 .chartbox h2{margin:0 0 8px;font-size:13px;color:#9ca3af;font-weight:600}
 .ok{color:#34d399}.warn{color:#fbbf24}.hot{color:#f87171}
</style>
</head>
<body>
<header>
 <h1>🖥 GPU PC リソースモニタリング</h1>
 <div id="updated">読み込み中...</div>
</header>
<div class="cards">
 <div class="card"><div class="label">CPU 使用率</div><div class="value" id="c_cpu">-</div></div>
 <div class="card"><div class="label">CPU 温度</div><div class="value" id="c_cput">-</div></div>
 <div class="card"><div class="label">GPU 使用率</div><div class="value" id="c_gpu">-</div></div>
 <div class="card"><div class="label">GPU 温度</div><div class="value" id="c_gput">-</div></div>
 <div class="card"><div class="label">GPU メモリ</div><div class="value" id="c_gmem">-</div><div class="sub" id="c_gmemsub"></div></div>
 <div class="card"><div class="label">ネットワーク</div><div class="value" id="c_net">-</div><div class="sub">↓受信 / ↑送信</div></div>
</div>
<div class="charts">
 <div class="chartbox"><h2>使用率 (%)</h2><canvas id="chUsage" height="90"></canvas></div>
 <div class="chartbox"><h2>温度 (°C)</h2><canvas id="chTemp" height="90"></canvas></div>
 <div class="chartbox"><h2>ネットワーク (KB/s)</h2><canvas id="chNet" height="90"></canvas></div>
</div>
<script>
const mk=(id,series)=>new Chart(document.getElementById(id),{
  type:'line',
  data:{labels:[],datasets:series.map(s=>({label:s.l,borderColor:s.c,backgroundColor:s.c+'22',
        data:[],tension:0.25,pointRadius:0,borderWidth:2,fill:true,spanGaps:true}))},
  options:{animation:false,responsive:true,interaction:{intersect:false,mode:'index'},
    scales:{x:{ticks:{color:'#6b7280',maxTicksLimit:8},grid:{color:'#232634'}},
            y:{ticks:{color:'#6b7280'},grid:{color:'#232634'},beginAtZero:true}},
    plugins:{legend:{labels:{color:'#9ca3af',boxWidth:12}}}}
});
const chU=mk('chUsage',[{l:'CPU %',c:'#60a5fa'},{l:'GPU %',c:'#34d399'}]);
const chT=mk('chTemp',[{l:'CPU °C',c:'#60a5fa'},{l:'GPU °C',c:'#f87171'}]);
const chN=mk('chNet',[{l:'受信 KB/s',c:'#a78bfa'},{l:'送信 KB/s',c:'#fbbf24'}]);
function tcls(v,w,h){return v==null?'':v>=h?'hot':v>=w?'warn':'ok';}
function setv(id,txt,cls){const e=document.getElementById(id);e.textContent=txt;e.className='value '+(cls||'');}
async function refresh(){
 try{
  const d=await (await fetch('/api/metrics')).json();
  if(!d.length)return;
  const last=d[d.length-1];
  const labels=d.map(x=>new Date(x.t*1000).toLocaleTimeString('ja-JP',{hour:'2-digit',minute:'2-digit',second:'2-digit'}));
  setv('c_cpu',last.cpu!=null?last.cpu+' %':'-',tcls(last.cpu,70,90));
  setv('c_cput',last.cpu_temp!=null?last.cpu_temp+' °C':'-',tcls(last.cpu_temp,70,85));
  setv('c_gpu',last.gpu!=null?last.gpu+' %':'-');
  setv('c_gput',last.gpu_temp!=null?last.gpu_temp+' °C':'N/A',tcls(last.gpu_temp,75,85));
  if(last.gpu_mem!=null){
   setv('c_gmem',(last.gpu_mem/1024).toFixed(1)+' GB');
   document.getElementById('c_gmemsub').textContent='/ '+(last.gpu_mem_total/1024).toFixed(0)+' GB';
  }
  setv('c_net','↓'+last.net_rx+' / ↑'+last.net_tx);
  document.getElementById('c_net').style.fontSize='17px';
  document.getElementById('updated').textContent='最終更新: '+new Date(last.t*1000).toLocaleString('ja-JP')+'  (5秒ごと自動更新, '+d.length+'点保持)';
  const upd=(ch,arrs)=>{ch.data.labels=labels;arrs.forEach((a,i)=>ch.data.datasets[i].data=a);ch.update('none');};
  upd(chU,[d.map(x=>x.cpu),d.map(x=>x.gpu)]);
  upd(chT,[d.map(x=>x.cpu_temp),d.map(x=>x.gpu_temp)]);
  upd(chN,[d.map(x=>x.net_rx),d.map(x=>x.net_tx)]);
 }catch(e){document.getElementById('updated').textContent='取得エラー: '+e;}
}
refresh();setInterval(refresh,5000);
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/metrics"):
            with LOCK:
                data = list(HISTORY)
            self._send(json.dumps(data).encode(), "application/json")
        elif self.path == "/" or self.path.startswith("/index"):
            self._send(HTML.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self.send_error(404)


if __name__ == "__main__":
    os.makedirs(BASE, exist_ok=True)
    load_csv()
    threading.Thread(target=collector, daemon=True).start()
    print("monitoring dashboard on :%d" % PORT)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
