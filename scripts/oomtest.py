import json, time, threading, urllib.request, random, os, sys

URL = "http://127.0.0.1:8080/v1/chat/completions"
random.seed(42)
SYL = ["ба","вло","гри","де","жу","зна","ки","лы","мо","не","про","ра","со","ту","ве","ха","ца","ча","ша","э","ю","я","kost","bran","slav","mir"]
def words(n):
    return " ".join(random.choice(SYL)+random.choice(SYL)+random.choice(SYL) for _ in range(n))

def req(text, max_tokens, timeout=1800):
    body = json.dumps({"model":"qwen3.8","messages":[{"role":"user","content":text}],
                       "max_tokens":max_tokens,"temperature":0.1}).encode()
    t0=time.time()
    r = urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type":"application/json"}), timeout=timeout)
    j = json.load(r)
    return time.time()-t0, j["usage"]["prompt_tokens"], j["usage"]["completion_tokens"]

def cgroup():
    def rd(p):
        try: return open(f"/sys/fs/cgroup/{p}").read().strip()
        except Exception: return "?"
    ev = {}
    for line in rd("memory.events").splitlines():
        k,v = line.split(); ev[k]=int(v)
    return {"cur_mb": int(rd("memory.current"))//2**20, "peak_mb": int(rd("memory.peak"))//2**20,
            "oom": ev.get("oom",0), "oom_kill": ev.get("oom_kill",0)}

def shm_mb():
    st = os.statvfs("/dev/shm"); return (st.f_blocks-st.f_bfree)*st.f_frsize//2**20

# calibrate words -> tokens with a small probe
_,pt,ct = req(words(4000), 1)
ratio = 4000/pt   # words per token
print(f"calibration: 4000 words = {pt} tokens -> {ratio:.3f} w/tok", flush=True)

print("baseline:", cgroup(), f"shm={shm_mb()}MB", flush=True)

def phase(name, n_req, target_tok, gen):
    print(f"--- {name}: {n_req} x ~{target_tok} tok prompt + {gen} gen ---", flush=True)
    mon_stop = threading.Event(); peak = {"cur":0}
    def mon():
        while not mon_stop.is_set():
            c = cgroup(); peak["cur"]=max(peak["cur"], c["cur_mb"])
            peak.setdefault("ev", c)
            time.sleep(3)
    threading.Thread(target=mon, daemon=True).start()
    nw = int(target_tok*ratio)
    out = {}
    def worker(i):
        try:
            dt, p, c = req(words(nw), gen)
            out[i] = (dt, p, c)
        except Exception as e:
            out[i] = ("ERR", repr(e)[:120], 0)
    ths = [threading.Thread(target=worker, args=(i,)) for i in range(n_req)]
    t0=time.time()
    [t.start() for t in ths]; [t.join() for t in ths]
    mon_stop.set(); time.sleep(0.2)
    ok = sum(1 for v in out.values() if v[0] != "ERR")
    for i in sorted(out):
        dt,p,c = out[i]
        if dt == "ERR": print(f"  req{i}: FAILED {p}", flush=True)
        else: print(f"  req{i}: {dt:.1f}s  prompt={p}  gen={c}  ({p/max(dt,0.01):.0f} tok/s prefill)", flush=True)
    print(f"{name}: {ok}/{n_req} ok in {time.time()-t0:.0f}s | mem peak during: {peak['cur']}MB | now: {cgroup()} shm={shm_mb()}MB", flush=True)

phase("P1-mid", 8, 250_000, 300)
phase("P2-overload", 5, 950_000, 300)
final = cgroup()
print("FINAL:", final, f"shm={shm_mb()}MB", flush=True)
print("VERDICT:", "NO OOM" if final["oom_kill"]==0 else f"OOM KILLS HAPPENED: {final['oom_kill']}", flush=True)
