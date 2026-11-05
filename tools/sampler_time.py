#!/usr/bin/env python3
"""The host sampler's cost per token over 248,320 logits (docs/weight-space.md 14.6, T8).

    python3 tools/sampler_time.py <scratch dir> build/cancho-gpu

Runs `cancho-gpu sample` (temperature 0.6, top-p 0.95, top-k 20) on 21 rows and on 1 row of 248,320 f32 logits, five times
each, and divides the difference of the medians by 20: the per-row cost with the process start and the file read taken out.
It is for no penalty and for a presence penalty of 1.5 with 1, 100, 1,000, 4,000 and 8,000 distinct seen ids. The figure
includes what the command does per row besides sampling (it parses the row's seen line, prints the nucleus), so it is an
upper bound on the sampler itself.
"""
D=sys.argv[1]; BIN=sys.argv[2]
V=248320; R=21
rng=np.random.default_rng(1)
row=rng.normal(0,2.0,V).astype('<f4'); row[:30]+=np.linspace(8,4,30).astype('<f4')
rows=np.tile(row,(R,1)); f=D+'/l.f32'; rows.tofile(f)
f1=D+'/l1.f32'; row.tofile(f1)
def t(path,extra,n=5):
    ts=[]
    for _ in range(n):
        a=time.perf_counter(); r=subprocess.run([BIN,'sample',path,str(V),'0.6','0.95','20','7']+extra,capture_output=True); ts.append(time.perf_counter()-a); assert r.returncode==0,r.stderr
    return statistics.median(ts)
def per(seen_n):
    if seen_n is None: e=[]; e1=[]
    else:
        ids=rng.choice(V,seen_n,replace=False)
        sf=D+'/s.txt'; open(sf,'w').write((' '.join(map(str,ids))+'\n')*R)
        e=['1.5',sf]; e1=e
    return (t(f,e)-t(f1,e1))/(R-1)*1000
print('no penalty', round(per(None),2),'ms/token')
for n in (1,100,1000,4000,8000): print('penalty, seen',n, round(per(n),2),'ms/token')
