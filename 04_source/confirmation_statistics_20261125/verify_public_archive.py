"""Resume full anonymous archive validation with bounded parallel public downloads."""
import concurrent.futures,hashlib,json,sys,threading,time,urllib.request
from pathlib import Path
R=Path('/mnt/project/perun2601449/dp_llm_cfd');A=R/'results/architecture_confirmation_archive_20261125_v1';O=R/'results/architecture_confirmation_statistics_20261125_v1'
def read(p):return json.loads(p.read_text())
def save(p,v):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
old=read(A/'publication_progress.json');uploaded=read(A/'uploaded_assets.json');verified=read(A/'public_readback_assets.json')
assert len(uploaded)==63 and old['uploaded']==63 and all(x['anonymous_full_readback_verified'] for x in verified.values())
repo='Infinitiv525/DP-architecture-comparison';tag='architecture-confirmation20-complete-20261125'
url='https://api.github.com/repos/'+repo+'/releases/tags/'+tag+'?v='+str(time.time_ns())
with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'DP-anonymous-readback'}),timeout=120) as h:public=json.load(h)
assert not public['draft'];fresh={x['name']:x for x in public['assets']}
for name,item in uploaded.items():
 assert fresh[name]['id']==item['asset_id'] and fresh[name]['size']==item['bytes']
 if fresh[name].get('digest'):assert fresh[name]['digest']=='sha256:'+item['sha256']
reused=len(verified);workers=24;start=time.monotonic();lock=threading.Lock()
save(A/'readback_parallelism_amendment.json',dict(uploaded_assets_unchanged=63,reused_completed_full_readbacks=reused,
 previous_parallel_readers=4,new_parallel_readers=workers,
 reason='Increase parallel anonymous readback only; all prior verified SHA records retained, no reupload or scientific computation.'))
def progress(phase):
 save(A/'publication_progress.json',dict(phase=phase,uploaded=63,publicly_read_back=len(verified),expected_assets=63,
  uploaded_bytes=sum(x['bytes'] for x in uploaded.values()),read_back_bytes=sum(x['bytes'] for x in verified.values()),
  elapsed_seconds=old['elapsed_seconds']+time.monotonic()-start,parallel_readers=workers,no_PC_copy=True))
def fetch(name):
 item=uploaded[name];url=fresh[name]['browser_download_url']+'?verify='+str(time.time_ns())
 h=hashlib.sha256();size=0
 with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'DP-anonymous-archive-check'}),timeout=900) as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block);size+=len(block)
 assert size==item['bytes'] and h.hexdigest()==item['sha256'],name
 result=dict(item,anonymous_full_readback_verified=True)
 with lock:
  verified[name]=result;save(A/'public_readback_assets.json',verified);progress('PUBLIC_ANONYMOUS_READBACK')
 print('READBACK',len(verified),name,size,flush=True)
 return result
pending=[name for name in sorted(uploaded) if name not in verified]
with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(fetch,pending))
assert len(verified)==63
receipt=dict(status='ALL60_MODELS12000FIELDS_STATISTICS_AND_SOURCES_PUBLIC_ANONYMOUS_READBACK_VERIFIED',
 repository=repo,release_id=public['id'],release_url='https://github.com/'+repo+'/releases/tag/'+tag,
 assets=[verified[n] for n in sorted(verified)],asset_count=63,total_bytes=sum(x['bytes'] for x in verified.values()),
 model_count=60,native_fields=12000,raw_timing_pairs=12000000,all_training_best_last_and_histories=True,
 statistics_bootstrap_replicates=200000,no_PC_copy=True,no_GPU_used_for_publication=True,
 readback_parallel_workers=workers,reused_previously_completed_readbacks=reused,
 elapsed_seconds=old['elapsed_seconds']+time.monotonic()-start)
save(A/'publication_receipt.json',receipt);save(O/'publication_receipt.json',receipt);progress('ALL_UPLOADS_AND_ANONYMOUS_READBACK_COMPLETE')
print(json.dumps({k:v for k,v in receipt.items() if k!='assets'}),flush=True)
