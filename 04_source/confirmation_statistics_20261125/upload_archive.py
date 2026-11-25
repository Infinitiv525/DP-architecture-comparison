"""Stream complete archives Perun->GitHub; auth stays only in stdin/RAM."""
import concurrent.futures,hashlib,json,sys,threading,time,urllib.request,urllib.error,urllib.parse
from pathlib import Path
ROOT=Path('/mnt/project/perun2601449/dp_llm_cfd')
A=ROOT/'results/architecture_confirmation_archive_20261125_v1'
O=ROOT/'results/architecture_confirmation_statistics_20261125_v1'
token=json.load(sys.stdin)['token'];repo='Infinitiv525/DP-architecture-comparison';tag='architecture-confirmation20-complete-20261125'
lock=threading.Lock();uploaded={};verified={};start=time.monotonic()
def read(p):return json.loads(p.read_text())
def save(p,v):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def api(method,path,data=None):
 req=urllib.request.Request('https://api.github.com/repos/'+repo+path,method=method,
  headers={'Authorization':'Bearer '+token,'User-Agent':'DP-confirmation-archive','Accept':'application/vnd.github+json','Content-Type':'application/json'},
  data=None if data is None else json.dumps(data).encode())
 with urllib.request.urlopen(req,timeout=180) as h:return json.load(h)
def progress(phase):
 save(A/'publication_progress.json',dict(phase=phase,uploaded=len(uploaded),publicly_read_back=len(verified),
  expected_assets=63,uploaded_bytes=sum(v['bytes'] for v in uploaded.values()),
  read_back_bytes=sum(v['bytes'] for v in verified.values()),elapsed_seconds=time.monotonic()-start,
  no_PC_copy=True))
def waitfile(p,limit=3600):
 deadline=time.monotonic()+limit
 while not p.exists():
  assert time.monotonic()<deadline,('Archive preparation not ready',str(p))
  time.sleep(5)
 return read(p)
results=read(O/'results_publication.json');assert results['status']=='STATISTICAL_RESULTS_PUBLIC_AND_ALL_FILES_READ_BACK_VERIFIED'
state=A/'release_state.json'
if state.exists():
 release=api('GET','/releases/'+str(read(state)['release_id']));assert release['tag_name']==tag
else:
 try:existing=api('GET','/releases/tags/'+tag)
 except urllib.error.HTTPError as e:
  if e.code!=404:raise
 else:raise RuntimeError('Inspect pre-existing release before adopting it')
 release=api('POST','/releases',dict(tag_name=tag,target_commitish=results['commit'],draft=True,
  name='20-seed architecture confirmation: all60models,12000fields and statistics',
  body='Complete new independent20-seed comparison ofTFNO,denseFNO andRealNPU. See the repository report for primary paired tests, exact sensitivities and geometry limitations. All60training histories, best/last checkpoints,12000native TEST fields and12million timing pairs. Model and statistical sources, inputs and all200000bootstrap replicates. Archives are prepared and verified onPerun, noPCcopy. Checksums inARTIFACT_MANIFEST.json andSHA256SUMS.txt.'))
 save(state,dict(release_id=release['id'],tag=tag,results_commit=results['commit']))
uploadbase=release['upload_url'].split('{')[0];assert urllib.parse.urlparse(uploadbase).hostname=='uploads.github.com'
known={x['name']:x for x in api('GET','/releases/'+str(release['id'])+'/assets?per_page=100')}
def upload(item):
 name=item['name'];asset=known.get(name);path=Path(item['path'])
 assert path.is_file() and path.stat().st_size==item['bytes']<1900000000
 if asset is not None:
  assert asset['state']=='uploaded' and asset['size']==item['bytes'],('Incomplete existing asset',name)
  if asset.get('digest'):assert asset['digest']=='sha256:'+item['sha256']
 else:
  url=uploadbase+'?name='+urllib.parse.quote(name)
  headers={'Authorization':'Bearer '+token,'User-Agent':'DP-confirmation-archive',
   'Content-Type':'application/octet-stream','Content-Length':str(item['bytes']),'Accept':'application/vnd.github+json'}
  with path.open('rb') as stream:
   req=urllib.request.Request(url,data=stream,method='POST',headers=headers)
   with urllib.request.urlopen(req,timeout=900) as h:asset=json.load(h)
  assert asset['state']=='uploaded' and asset['size']==item['bytes']
  if asset.get('digest'):assert asset['digest']=='sha256:'+item['sha256']
 rec={k:v for k,v in item.items() if k!='path'}
 rec.update(asset_id=asset['id'],url=asset['browser_download_url'],github_digest=asset.get('digest'))
 with lock:
  uploaded[name]=rec;progress('UPLOADING')
  save(A/'uploaded_assets.json',uploaded)
 print('UPLOADED',len(uploaded),name,item['bytes'],flush=True)
 return rec
rows=read(ROOT/'test/architecture_confirmation_test_20261125_v2/models.json')['models']
def uploadmodel(row):
 d=waitfile(A/'manifests'/(row['id']+'.json'))
 assert d['all_members_fully_read_back_verified']
 return upload(dict(name=d['archive_name'],path=str(A/'packages'/d['archive_name']),bytes=d['bytes'],sha256=d['sha256'],
  model_id=d['id'],method=d['method'],seed=d['seed'],native_fields=200))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 list(pool.map(uploadmodel,rows))
catalog=waitfile(A/'archive_verification.json')
assert catalog['status']=='ALL_ARCHIVES_FULLY_VERIFIED_READY_FOR_DIRECT_PUBLICATION' and len(catalog['assets'])==63
for item in catalog['assets']:
 if item['name'] not in uploaded:upload(item)
assert len(uploaded)==63 and {x['name'] for x in catalog['assets']}==set(uploaded)
api('PATCH','/releases/'+str(release['id']),dict(draft=False))
# Fetch public URLs after publication, avoiding stale redirects from a draft.
url='https://api.github.com/repos/'+repo+'/releases/tags/'+tag+'?v='+str(time.time_ns())
with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'DP-anonymous-archive-check'}),timeout=120) as h:public=json.load(h)
assert not public['draft']
fresh={v['name']:v for v in public['assets']}
assert set(uploaded).issubset(fresh)
with lock:progress('PUBLIC_ANONYMOUS_READBACK')
def readback(name):
 item=uploaded[name];u=fresh[name]['browser_download_url']+'?verify='+str(time.time_ns())
 h=hashlib.sha256();size=0
 req=urllib.request.Request(u,headers={'User-Agent':'DP-anonymous-archive-check'})
 with urllib.request.urlopen(req,timeout=900) as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block);size+=len(block)
 assert size==item['bytes'] and h.hexdigest()==item['sha256'],name
 result=dict(item,anonymous_full_readback_verified=True)
 with lock:
  verified[name]=result;progress('PUBLIC_ANONYMOUS_READBACK')
  save(A/'public_readback_assets.json',verified)
 print('READBACK',len(verified),name,size,flush=True)
 return result
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(readback,sorted(uploaded)))
assert len(verified)==63
receipt=dict(status='ALL60_MODELS12000FIELDS_STATISTICS_AND_SOURCES_PUBLIC_ANONYMOUS_READBACK_VERIFIED',
 repository=repo,release_id=release['id'],release_url='https://github.com/'+repo+'/releases/tag/'+tag,
 assets=list(verified.values()),asset_count=63,total_bytes=sum(v['bytes'] for v in verified.values()),
 model_count=60,native_fields=12000,raw_timing_pairs=12000000,all_training_best_last_and_histories=True,
 statistics_bootstrap_replicates=200000,no_PC_copy=True,no_GPU_used_for_publication=True,elapsed_seconds=time.monotonic()-start)
save(A/'publication_receipt.json',receipt);save(O/'publication_receipt.json',receipt)
with lock:progress('ALL_UPLOADS_AND_ANONYMOUS_READBACK_COMPLETE')
print(json.dumps({k:v for k,v in receipt.items() if k!='assets'}),flush=True)
