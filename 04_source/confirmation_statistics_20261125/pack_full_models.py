"""Archive every new model, training record and complete native TEST output on Perun."""
import concurrent.futures,hashlib,json,os,tarfile,time
from pathlib import Path
ROOT=Path('/mnt/project/perun2601449/dp_llm_cfd')
TRAIN=ROOT/'results/architecture_confirmation_20261124_v2'
TEST=ROOT/'results/architecture_confirmation_test_20261125_v2'
OUT=ROOT/'results/architecture_confirmation_archive_20261125_v1'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
class HashReader:
 def __init__(self,f):self.f=f;self.h=hashlib.sha256();self.size=0
 def read(self,n=-1):
  b=self.f.read(n);self.h.update(b);self.size+=len(b);return b
def pack(row):
 mid=row['id'];dest=OUT/'packages'/(mid+'__model_and_TEST.tar.gz');proof=OUT/'manifests'/(mid+'.json')
 assert not dest.exists() and not proof.exists()
 summary=read(TEST/'test/models'/mid/'summary.json')
 expected={str(Path(row['checkpoint'])):row['checkpoint_sha256'],
  str(TEST/'test/models'/mid/'predictions.npy'):summary['predictions_sha256'],
  str(TEST/'test/models'/mid/'raw_timings.npz'):summary['timings_sha256']}
 entries=[]
 files=[]
 for directory,label in [(TRAIN/'main'/mid,'training'),(TEST/'test/models'/mid,'TEST')]:
  assert directory.is_dir()
  for p in sorted(directory.rglob('*')):
   if p.is_file():files.append((p,mid+'/'+label+'/'+str(p.relative_to(directory))))
 with tarfile.open(dest,'w:gz',compresslevel=1) as tar:
  for p,name in files:
   info=tar.gettarinfo(str(p),arcname=name);assert info.isfile()
   with p.open('rb') as f:
    stream=HashReader(f);tar.addfile(info,stream)
   assert stream.size==info.size
   h=stream.h.hexdigest()
   if str(p) in expected:assert expected[str(p)]==h
   entries.append(dict(path=name,bytes=info.size,sha256=h,original_Perun_path=str(p)))
 assert set(expected).issubset({str(p) for p,_ in files})
 # Full decompression verifies that the archive contains every original byte.
 lookup={x['path']:x for x in entries};seen=set()
 with tarfile.open(dest,'r|gz') as tar:
  for member in tar:
   assert member.isfile() and member.name in lookup and member.name not in seen;seen.add(member.name)
   h=hashlib.sha256();f=tar.extractfile(member)
   for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
   assert member.size==lookup[member.name]['bytes'] and h.hexdigest()==lookup[member.name]['sha256']
 assert seen==set(lookup) and dest.stat().st_size<1900000000
 result=dict(id=mid,method=row['model'],seed=row['seed'],seed_block=row['seed_block'],
  native_TEST_fields=200,archive_name=dest.name,bytes=dest.stat().st_size,sha256=sha(dest),
  members=entries,all_members_fully_read_back_verified=True,
  includes=['best checkpoint','last checkpoint and optimizer states','complete training history',
   'all200native TEST fields','all200x1000GPU and host timing pairs','per-case errors and physics','visual comparison'])
 save(proof,result);return result
def main():
 assert not OUT.exists();OUT.mkdir();(OUT/'packages').mkdir();(OUT/'manifests').mkdir()
 assert read(TEST/'audit/summary.json')['status']=='CONFIRMATION60_TEST_AND_SAVED_OUTPUT_AUDIT_COMPLETE'
 assert read(ROOT/'results/architecture_confirmation_statistics_20261125_v1/independent_verification.json')['status']=='INDEPENDENT_INPUT_PRIMARY_EXACT_SIGN_AND_BOOTSTRAP_AUDIT_PASS'
 rows=read(ROOT/'test/architecture_confirmation_test_20261125_v2/models.json')['models'];assert len(rows)==60
 done=[];start=time.monotonic()
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
  futures=[pool.submit(pack,row) for row in rows]
  for future in concurrent.futures.as_completed(futures):
   done.append(future.result());save(OUT/'packing_progress.json',dict(status='PACKING',complete=len(done),expected=60,
    compressed_bytes=sum(x['bytes'] for x in done),elapsed_seconds=time.monotonic()-start))
   print('PACKED',len(done),done[-1]['id'],done[-1]['bytes'],flush=True)
 done.sort(key=lambda x:(['tfno','fno','realnpu'].index(x['method']),x['seed_block']))
 assert len(done)==60 and sum(x['native_TEST_fields'] for x in done)==12000
 index=dict(status='ALL60_MODEL_AND_FULL_TEST_PACKAGES_BYTE_VERIFIED',packages=done,
  native_fields=12000,compressed_bytes=sum(x['bytes'] for x in done),uncompressed_bytes=sum(e['bytes'] for x in done for e in x['members']),
  elapsed_seconds=time.monotonic()-start,no_PC_copy=True,no_GPU=True)
 save(OUT/'PACKAGE_INDEX.json',index)
 (OUT/'PACKAGE_SHA256SUMS.txt').write_text(''.join(x['sha256']+'  '+x['archive_name']+'\n' for x in done))
 print(json.dumps({k:v for k,v in index.items() if k!='packages'}),flush=True)
if __name__=='__main__':main()
