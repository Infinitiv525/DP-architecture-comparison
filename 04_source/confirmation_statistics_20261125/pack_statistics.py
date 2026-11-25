"""Build the full numerical/source archive and complete release catalog; CPU only."""
import hashlib,json,tarfile,sys
from pathlib import Path
ROOT=Path('/mnt/project/perun2601449/dp_llm_cfd')
S=ROOT/'test/architecture_confirmation_statistics_20261125_v1'
O=ROOT/'results/architecture_confirmation_statistics_20261125_v1'
A=ROOT/'results/architecture_confirmation_archive_20261125_v1'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert read(A/'PACKAGE_INDEX.json')['status']=='ALL60_MODEL_AND_FULL_TEST_PACKAGES_BYTE_VERIFIED'
assert read(O/'independent_verification.json')['status']=='INDEPENDENT_INPUT_PRIMARY_EXACT_SIGN_AND_BOOTSTRAP_AUDIT_PASS'
entries={}
def add(p,name):
 p=Path(p)
 if name in entries:assert entries[name]==p
 else:entries[name]=p
for directory,prefix in [(S,'statistics_source'),(O,'statistics_results')]:
 for p in sorted(directory.iterdir()):
  if p.is_file() and p.suffix in ['.py','.sbatch','.txt','.md','.json','.csv','.npz','.out','.err']:
   if p.name not in ['publication_progress.json','publication_receipt.json','archive_verification.json']:
    add(p,prefix+'/'+p.name)
for path,h in read(O/'input_manifest.json')['files'].items():
 p=Path(path);assert sha(p)==h;add(p,'perun/'+str(p.relative_to(ROOT)))
for directory in [ROOT/'test/architecture_confirmation_20261124_v2',ROOT/'test/architecture_confirmation_test_20261125_v2']:
 for p in directory.rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.py','.sbatch','.json','.txt','.md']:
   add(p,'perun/'+str(p.relative_to(ROOT)))
config=read(ROOT/'test/architecture_confirmation_test_20261125_v2/config.json')
dependencies=list(config['legacy_source_sha256'])+list(config['extra_source_sha256'])
dependencies += [str(Path(config['field_source'])/n) for n in config['field_source_sha256']]
dependencies += [str(p) for p in (ROOT/'test/architecture_fair_pilot_20261111_v2/revisions/r_0bc20031d1e9f75f').iterdir() if p.is_file() and p.suffix in ['.py','.json','.txt']]
dependencies += [str(ROOT/'test/architecture_pilot_20261110_v1/revisions/r_e9dafab653bac9e5/models.py')]
for p in dependencies:add(Path(p),'perun/'+str(Path(p).relative_to(ROOT)))
add(A/'PACKAGE_INDEX.json','archive_index/PACKAGE_INDEX.json')
add(A/'PACKAGE_SHA256SUMS.txt','archive_index/PACKAGE_SHA256SUMS.txt')
members=[dict(path=name,source=str(p),bytes=p.stat().st_size,sha256=sha(p)) for name,p in sorted(entries.items())]
target=A/'confirmation_statistics_and_source_20261125.tar.gz';assert not target.exists()
with tarfile.open(target,'w:gz',compresslevel=1) as tar:
 for x in members:tar.add(x['source'],arcname=x['path'],recursive=False)
expected={x['path']:x for x in members};seen=set()
with tarfile.open(target,'r|gz') as tar:
 for member in tar:
  assert member.isfile() and member.name in expected and member.name not in seen;seen.add(member.name)
  h=hashlib.sha256();f=tar.extractfile(member)
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
  assert member.size==expected[member.name]['bytes'] and h.hexdigest()==expected[member.name]['sha256']
assert seen==set(expected)
index=read(A/'PACKAGE_INDEX.json')
assets=[dict(name=x['archive_name'],path=str(A/'packages'/x['archive_name']),bytes=x['bytes'],sha256=x['sha256'],id=x['id'],method=x['method'],seed=x['seed']) for x in index['packages']]
assets.append(dict(name=target.name,path=str(target),bytes=target.stat().st_size,sha256=sha(target),kind='all numerical results,200000resamples,inputs,sources and audits'))
manifest=dict(status='ALL60_MODELS12000FIELDS_AND_STATISTICS_ARCHIVED_AND_BYTE_VERIFIED',
 packages=index['packages'],statistics_archive=dict(name=target.name,members=[{k:v for k,v in x.items() if k!='source'} for x in members]),
 all_archive_payloads_read_back=True,all_seeds_retained=True,no_PC_copy=True,model_count=60,native_fields=12000)
mp=A/'ARTIFACT_MANIFEST.json';save(mp,manifest)
assets.append(dict(name=mp.name,path=str(mp),bytes=mp.stat().st_size,sha256=sha(mp),kind='per-file and per-package checksums'))
sp=A/'SHA256SUMS.txt';sp.write_text(''.join(x['sha256']+'  '+x['name']+'\n' for x in assets))
assets.append(dict(name=sp.name,path=str(sp),bytes=sp.stat().st_size,sha256=sha(sp),kind='release-asset checksums'))
receipt=dict(status='ALL_ARCHIVES_FULLY_VERIFIED_READY_FOR_DIRECT_PUBLICATION',assets=assets,
 model_count=60,native_fields=12000,total_bytes=sum(x['bytes'] for x in assets),
 statistics_archive_members=len(members),no_GPU=True,no_PC_copy=True)
save(A/'archive_verification.json',receipt)
print(json.dumps({k:v for k,v in receipt.items() if k!='assets'}),flush=True)
