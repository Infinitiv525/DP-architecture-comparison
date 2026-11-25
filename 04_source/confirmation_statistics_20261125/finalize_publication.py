"""Finalize human catalogs after all63public assets passed anonymous full readback."""
import base64,csv,hashlib,io,json,sys,urllib.request
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
token=json.load(sys.stdin)['token'];repo='Infinitiv525/DP-architecture-comparison'
R=Path('/mnt/project/perun2601449/dp_llm_cfd');O=R/'results/architecture_confirmation_statistics_20261125_v1';A=R/'results/architecture_confirmation_archive_20261125_v1';S=R/'test/architecture_confirmation_statistics_20261125_v1'
T=R/'results/architecture_confirmation_test_20261125_v2';TR=R/'results/architecture_confirmation_20261124_v2'
def read(p):return json.loads(p.read_text())
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
receipt=read(A/'publication_receipt.json')
assert receipt['status']=='ALL60_MODELS12000FIELDS_STATISTICS_AND_SOURCES_PUBLIC_ANONYMOUS_READBACK_VERIFIED'
assert receipt['asset_count']==63 and all(x['anonymous_full_readback_verified'] for x in receipt['assets'])
index=read(A/'PACKAGE_INDEX.json');assets={x['name']:x for x in receipt['assets']};release=receipt['release_url']
models=read(R/'test/architecture_confirmation_test_20261125_v2/models.json')['models'];byid={x['id']:x for x in models}
labels={'tfno':'TFNO','fno':'dense FNO','realnpu':'RealNPU'};rows=[]
catalog=['# Všetkých60nových modelov a12000predikovaných polí','',
'Každý odkaz nižšie stiahne iba jeden balík jedného tréningu. Bežné klonovanie repozitára nesťahuje váhy ani veľké polia zReleases.',
'Každý balík obsahuje pôvodné best.pt ajlast.pt s dostupným stavom optimalizátora, celú tréningovú históriu,200natívnych TESTpolí512×512×3,200×1000dvojíc GPU/hostčasovaní, prípadové chyby,fyzikálne diagnostiky a obrazový príklad.',
'Vnútorná cesta:IDmodelu/training/best.pt aIDmodelu/TEST/predictions.npy. Metriky a vstupný/výstupný tvar sú rovnaké pre všetkých60modelov. Každá architektúra má20čerstvých behov; rovnaké číslo behu tvorí párovaný seedový blok.',
'[Výsledky a štatistika](../../01_results/confirmation_statistics_20261125/VYSLEDKY.txt) · [Celý Release]('+release+') · [Manifest všetkých súborov]('+assets['ARTIFACT_MANIFEST.json']['url']+').',
'']
for method in ['tfno','fno','realnpu']:
 catalog+=['## '+labels[method],'','| Beh | Semeno | Balík MiB | Váhy,celý TEST a záznamy |','|---:|---:|---:|---|']
 group=[x for x in index['packages'] if x['method']==method];assert len(group)==20
 for x in group:
  asset=assets[x['archive_name']];m=byid[x['id']]
  catalog.append('| %d | %d | %.1f | [Stiahnuť %s](%s) |'%(x['seed_block']+1,x['seed'],x['bytes']/1024**2,x['id'],asset['url']))
  rows.append(dict(model_id=x['id'],architecture=method,seed_block=x['seed_block'],seed=x['seed'],
   checkpoint_SHA256=m['checkpoint_sha256'],archive_name=x['archive_name'],archive_bytes=x['bytes'],
   archive_SHA256=x['sha256'],public_download_url=asset['url'],native_TEST_fields=200,
   selected_parameters=m['parameter_ledger']['learned_or_fitted_total']))
 catalog.append('')
catalog+=['## Reprodukcia a kontrolné súčty','',
'Všetky balíky sa naPerune celé dekomprimovali a overil sa SHA256 každého súboru. Po zverejnení sa všetkých63príloh anonymne celých prečítalo späť a porovnali sa ich SHA256. Bez kópie vPC.',
'Samostatný [archív zdrojov a všetkých numerických podkladov]('+assets['confirmation_statistics_and_source_20261125.tar.gz']['url']+') obsahuje ajvšetkých200000bootstrapreplík, kompaktné vstupy a nezávislé overenie.',
'Na overenie primárnej štatistiky po rozbalení:python statistics_source/verify_portable.py statistics_results . StačíPython,NumPy aSciPy; netrebaGPU ani veľké polia.',
'Externé referenčné dátaFlowBench sa v týchto balíkoch neduplikujú. Ich zdroje, rozdelenie, normalizácia a presné odtlačky sú zachované v metodike a konfigurácii.']
(A/'MODEL_CATALOG.md').write_text('\n'.join(catalog)+'\n',encoding='utf-8')
buf=io.StringIO(newline='');w=csv.DictWriter(buf,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with (A/'model_index.csv').open('w',encoding='utf-8',newline='') as stream:stream.write(buf.getvalue())
text=['ÚPLNÝ VEREJNÝ ARCHÍV POTVRDENIA20NOVÝCH TRÉNINGOV','',
'Všetkých60modelov,12000celých TESTpolí a kompletná štatistika sú verejne uložené a overené.',
'Release:'+release,
'Katalóg:https://github.com/'+repo+'/blob/main/02_models/confirmation20_20261125/README.md','',
'60samostatných modelových balíkov:20TFNO,20denseFNO,20RealNPU. Každý obsahuje best/lastváhy,dostupné optimizerstavy,tréningové histórie,200natívnych predikčných polí a všetky časovania,chyby,fyzikálne diagnostiky a ukážku.',
'Jeden samostatný numerický/sourcebalík obsahuje kompaktné vstupné metriky,všetkých200000bootstrapreplík,zdroje,protokoly a audity. Snapshot vznikol pred publikáciou; aktuálne potvrdenie uploadu je publication_receipt.json v repozitári.',
'ARTIFACT_MANIFEST.json určuje vnútorné súbory a SHA256. SHA256SUMS.txt určuje kontrolné súčty príloh.',
'Všetkých63príloh sa anonymne kompletne prečítalo späť z verejných URL; veľkosti a SHA256sedia.',
'Prenos ibaPerun→GitHub,žiadne objemné dáta vPC. Bežný gitclone veľké prílohy nesťahuje.',
'Celková veľkosť:%dbytes(%.3fGB v desiatkovej sústave).'%(receipt['total_bytes'],receipt['total_bytes']/1e9),'',
'Postup pre štatistiku:rozbaľ confirmation_statistics_and_source_20261125.tar.gz a spusti python statistics_source/verify_portable.py statistics_results . NetrebaGPU ani CFDpolia.']
(O/'ARCHIV.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
complete=dict(status='ALL60_TEST_STATISTICS_AND_COMPLETE_PUBLIC_ARCHIVE_VERIFIED',
 models=60,independent_seeds_per_method=20,native_fields_audited=12000,timing_pairs_audited=12000000,
 primary_contrasts=9,primary_Holm_significant=8,all60_per_case_p99_5ms_pass=True,
 public_assets=63,total_public_asset_bytes=receipt['total_bytes'],release_url=release,
 anonymous_complete_readback_verified=True,full_public_archive_pending=False,
 no_PC_copy=True,no_new_GPU_for_statistics_or_publication=True,
 completed_at=datetime.now(ZoneInfo('Europe/Bratislava')).isoformat())
save(O/'completion_status.json',complete);save(T/'status.json',complete);save(TR/'status.json',complete)
(O/'README.md').write_text('''# Potvrdenie troch architektúr na20nových tréningoch
[Čitateľné vyhodnotenie](VYSLEDKY.txt) · [Všetky primárne testy a intervaly](paired_t_primary.json) · [Nezávislé overenie](independent_verification.json).
TFNO má na pevnom200-prípadovom benchmarku nižšie chyby než denseFNO ajRealNPU vo všetkých3metrikách. Osem z9vopred určených kontrastov prešloHolm korekciou; celkováMSE denseFNO–RealNPU sa nepreukázala.
Používame20nových semien/metódu;pôvodných5pilotných behov je oddelených.
Geometrická bootstrapcitlivosť je slabšia/závisí od variantu intervalu. Všeobecnú prevahu na nevidených geometriách netvrdíme.
Všetkých60modelov splnilo per-casep99<=5ms naH200; jednotlivé prekročenia sú priznané v reporte.
[Kompletný archív a overenie](ARCHIV.txt) · [Váhy a všetky polia podľa architektúry a behu](../../02_models/confirmation20_20261125/README.md).
[Protokol](../../03_methodology/confirmation_statistics_20261125/protokol.txt) · [Zdroj](../../04_source/confirmation_statistics_20261125).
''',encoding='utf-8')
(T/'README.md').write_text('''# TEST60modelov dokončený a archivovaný
Všetkých60modelov×200prípadov=12000natívnych polí a12000000dvojíc časovaní prešlo nezávislýmCPUauditom.
[Opis TESTu](audit_result.txt) · [Súhrn auditu](saved_output_audit.json) · [Všetky prípadové metriky](case_metrics.csv).
[Štatistické porovnanie20nových semien](../confirmation_statistics_20261125/README.md).
[Všetky váhy,polia a záznamy](../../02_models/confirmation20_20261125/README.md) sú verejne zachované.63príloh sa anonymne celé prečítalo späť a overiloSHA256,bez kópie vPC.
Všetkých60modelov spĺňa per-casep99<=5ms; jednotlivé časové špičky sú zachované.
''',encoding='utf-8')
with (TR/'dokumentacia.txt').open('a',encoding='utf-8') as f:f.write('\n25.11.: Kompletnyverejnyarchiv60modelov/12000poli/statistiky:zverejnenych63priloh,%.3fGB,anonymnyplnySHA256readbackvsetkychpresiel. NicobjemnesanekopirovalodoPC. Zaveryplatiaprepevnybenchmarkageometrickeobmedzeniaostavaju.\n'%(receipt['total_bytes']/1e9))
save(O/'OUTPUT_MANIFEST.json',{p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in O.iterdir() if p.is_file() and p.name!='OUTPUT_MANIFEST.json'})
def api(method,path,data=None):
 req=urllib.request.Request('https://api.github.com/repos/'+repo+path,method=method,
  headers={'Authorization':'Bearer '+token,'User-Agent':'DP-confirmation-finalize','Accept':'application/vnd.github+json','Content-Type':'application/json'},
  data=None if data is None else json.dumps(data,ensure_ascii=False).encode())
 with urllib.request.urlopen(req,timeout=180) as h:return json.load(h)
files={}
for name in ['ARCHIV.txt','README.md','publication_receipt.json','completion_status.json','OUTPUT_MANIFEST.json','ENVIRONMENT.json']:
 files['01_results/confirmation_statistics_20261125/'+name]=(O/name).read_bytes()
for name in ['status.json','README.md']:files['01_results/confirmation_test_20261125/'+name]=(T/name).read_bytes()
for name in ['status.json','dokumentacia.txt']:files['01_results/confirmation_20261124/'+name]=(TR/name).read_bytes()
files['02_models/confirmation20_20261125/README.md']=(A/'MODEL_CATALOG.md').read_bytes()
files['02_models/confirmation20_20261125/model_index.csv']=(A/'model_index.csv').read_bytes()
for name in ['upload_archive.py','verify_public_archive.py','finalize_publication.py']:
 files['04_source/confirmation_statistics_20261125/'+name]=(S/name).read_bytes()
head=api('GET','/git/ref/heads/main')['object']['sha'];rd=api('GET','/contents/README.md?ref='+head)
body=base64.b64decode(rd['content']).decode()
old='Verejný archív všetkých nových váh/polí sa dokončuje priamo zPerunu.'
assert old in body
body=body.replace(old,'**[Kompletný verejný archív všetkých60modelov a12000polí](02_models/confirmation20_20261125/README.md) je overený.** Všetkých63príloh sa anonymne celých prečítalo späť a porovnal saSHA256;bez kópie vPC.')
files['README.md']=body.encode()
tree=api('GET','/git/commits/'+head)['tree']['sha']
tree=api('POST','/git/trees',dict(base_tree=tree,tree=[dict(path=p,mode='100644',type='blob',content=b.decode()) for p,b in files.items()]))['sha']
commit=api('POST','/git/commits',dict(message='Verify full public archive of60new models12000fields and final statistical comparison',tree=tree,parents=[head]))['sha']
assert api('GET','/git/ref/heads/main')['object']['sha']==head
api('PATCH','/git/refs/heads/main',dict(sha=commit,force=False))
for path,b in files.items():
 h=hashlib.sha1(('blob '+str(len(b))+'\0').encode()+b).hexdigest()
 assert base64.b64decode(api('GET','/git/blobs/'+h)['content'])==b,path
final=dict(status='ALL_STATISTICS_FULL_ARCHIVES_AND_HUMAN_CATALOGS_PUBLIC_VERIFIED',
 commit=commit,files=len(files),public_assets=63,public_asset_bytes=receipt['total_bytes'],
 release_url=release,report='https://github.com/'+repo+'/blob/main/01_results/confirmation_statistics_20261125/VYSLEDKY.txt',
 model_catalog='https://github.com/'+repo+'/blob/main/02_models/confirmation20_20261125/README.md',no_PC_copy=True)
save(O/'publication_commit.json',final);print(json.dumps(final),flush=True)
