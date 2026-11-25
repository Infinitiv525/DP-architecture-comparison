"""Publish verified inference results; large payloads remain separate release assets."""
import base64,hashlib,json,sys,urllib.request
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
token=json.load(sys.stdin)['token'];repo='Infinitiv525/DP-architecture-comparison'
R=Path('/mnt/project/perun2601449/dp_llm_cfd');S=R/'test/architecture_confirmation_statistics_20261125_v1';O=R/'results/architecture_confirmation_statistics_20261125_v1'
T=R/'results/architecture_confirmation_test_20261125_v2';TR=R/'results/architecture_confirmation_20261124_v2'
def read(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def api(method,path,data=None):
 req=urllib.request.Request('https://api.github.com/repos/'+repo+path,method=method,
  headers={'Authorization':'Bearer '+token,'User-Agent':'DP-confirmation-statistics','Accept':'application/vnd.github+json','Content-Type':'application/json'},
  data=None if data is None else json.dumps(data,ensure_ascii=False).encode())
 with urllib.request.urlopen(req,timeout=120) as h:return json.load(h)
assert read(O/'independent_verification.json')['status']=='INDEPENDENT_INPUT_PRIMARY_EXACT_SIGN_AND_BOOTSTRAP_AUDIT_PASS'
now=datetime.now(ZoneInfo('Europe/Bratislava')).isoformat()
tag='architecture-confirmation20-complete-20261125'
release='https://github.com/'+repo+'/releases/tag/'+tag
(O/'ARCHIV.txt').write_text('''ÚPLNÝ ARCHÍV NOVÉHO POTVRDENIA
Verejný cieľ archívu:
'''+release+'''
Štatistika je dokončená a nezávisle overená. Úplné balíky sa pripravujú a ich verejné nahratie sa overí osobitne; toto nie je potvrdenie dokončeného uploadu.
Plán:60balíkov, každý obsahuje best/last váhy, optimizer, celú históriu tréningu, všetkých200TESTpolí, časovania, chyby a obrazovú ukážku.
Samostatný numerický balík obsahuje kompletné vstupné metriky,všetkých200000bootstrapreplík,zdroje a audity.
ARTIFACT_MANIFEST.json aSHA256SUMS.txt určia každý súbor a jeho kontrolný súčet.
Je to snapshot pred publikáciou. Konečný stav a verejné spätné overenie určí publication_receipt.json a modelový katalóg po dokončení.
Na PC sa objemné dáta nekopírujú.
''',encoding='utf-8')
status=dict(status='ALL60_TRAINING_TEST_AUDIT_AND_STATISTICS_COMPLETE_ARCHIVE_IN_PROGRESS',
 recorded_at=now,models=60,independent_seeds_per_method=20,native_fields_audited=12000,
 primary_contrasts=9,Holm_significant=8,all60_GPU_p99_5ms_pass=True,
 no_new_training=True,full_public_archive_pending=True)
save(T/'status.json',status);save(TR/'status.json',status)
with (TR/'dokumentacia.txt').open('a',encoding='utf-8') as f:f.write('\n25.11.: TEST60/60 aCPUaudit12000poli/12milionovcasovani presli. Statistika20novychsemien:8z9Holmkontrastov,TFNOlepsieodFNOaRealNPUv3metrikach;MSEFNO–RealNPUnepreukazane. Nezavisleovereniett,p,H​olm,signflip,sign testabootstrap preslo. Geometricka citlivost jeosobitna aniejerovnakosilna. StatistikanevyuzilaGPU;priamyverejnyarchivvahapoli sapripravuje.\n'.replace('\u200b',''))
(T/'README.md').write_text('''# TEST60modelov dokončený
Všetkých60modelov×200prípadov=12000polí a12000000dvojíc časovaní prešlo nezávislýmCPUauditom.
[Čitateľný opis TESTu](audit_result.txt) · [Súhrn auditu](saved_output_audit.json) · [Metriky všetkých prípadov](case_metrics.csv).
Všetkých60modelov splnilo per-casep99<=5ms. Jednotlivé prekročenia sú uvedené v novom štatistickom reporte.
[Nové štatistické porovnanie20semien](../confirmation_statistics_20261125/README.md).
Úplný priamy archív nových váh a polí naGitHub sa dokončuje; aktuálny stav je v status.json.
''',encoding='utf-8')
files={}
for p in O.iterdir():
 if p.is_file() and p.suffix in ['.json','.txt','.md','.csv'] and p.name not in ['publication_progress.json','publication_receipt.json']:
  files['01_results/confirmation_statistics_20261125/'+p.name]=p.read_bytes()
for p in S.iterdir():
 if p.is_file() and p.suffix in ['.py','.sbatch','.json','.txt','.out','.err']:
  dest=('03_methodology/confirmation_statistics_20261125/' if p.name=='protokol.txt' else '04_source/confirmation_statistics_20261125/')+p.name
  files[dest]=p.read_bytes()
for name in ['README.md','status.json']:
 files['01_results/confirmation_test_20261125/'+name]=(T/name).read_bytes()
for name,source in [('saved_output_audit.json','summary.json'),('model_metrics.csv','model_metrics.csv'),('case_metrics.csv','case_metrics.csv'),('audit_result.txt','VYSLEDOK.txt')]:
 files['01_results/confirmation_test_20261125/'+name]=(T/'audit'/source).read_bytes()
for name in ['status.json','dokumentacia.txt']:files['01_results/confirmation_20261124/'+name]=(TR/name).read_bytes()
head=api('GET','/git/ref/heads/main')['object']['sha']
rd=api('GET','/contents/README.md?ref='+head);body=base64.b64decode(rd['content']).decode()
marker='## Nové potvrdenie20tréningov každej architektúry'
assert marker not in body
intro=marker+''' — dokončené25.11.2026

**60tréningov,12000TESTpolí a štatistické porovnanie sú hotové a nezávisle overené.**
Na pevnom benchmarku máTFNO nižšieMSE ajL2chyby neždenseFNO aRealNPU. Z9vopred určených kontrastov prešlo8Holm korekciou; MSEdenseFNO–RealNPU sa nepreukázala.
[Čitateľný výsledok a obmedzenia](01_results/confirmation_statistics_20261125/VYSLEDKY.txt) · [Úplné testy a intervaly](01_results/confirmation_statistics_20261125/paired_t_primary.json).
Záver cez tréningové semená platí pre200pevnýchTESTprípadov. Geometrická bootstrapcitlivosť je slabšia/závisí od variantu intervalu; všeobecnú prevahu na nevidených geometriách netvrdíme.
Všetkých60modelov splnilo per-casep99<=5ms. Štatistika nepoužilaGPU a nepridala tréningy. Verejný archív všetkých nových váh/polí sa dokončuje priamo zPerunu.

---

'''
files['README.md']=intro+body
files['README.md']=files['README.md'].encode()
tree=api('GET','/git/commits/'+head)['tree']['sha']
entries=[dict(path=p,mode='100644',type='blob',content=b.decode()) for p,b in sorted(files.items())]
tree=api('POST','/git/trees',dict(base_tree=tree,tree=entries))['sha']
commit=api('POST','/git/commits',dict(message='Complete20-seed architecture confirmation with nine preplanned tests and independent audit',tree=tree,parents=[head]))['sha']
assert api('GET','/git/ref/heads/main')['object']['sha']==head
api('PATCH','/git/refs/heads/main',dict(sha=commit,force=False))
for path,b in files.items():
 h=hashlib.sha1(('blob '+str(len(b))+'\0').encode()+b).hexdigest()
 got=api('GET','/git/blobs/'+h);assert base64.b64decode(got['content'])==b,path
receipt=dict(status='STATISTICAL_RESULTS_PUBLIC_AND_ALL_FILES_READ_BACK_VERIFIED',
 commit=commit,repository=repo,files=len(files),bytes=sum(map(len,files.values())),published_at=now,
 full_archive_pending=True,no_PC_copy=True,no_new_GPU=True)
save(O/'results_publication.json',receipt);print(json.dumps(receipt))
