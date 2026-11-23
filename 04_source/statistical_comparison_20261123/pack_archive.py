"""Build and fully verify a portable statistics archive on Perun."""
import hashlib,json,os,subprocess,tarfile,time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
root=Path("/mnt/project/perun2601449/dp_llm_cfd")
source=root/"test/architecture_statistics_20261123_v1"
out=root/"results/architecture_statistics_20261123_v1"
archive=out/"archive";archive.mkdir(exist_ok=False)
assert json.loads((out/"independent_verification.json").read_text())["status"]=="INDEPENDENT_STATISTICAL_VERIFICATION_PASS"
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for job,name in [("172503","dp-arch-stats"),("172504","dp-arch-stat-audit")]:
    for suffix in ["out","err"]:
        original=root/"logs"/(name+"_"+job+"."+suffix)
        assert original.exists()
        (out/("slurm_"+job+"."+suffix+".txt")).write_bytes(original.read_bytes())
    status=subprocess.run(["scontrol","show","job",job],capture_output=True,text=True)
    (out/("scheduler_"+job+".json")).write_text(json.dumps(dict(returncode=status.returncode,stdout=status.stdout,stderr=status.stderr),indent=2)+"\n")
archive_note="""ÚPLNÝ ARCHÍV ŠTATISTIKY ARCHITEKTÚR

Verejný Release:
https://github.com/Infinitiv525/DP-architecture-comparison/releases/tag/architecture-statistics-20261123

architecture_statistics_complete_20261123.tar.gz obsahuje kompletné
výsledky, vstupné metrické tenzory, všetkých200000 bootstrap replík
(vrátane variantu s nezávislými seedmi), váhy resamplovania, zdroje,
konfiguráciu, metodiku a nezávislé overenie. Obsahuje aj snapshot
pôvodného súhrnu55modelov, podľa ktorého sa overili všetky agregáty.
Pôvodné veľké predikčné polia ani modelové váhy sa neduplikujú;
sú už v skorších vydaniach toho istého repozitára.

Je to snapshot pred publikáciou. Najnovší stav publikácie a overenie
verejného spätného prečítania sú v publication_receipt.json v repozitári.
Súbor ARCHIVE_MANIFEST.json obsahuje SHA256 každého vnútorného súboru;
SHA256SUMS.txt kontroluje výsledné verejné prílohy.

Pre reprodukciu: Python3.12 a NumPy2.5.4. Stiahni zmrazené pôvodnéCSV
uvedené v input_manifest.json alebo príslušný commit repozitára.
Spusti analyze.py s --config config.json --published-summary
frozen_summary55.json --input-root cesta_k_repozitaru --output novy_priecinok.
Voľby a semeno ostanú nezmenené. Skript odmietne prepísať existujúci výsledok.
"""
(out/"ARCHIV.txt").write_text(archive_note,encoding="utf-8")
entries=[]
for parent,prefix in [(source,"source"),(out,"results")]:
    for p in sorted(parent.iterdir()):
        if not p.is_file():
            continue
        assert p.suffix in [".py",".txt",".json",".sbatch",".csv",".md",".npz"]
        entries.append(dict(path=prefix+"/"+p.name,bytes=p.stat().st_size,sha256=sha(p),source=str(p)))
manifest=dict(created_at=datetime.now(ZoneInfo("Europe/Bratislava")).isoformat(),
    status="ALL_ARCHIVE_MEMBERS_SELECTED",no_PC_copy=True,files=[{k:v for k,v in e.items() if k!="source"} for e in entries])
manifest_path=archive/"ARCHIVE_MANIFEST.json"
manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
target=archive/"architecture_statistics_complete_20261123.tar.gz"
with tarfile.open(target,"w:gz",compresslevel=1) as tar:
    for entry in entries:
        tar.add(entry["source"],arcname=entry["path"],recursive=False)
expected={e["path"]:e for e in entries};seen=set()
with tarfile.open(target,"r:gz") as tar:
    for member in tar:
        assert member.isfile() and member.name in expected and member.name not in seen
        seen.add(member.name)
        assert member.size==expected[member.name]["bytes"]
        h=hashlib.sha256()
        handle=tar.extractfile(member)
        for block in iter(lambda:handle.read(1024*1024),b""):h.update(block)
        assert h.hexdigest()==expected[member.name]["sha256"]
assert seen==set(expected)
sums=archive/"SHA256SUMS.txt"
sums.write_text("".join(sha(p)+"  "+p.name+"\n" for p in [target,manifest_path]))
receipt=dict(status="ALL_ARCHIVE_MEMBERS_FULLY_READ_AND_HASH_VERIFIED",
    created_at=datetime.now(ZoneInfo("Europe/Bratislava")).isoformat(),members=len(entries),
    uncompressed_bytes=sum(e["bytes"] for e in entries),
    assets=[dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in [target,manifest_path,sums]],
    GPU_used=False,no_PC_copy=True)
(out/"archive_verification.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt),flush=True)
