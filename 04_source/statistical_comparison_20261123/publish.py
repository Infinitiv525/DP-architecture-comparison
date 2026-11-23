"""Publish verified statistics directly from Perun; credential is stdin/RAM only."""
import base64,hashlib,json,sys,urllib.error,urllib.parse,urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
payload=json.load(sys.stdin);token=payload.pop("token")
root=Path("/mnt/project/perun2601449/dp_llm_cfd")
source=root/"test/architecture_statistics_20261123_v1"
out=root/"results/architecture_statistics_20261123_v1"
repo="Infinitiv525/DP-architecture-comparison"
base="https://api.github.com/repos/"+repo
tag="architecture-statistics-20261123"
release_url="https://github.com/"+repo+"/releases/tag/"+tag
git_folder="01_results/statistical_comparison_20261123"
def now():
    return datetime.now(ZoneInfo("Europe/Bratislava")).isoformat()
def api(method,path,data=None):
    headers={"Authorization":"Bearer "+token,"User-Agent":"DP-statistics-publication",
             "Accept":"application/vnd.github+json","Content-Type":"application/json"}
    req=urllib.request.Request(base+path,method=method,headers=headers,
        data=None if data is None else json.dumps(data).encode("utf-8"))
    with urllib.request.urlopen(req,timeout=90) as response:return json.load(response)
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
    return h.hexdigest()
def save(name,value):
    (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def checkpoint(phase,**fields):
    save("publication_progress.json",dict(phase=phase,at=now(),**fields))
    print(json.dumps(dict(phase=phase,**fields)),flush=True)
def commit_files(entries,message):
    head=api("GET","/git/ref/heads/main")["object"]["sha"]
    commit=api("GET","/git/commits/"+head)
    tree=api("POST","/git/trees",dict(base_tree=commit["tree"]["sha"],tree=entries))
    new=api("POST","/git/commits",dict(message=message,tree=tree["sha"],parents=[head]))
    assert api("GET","/git/ref/heads/main")["object"]["sha"]==head,"Concurrent main update; retry with fresh head."
    api("PATCH","/git/refs/heads/main",dict(sha=new["sha"],force=False))
    return new["sha"]
def entry(path,text):
    assert "\ufffd" not in text, "Invalid UTF-8 replacement character"
    return dict(path=path,mode="100644",type="blob",content=text)

assert json.loads((out/"independent_verification.json").read_text())["status"]=="INDEPENDENT_STATISTICAL_VERIFICATION_PASS"
verification=json.loads((out/"archive_verification.json").read_text())
assert verification["status"]=="ALL_ARCHIVE_MEMBERS_FULLY_READ_AND_HASH_VERIFIED"
for a in verification["assets"]:
    path=out/"archive"/a["name"]
    assert path.stat().st_size==a["bytes"] and sha(path)==a["sha256"]
assert not (out/"publication_receipt.json").exists(),"Already published; verify existing receipt instead."
entries=[]
for path in sorted(out.iterdir()):
    if path.is_file() and path.suffix in [".json",".csv",".txt",".md"] and path.name!="publication_progress.json":
        entries.append(entry(git_folder+"/"+path.name,path.read_text(encoding="utf-8")))
for path in sorted(source.iterdir()):
    if not path.is_file():continue
    if path.name=="protokol.txt":
        dest="03_methodology/statistical_comparison_20261123/"+path.name
    else:
        dest="04_source/statistical_comparison_20261123/"+path.name
    entries.append(entry(dest,path.read_text(encoding="utf-8")))
readme_blob=api("GET","/contents/README.md")
readme=base64.b64decode(readme_blob["content"]).decode("utf-8")
marker="## Štatistické porovnanie architektúr"
assert marker not in readme
readme+="\n"+marker+" — dokončené 24. 11. 2026\n\n"
readme+="[Čitateľný výsledok, všetky intervaly a kontroly](01_results/statistical_comparison_20261123/README.md). "
readme+="Dodatočná analýza zahŕňa päť tréningov každej metódy, 200 TEST prípadov a 85 geometrických skupín, "
readme+="200 000 krížených bootstrap replík a opravu na viacnásobné porovnávanie. "
readme+="TFNO má v tejto približnej analýze nižšie L2 chyby než všetkých osem hlavných alternatív; "
readme+="RealNPU je z nich najsilnejšie podľa oboch L2 metrík. Rozdiel TFNO–dense FNO nie je stabilne preukázaný. "
readme+="Obmedzenie iba piatich tréningov, posthoc charakter, samostatné kohorty aj nesplnenia 5 ms zostávajú uvedené. "
readme+="[Úplné resampling dáta a zdroje]("+release_url+").\n"
entries.append(entry("README.md",readme))
results_commit=commit_files(entries,"Complete post-hoc architecture statistics with multiplicity and robustness checks")
checkpoint("SMALL_RESULTS_COMMITTED",commit=results_commit,files=len(entries))
try:
    api("GET","/releases/tags/"+tag)
except urllib.error.HTTPError as error:
    if error.code!=404:raise
else:
    raise RuntimeError("Release tag already exists; inspect instead of replacing it.")
body="""Completed post-hoc architecture comparison, computed on Perun with no GPU inference or retraining.
Nine primary methods x five training repeats, 200 TEST cases in 85 geometry clusters.
Includes all 200000 bootstrap replicas and resampling weights, independent audit, exact small-seed sensitivity, code and configuration.
The inferential coverage is approximate with only five repeats; supplemental FNO/PINN and second-chance results are kept separate.
Every archive member was unpacked and SHA256 verified. See ARCHIVE_MANIFEST.json and SHA256SUMS.txt.
The tarball is a snapshot before publication; the repository publication receipt records final public readback."""
release=api("POST","/releases",dict(tag_name=tag,target_commitish=results_commit,
    name="Architecture statistics: full results and reproducibility archive",body=body,draft=True,prerelease=False))
uploaded=[]
for item in verification["assets"]:
    path=out/"archive"/item["name"]
    url=release["upload_url"].split("{")[0]+"?name="+urllib.parse.quote(path.name)
    headers={"Authorization":"Bearer "+token,"User-Agent":"DP-statistics-publication",
        "Content-Type":"application/gzip" if path.suffix==".gz" else "application/octet-stream",
        "Content-Length":str(path.stat().st_size),"Accept":"application/vnd.github+json"}
    with path.open("rb") as stream:
        request=urllib.request.Request(url,data=stream,method="POST",headers=headers)
        with urllib.request.urlopen(request,timeout=180) as response:asset=json.load(response)
    assert asset["size"]==item["bytes"] and asset["state"]=="uploaded"
    if asset.get("digest"):
        assert asset["digest"]=="sha256:"+item["sha256"]
    uploaded.append(dict(**item,url=asset["browser_download_url"],asset_id=asset["id"],
                         github_digest=asset.get("digest")))
    checkpoint("ASSET_UPLOADED",name=path.name,bytes=item["bytes"],uploaded=len(uploaded))
api("PATCH","/releases/"+str(release["id"]),dict(draft=False,target_commitish=results_commit))
checkpoint("RELEASE_PUBLIC",release_url=release_url)
for item in uploaded:
    # Deliberately anonymous download: no Authorization follows redirects.
    request=urllib.request.Request(item["url"],headers={"User-Agent":"DP-anonymous-archive-verification"})
    h=hashlib.sha256();size=0
    with urllib.request.urlopen(request,timeout=180) as response:
        for block in iter(lambda:response.read(1024*1024),b""):
            h.update(block);size+=len(block)
    assert size==item["bytes"] and h.hexdigest()==item["sha256"]
    item["anonymous_full_readback_verified"]=True
    checkpoint("ANONYMOUS_FULL_READBACK_PASS",name=item["name"],bytes=size)
receipt=dict(status="STATISTICS_AND_FULL_ARCHIVE_PUBLIC_AND_READBACK_VERIFIED",
    completed_at=now(),repository="https://github.com/"+repo,results_commit=results_commit,
    report="https://github.com/"+repo+"/blob/main/"+git_folder+"/VYSLEDKY.txt",
    release_url=release_url,assets=uploaded,no_PC_copy=True,GPU_used=False,
    original_metrics_and_weights_preserved=True,independent_verification="PASS",
    main_contrasts=108,supplementary_contrasts=57,second_chance_conditional_contrasts=6)
save("publication_receipt.json",receipt)
status=json.loads((out/"completion_status.json").read_text())
status.update(status="STATISTICS_AND_COMPLETE_PUBLIC_ARCHIVE_VERIFIED",
    publication_pending=False,release_url=release_url,public_archive_assets=len(uploaded),
    anonymous_full_readback_verified=True,no_PC_copy=True,publication_completed_at=receipt["completed_at"])
save("completion_status.json",status)
final_commit=commit_files([
    entry(git_folder+"/publication_receipt.json",json.dumps(receipt,ensure_ascii=False,indent=2)+"\n"),
    entry(git_folder+"/completion_status.json",json.dumps(status,ensure_ascii=False,indent=2)+"\n")
], "Verify complete public statistics archive and anonymous byte-for-byte readback")
# Check final repository contents independently of the Git ref response.
for name in ["VYSLEDKY.txt","publication_receipt.json","completion_status.json"]:
    remote=api("GET","/contents/"+git_folder+"/"+name+"?ref="+final_commit)
    assert base64.b64decode(remote["content"])==(out/name).read_bytes()
final=dict(status="ALL_PUBLICATIONS_COMPLETE",commit=final_commit,release=release_url,
    report=receipt["report"],files_in_results_commit=len(entries),assets=len(uploaded),
    no_PC_copy=True,source_and_results_verified=True)
save("publication_commit.json",final)
print(json.dumps(final),flush=True)
