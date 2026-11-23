"""Resume publication verification from fresh public release metadata; RAM-only credential."""
import base64,hashlib,json,sys,time,urllib.error,urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
token=json.load(sys.stdin)["token"]
root=Path("/mnt/project/perun2601449/dp_llm_cfd")
source=root/"test/architecture_statistics_20261123_v1"
out=root/"results/architecture_statistics_20261123_v1"
repo="Infinitiv525/DP-architecture-comparison"
base="https://api.github.com/repos/"+repo
folder="01_results/statistical_comparison_20261123"
tag="architecture-statistics-20261123"
def now():return datetime.now(ZoneInfo("Europe/Bratislava")).isoformat()
def save(name,value):(out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def api(method,path,data=None):
    request=urllib.request.Request(base+path,method=method,
        data=None if data is None else json.dumps(data).encode(),
        headers={"Authorization":"Bearer "+token,"User-Agent":"DP-statistics-finalization","Accept":"application/vnd.github+json","Content-Type":"application/json"})
    with urllib.request.urlopen(request,timeout=90) as response:return json.load(response)
release=api("GET","/releases/tags/"+tag)
assert release["draft"] is False
expected=json.loads((out/"archive_verification.json").read_text())
assets={a["name"]:a for a in release["assets"]}
assert len(assets)==len(expected["assets"])==3
uploaded=[]
for item in expected["assets"]:
    asset=assets[item["name"]]
    assert asset["size"]==item["bytes"] and asset["state"]=="uploaded"
    assert asset["digest"]=="sha256:"+item["sha256"]
    retries=[]
    for attempt in range(5):
        try:
            # Refresh URLs after publishing; do not use URLs retained from draft.
            release=api("GET","/releases/tags/"+tag)
            asset=next(a for a in release["assets"] if a["name"]==item["name"])
            h=hashlib.sha256();size=0
            request=urllib.request.Request(asset["browser_download_url"],headers={"User-Agent":"DP-anonymous-full-readback"})
            with urllib.request.urlopen(request,timeout=180) as response:
                for block in iter(lambda:response.read(1024*1024),b""):
                    h.update(block);size+=len(block)
            assert size==item["bytes"] and h.hexdigest()==item["sha256"]
            break
        except urllib.error.HTTPError as error:
            retries.append(error.code)
            if error.code not in [404,429,500,502,503,504] or attempt==4:raise
            time.sleep(min(2**attempt,8))
    uploaded.append(dict(**item,url=asset["browser_download_url"],asset_id=asset["id"],
        github_digest=asset["digest"],anonymous_full_readback_verified=True,retried_http_statuses=retries))
    print(json.dumps(dict(status="ANONYMOUS_FULL_READBACK_PASS",name=item["name"],bytes=size)),flush=True)
first_attempt=dict(status="INITIAL_PUBLICATION_FINALIZATION_HTTP_404",
    results_commit=release["target_commitish"],all_three_assets_already_uploaded=True,
    release_already_public=True,action="Read fresh published asset URLs and verify again",
    no_new_training_or_analysis=True,no_asset_reupload=True)
save("publication_first_attempt.json",first_attempt)
receipt=dict(status="STATISTICS_AND_FULL_ARCHIVE_PUBLIC_AND_READBACK_VERIFIED",completed_at=now(),
    repository="https://github.com/"+repo,results_commit=release["target_commitish"],
    report="https://github.com/"+repo+"/blob/main/"+folder+"/VYSLEDKY.txt",
    release_url=release["html_url"],assets=uploaded,no_PC_copy=True,GPU_used=False,
    original_metrics_and_weights_preserved=True,independent_verification="PASS",
    main_contrasts=108,supplementary_contrasts=57,second_chance_conditional_contrasts=6,
    initial_publication_http_404_preserved=True,
    no_scientific_result_rerun_or_asset_replacement=True)
save("publication_receipt.json",receipt)
status=json.loads((out/"completion_status.json").read_text())
status.update(status="STATISTICS_AND_COMPLETE_PUBLIC_ARCHIVE_VERIFIED",publication_pending=False,
    release_url=release["html_url"],public_archive_assets=3,anonymous_full_readback_verified=True,
    no_PC_copy=True,publication_completed_at=receipt["completed_at"])
save("completion_status.json",status)
head=api("GET","/git/ref/heads/main")["object"]["sha"]
commit=api("GET","/git/commits/"+head)
entries=[]
for filename in ["publication_receipt.json","completion_status.json","publication_first_attempt.json"]:
    entries.append(dict(path=folder+"/"+filename,mode="100644",type="blob",content=(out/filename).read_text()))
entries.append(dict(path="04_source/statistical_comparison_20261123/publish_finalize_v2.py",
    mode="100644",type="blob",content=(source/"publish_finalize_v2.py").read_text()))
tree=api("POST","/git/trees",dict(base_tree=commit["tree"]["sha"],tree=entries))
new=api("POST","/git/commits",dict(message="Verify public statistics archive with fresh asset URLs and full anonymous readback",tree=tree["sha"],parents=[head]))
assert api("GET","/git/ref/heads/main")["object"]["sha"]==head
api("PATCH","/git/refs/heads/main",dict(sha=new["sha"],force=False))
for name in ["VYSLEDKY.txt","publication_receipt.json","completion_status.json"]:
    remote=api("GET","/contents/"+folder+"/"+name+"?ref="+new["sha"])
    assert base64.b64decode(remote["content"])==(out/name).read_bytes()
final=dict(status="ALL_PUBLICATIONS_COMPLETE",commit=new["sha"],release=release["html_url"],
    report=receipt["report"],assets=3,archive_members=expected["members"],
    no_PC_copy=True,source_and_results_verified=True)
save("publication_commit.json",final)
print(json.dumps(final),flush=True)
