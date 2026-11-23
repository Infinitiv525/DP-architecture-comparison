"""Generate a human report without changing frozen numerical results."""
import csv,json,sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
out=Path(sys.argv[1])
verification=json.loads((out/"independent_verification.json").read_text())
assert verification["status"]=="INDEPENDENT_STATISTICAL_VERIFICATION_PASS"
rows=json.loads((out/"contrasts.json").read_text())
summary=json.loads((out/"summary.json").read_text())
descriptions=json.loads((out/"method_summary.json").read_text())
latency=json.loads((out/"latency_descriptive.json").read_text())
second=json.loads((out/"second_chance_conditional.json").read_text())
concentration=json.loads((out/"error_concentration.json").read_text())
lookup={(r["method"],r["metric"]):r for r in descriptions}
time_lookup={r["method"]:r for r in latency}
methods=["tfno","chebkan","pinet","dendrite","realnpu","rbf","elm","scn","cascor","fno","pinn"]
names=dict(tfno="TFNO",chebkan="ChebKAN",pinet="PiNet",dendrite="Dendrite",
    realnpu="RealNPU",rbf="RBF",elm="ELM",scn="SCN",cascor="CasCor",fno="dense FNO*",pinn="PINN**")
metrics=["all_cell_mse","fluid_velocity_relative_l2","fluid_pressure_relative_l2"]
labels=dict(zip(metrics,["MSE u/v/p","L2 rýchlosti","L2 tlaku"]))
def same_sign(a,b):
    return bool(np.sign(a)==np.sign(b) and a!=0 and b!=0)
interpretation=[]
for row in rows:
    point=row["difference_a_minus_b"]
    ci=row["ci_family_basic"]
    independent=row["independent_seeds_ci_family_basic"]
    percentile=row["ci_family_percentile"]
    direction_consistent=all(same_sign(point,v) for v in [*ci,*independent,*percentile,row["equal_geometry_difference"]])
    if row["leave_one_seed_out_min"] is not None:
        direction_consistent &= same_sign(point,row["leave_one_seed_out_min"]) and same_sign(point,row["leave_one_seed_out_max"])
    stable=bool(row["basic_bonferroni_excludes_zero"] and row["percentile_bonferroni_excludes_zero"]
        and row["independent_seeds_bonferroni_excludes_zero"] and row["half_decisions_agree"]
        and row["half_directions_agree"] and direction_consistent)
    if stable:
        category="STABLE_APPROXIMATE_BOOTSTRAP_SUPPORT"
    elif row["basic_bonferroni_excludes_zero"]:
        category="SENSITIVE_APPROXIMATE_BOOTSTRAP_SUPPORT"
    else:
        category="NOT_ESTABLISHED_BY_CONSERVATIVE_INTERVAL"
    interpretation.append(dict(method_a=row["method_a"],method_b=row["method_b"],metric=row["metric"],
        family=row["family"],interpretation=category,stable_across_reported_controls=stable,
        smaller_observed_error=row["method_a"] if point<0 else row["method_b"],
        not_a_confirmatory_or_universal_claim=True))
(out/"interpretation.json").write_text(json.dumps(interpretation,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
with (out/"interpretation.csv").open("w",encoding="utf-8",newline="") as h:
    w=csv.DictWriter(h,fieldnames=list(interpretation[0]));w.writeheader();w.writerows(interpretation)
def pair(a,b,key):
    for r in rows:
        if r["metric"]==key and {r["method_a"],r["method_b"]}=={a,b}:
            reverse=r["method_a"]!=a
            if reverse:
                return -r["difference_a_minus_b"],[-r["ci_family_basic"][1],-r["ci_family_basic"][0]],r
            return r["difference_a_minus_b"],r["ci_family_basic"],r
    raise KeyError((a,b,key))
finished=datetime.now(ZoneInfo("Europe/Bratislava")).isoformat()
lines=["ŠTATISTICKÉ POROVNANIE ARCHITEKTÚR — DOKONČENÉ",
       "Analýza začatá 23. 11. 2026; report dokončený "+finished+".","",
       "HLAVNÝ VÝSLEDOK",
       "V nominálnej analýze s opravou na 108 porovnaní má TFNO nižšiu",
       "relatívnu L2 chybu rýchlosti aj tlaku než všetkých osem hlavných",
       "alternatív. RealNPU má najnižšie priemerné L2 chyby spomedzi týchto",
       "ôsmich alternatív a jeho rozdiely oproti zvyšným siedmim sú podporené",
       "rovnakou analýzou aj kontrolami citlivosti. Závery sú o konkrétnych",
       "implementáciách pri rovnakom maximálnom tréningovom rozpočte.",
       "",
       "DÔLEŽITÁ HRANICA ISTOTY",
       "Ide o dodatočnú exploratívnu analýzu už známeho TESTu. Bootstrap",
       "je približný a máme iba päť tréningov každej metódy. Bonferroni ani",
       "Holm nevyriešia prípadnú nepresnosť základného odhadu neistoty.",
       "Samostatná kontrola všetkých 32 znamienkových permutácií piatich",
       "seedových rozdielov má najmenšie možné obojstranné p=0,0625;",
       "žiadne porovnanie preto týmto malovzorkovým testom nedosiahne p<0,05.",
       "Bootstrap podporu nevydávame za exaktný alebo definitívny dôkaz.",
       "Počet 200000 resampling replík nepridáva ďalšie tréningy.",
       "",
       "ROZSAH A POSTUP",
       "Hlavná kohorta: 9 architektúr × 5 tréningov, každý model na rovnakých",
       "200 TEST prípadoch z 85 geometrií. Súvisiace Re varianty geometrie",
       "zostávajú spolu. Resamplovali sa nezávisle seedové bloky a geometrie;",
       "rozdiely sa počítali párovane. Historické PINN seedy nie sú umelo",
       "spárované s novými architektúrami. Pôvodný priemer cez prípady ostal",
       "zachovaný. Celkovo 108 hlavných kontrastov (36 dvojíc ×3 metriky),",
       "57 samostatných doplnkových kontrastov a 6 podmienených kontrastov",
       "druhej šance. Žiadne nové učenie ani predikcie, GPU spotreba nula.",
       "Audit overil pôvodné CSV podľa git blob SHA/SHA256 aj pôvodné agregáty.",
       "",
       "OPISNÉ VÝSLEDKY — VŠETKÝCH PÄŤ TRÉNINGOV",
       "Metóda       MSE u/v/p       L2 rýchl. [%]   L2 tlak [%]   p99<=5ms GPU/host"]
for method in methods:
    v=[lookup[(method,k)]["mean"] for k in metrics]
    t=time_lookup[method]
    lines.append(f"{names[method]:12s} {v[0]:.8g}       {100*v[1]:8.4f}       {100*v[2]:8.4f}      {t['gpu_pass_models']}/5 | {t['host_pass_models']}/5")
lines += ["",
    "* Dense FNO bolo pridané po znalosti pôvodných TEST výsledkov.",
    "** PINN je historická pomocná referencia s odlišným tréningom.",
    "Latencia je najhoršie per-case p99, nie maximum každého volania.",
    "Vstupy už boli na H200; prenosy a vizualizácia sú mimo merania.",
    "Počet splnení 5 ms sa nevykladá ako presný odhad pravdepodobnosti",
    "splnenia na ľubovoľnom novom stroji alebo tréningu.",
    "",
    "HLAVNÉ KONTRASTY PROTI TFNO",
    "Rozdiel = alternatíva mínus TFNO; kladné číslo znamená vyššiu chybu",
    "alternatívy. L2 rozdiel aj interval sú v percentuálnych bodoch.",
    "Uvedené sú basic intervaly s nominálnym individuálnym pokrytím",
    "99,9537037 % (Bonferroni pre rodinu 108; nominálny rodinný cieľ 95 %).",
    "Nie je to tvrdenie presne overeného pokrytia pri n=5.",
    "Metóda       delta rýchlosti [CI]                    delta tlaku [CI]"]
for method in methods[1:9]:
    a,ci,_=pair(method,"tfno",metrics[1]);b,cj,_=pair(method,"tfno",metrics[2])
    lines.append(f"{names[method]:12s} {a*100:8.4f} [{ci[0]*100:.4f};{ci[1]*100:.4f}]    {b*100:8.4f} [{cj[0]*100:.4f};{cj[1]*100:.4f}]")
lines += ["",
    "Tieto L2 závery sa zhodujú aj pri nezávislom resamplovaní tréningov,",
    "pri alternate percentile CI a po rozdelení bootstrapu na dve polovice.",
    "Jednotlivé modelové priemery, SD a presné hodnoty sú v method_summary.csv.",
    "",
    "VŠETKY DVOJICE A METRIKY"]
for family,title in [("original_nine_methods","Hlavná rodina"),("posthoc_FNO_and_auxiliary_PINN","Doplnková rodina")]:
    rr=[r for r in rows if r["family"]==family]
    ii=[r for r in interpretation if r["family"]==family]
    lines.append(f"{title}: konzervatívny basic Bonferroni interval vylučuje nulu v {sum(r['basic_bonferroni_excludes_zero'] for r in rr)}/{len(rr)} kontrastoch.")
    lines.append(f"Stabilná podpora naprieč uvedenými kontrolami: {sum(r['stable_across_reported_controls'] for r in ii)}/{len(ii)}; Holm p<.05: {sum(r['holm_reject_alpha_05'] for r in rr)}/{len(rr)}.")
for metric in metrics:
    rr=[r for r in rows if r["family"]=="original_nine_methods" and r["metric"]==metric]
    lines.append(labels[metric]+f": basic Bonferroni {sum(r['basic_bonferroni_excludes_zero'] for r in rr)}/36, Holm {sum(r['holm_reject_alpha_05'] for r in rr)}/36.")
lines += ["",
    "Holmova step-down úprava je menej konzervatívna než jednotlivé",
    "Bonferroni intervaly. Ich odlišné počty nie sú chyba výpočtu.",
    "Na opatrné slovné závery sa používa vopred opísané prísnejšie kritérium",
    "a kontrola stability. Všetky p-hodnoty aj oba typy CI ostávajú viditeľné.",
    "MSE a relatívne L2 nie sú zameniteľné metriky. Napríklad rozdiel",
    "TFNO–RealNPU v MSE konzervatívny interval nepreukázal, hoci L2 áno.",
    "MSE niektorých metód je silno ovplyvnená niekoľkými geometriami;",
    "presné podiely zaznamenáva error_concentration.json.",
    "",
    "DENSE FNO A PINN — SAMOSTATNÝ DOPLNOK",
    "TFNO má opisne nižšie priemerné chyby než dense FNO. Konzervatívne",
    "intervaly pre MSE a L2 rýchlosti zahŕňajú nulu. Pri L2 tlaku je",
    "interval tesne pod nulou, ale záver sa zmení medzi polovicami",
    "bootstrapu a pri nezávislom výbere tréningov. Rozdiel TFNO–dense FNO",
    "preto neoznačujeme za stabilne preukázaný. Nejde o dôkaz ekvivalencie.",
    "TFNO aj dense FNO majú nižšie L2 chyby než historická PINN; odlišné",
    "učenie PINN a dodatočné zaradenie FNO obmedzujú výklad príčiny rozdielu.",
    "",
    "DRUHÁ ŠANCA CHEBKAN/RBF",
    "Priemer párových zmien na dvoch fixných tréningových líniách;",
    "resamplované sú len geometrie. Šesť Bonferroni kontrastov.",
    "Model        Met­rika          Relatívna zmena chyby     basic CI vylučuje nulu"]
for row in second:
    lines.append(f"{names[row['method']]:12s} {labels[row['metric']]:17s} {row['relative_difference_percent']:9.3f}%               {'áno' if row['basic_bonferroni_excludes_zero'] else 'nie'}")
lines += ["",
    "Zlepšenia L2 sú podporené podmienenými intervalmi pri oboch rodinách;",
    "pri MSE je záver pre ChebKAN citlivý na typ CI. Dva seedy nepostačujú",
    "na všeobecný záver o prínose predĺženia pre nové tréningy. Hlavné",
    "porovnanie pri 4h sa týmito pokračovaniami nemení.",
    "",
    "KDE SÚ PRESNÉ DÁTA A KONTROLY",
    "pairwise_comparisons.csv / contrasts.json — všetkých165 kontrastov,",
    "rozdiely, relatívne zmeny,95% aj upravené CI, p, Holm a citlivosť.",
    "interpretation.csv — opatrná klasifikácia podľa stability kontrol.",
    "exact_seed_sensitivity.csv — exaktné vymenovanie znamienok piatich blokov.",
    "second_chance_conditional.csv — oddelené predĺžené tréningy.",
    "case_cluster_map.csv —200 prípadov a85 geometrických skupín.",
    "independent_verification.json — nezávislá implementácia váženia,",
    "kontrola všetkých165 intervalov/p a úprav, overenie vstupných SHA.",
    "Bootstrap váhy, všetky modelové priemery replík, vstupné metrické",
    "tenzory a celé zdroje sú v samostatnom verejnom Release.",
    "Pôvodné výsledky ani váhy sa nemenili a doPC sa nič nekopírovalo.",
    "",
    "ZDROJE METÓD",
    "Podrobná metodika:03_methodology/statistical_comparison_20261123/protokol.txt",
    "Owen2007:https://arxiv.org/abs/0712.1111",
    "Owen/Eckles2012:https://arxiv.org/abs/1106.2125",
    "Holm1979:https://www.jstor.org/stable/4615733",
    "Hesterberg2015:https://arxiv.org/abs/1411.5279",
    "Konkrétne rodiny porovnaní a voľby analýzy sú naše posthoc rozhodnutia."]
(out/"VYSLEDKY.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
readme="""# Štatistické porovnanie architektúr

**[Výsledky ľudskou rečou](VYSLEDKY.txt)** · [Všetky kontrasty](pairwise_comparisons.csv) · [Súhrn metód](method_summary.csv)

Analýza zohľadňuje päť tréningov každej architektúry a 85 geometrických skupín v 200 TEST prípadoch. Používa 200 000 krížených bootstrap replík, Bonferroni intervaly a Holm upravené približné p-hodnoty. Je dodatočná a exploratívna; iba päť tréningov obmedzuje istotu záverov.

V rámci tejto analýzy TFNO dosiahlo nižšie relatívne L2 chyby rýchlosti aj tlaku než všetkých osem hlavných alternatív. RealNPU bolo najsilnejšou z ôsmich alternatív podľa oboch L2 metrík. Rozdiel TFNO oproti dodatočnému dense FNO nepovažujeme za stabilne preukázaný. Pôvodné váhy, metriky a merania ostali zachované.

- [Opatrná klasifikácia všetkých kontrastov](interpretation.csv)
- [Úplné presné intervaly, p-hodnoty a kontroly](contrasts.json)
- [Obmedzenie piatich seedov — znamienková kontrola](exact_seed_sensitivity.csv)
- [Druhá šanca — len dva konkrétne tréningové seedy](second_chance_conditional.csv)
- [Latencia — opisné vyhodnotenie](latency_descriptive.csv)
- [Nezávislé overenie výpočtov](independent_verification.json)
- [Vstupné SHA a nemenný pôvod údajov](input_manifest.json)
- [Podrobná metodika](../../03_methodology/statistical_comparison_20261123/protokol.txt)
- [Spustiteľný kód a nastavenie](../../04_source/statistical_comparison_20261123)
- [Všetky resampling dáta a zdroje v Release](https://github.com/Infinitiv525/DP-architecture-comparison/releases/tag/architecture-statistics-20261123)

Ani nominálny bootstrap interval, ani nezamietnutie nulovej hypotézy nie sú dôkazom ekvivalencie alebo univerzálnej nadradenosti rodiny sietí. Rozsahom sú tieto implementácie, známe geometrie FlowBench a konkrétny tréningový rozpočet.
"""
(out/"README.md").write_text(readme,encoding="utf-8")
status=dict(status="STATISTICS_AND_INDEPENDENT_VERIFICATION_COMPLETE",
    analysis_started="2026-11-23",human_report_finished_at=finished,main_contrasts=108,supplementary_contrasts=57,
    primary_stable_bootstrap_support=sum(r["stable_across_reported_controls"] for r in interpretation if r["family"]=="original_nine_methods"),
    supplementary_stable_bootstrap_support=sum(r["stable_across_reported_controls"] for r in interpretation if r["family"]!="original_nine_methods"),
    inference_is_posthoc_and_approximate=True,training_replicates_per_method=5,
    no_new_GPU_usage=True,no_new_training_or_inference=True,publication_pending=True)
(out/"completion_status.json").write_text(json.dumps(status,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(status,ensure_ascii=False))
