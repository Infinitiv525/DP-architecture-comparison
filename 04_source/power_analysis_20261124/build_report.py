"""Human-readable prospective power recommendation; no new training."""
import csv,json,math,re,sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
out=Path(sys.argv[1])
assert json.loads((out/"independent_verification.json").read_text())["status"]=="INDEPENDENT_POWER_AND_COST_AUDIT_PASS"
designs=json.loads((out/"sample_size_design_summary.json").read_text())
options=list(csv.DictReader((out/"cost_options.csv").open()))
def scenario(key,effect,power):
    return next(r for r in designs if r["design"]==key and r["relative_effect"]==effect and r["target_power"]==power and r["sd_scenario"]=="upper95_pilot_sd")
def option(key,n,reuse=False):
    row=next(r for r in options if r["design"]==key and int(r["training_pairs_n"])==n and (r["pilot_reused"]=="True")==reuse)
    return {k:float(v) if k not in ["design","pilot_reused"] else v for k,v in row.items()}
core=scenario("core_three_all_metrics",.1,.9)
full=scenario("ten_all_metrics_L2",.2,.9)
baseline=scenario("ten_against_TFNO_all_metrics_L2",.2,.9)
assert core["maximum_required_n"]==19 and full["maximum_required_n"]==30 and baseline["maximum_required_n"]==24
recommended=option("core_three_all_metrics",20)
expanded=option("core_three_all_metrics",20,True)
broad=option("ten_all_metrics_L2",30)
finished=datetime.now(ZoneInfo("Europe/Bratislava")).isoformat()
recommendations=dict(status="POWER_ANALYSIS_VERIFIED_RECOMMENDATION_NOT_TRAINING_AUTHORIZATION",
    completed_at=finished,recommended_scope="Separate focused confirmation of TFNO, dense FNO and RealNPU on the fixed existing benchmark",
    original_nine_family_experiment_remains_unchanged=True,
    focused=dict(methods=["tfno","fno","realnpu"],metrics=3,multiple_comparisons=9,
        target_power=.9,symmetric_relative_effect=.1,
        minimum_n_under_assumptions=19,recommended_new_n_per_method=20,**recommended),
    broad=dict(methods=full["methods"],primary_metrics=["fluid_velocity_relative_l2","fluid_pressure_relative_l2"],
        secondary_metric="all_cell_mse",multiple_primary_comparisons=90,
        target_power=.9,symmetric_relative_effect=.2,
        recommended_new_n_per_method=30,**broad),
    pooled_exploratory_alternative=dict(explanation="Existing five pilot runs plus fifteen new runs per selected method; not an independent twenty-run cohort.",**expanded),
    no_new_training_submitted=True,
    conditions=["Power is model-based and conditional on hypothetical true effect and normal seed-difference model.",
                "Upper one-sided95% SD bound from five pilot runs; not simultaneous95% protection of all SD estimates.",
                "Fixed200TEST benchmark. Geometry-population uncertainty does not vanish with more seeds.",
                "Choosing a focused or L2-primary follow-up must be documented before new runs; it does not rewrite the old protocol."])
(out/"recommendation.json").write_text(json.dumps(recommendations,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
lines=["ANALÝZA ŠTATISTICKEJ SILY A ODHAD GPU HODÍN",
       "Dokončené "+finished,"",
       "ODPORÚČANIE",
       "Najrozumnejší cielený dodatok je20NOVÝCH tréningov každej z metód",
       "TFNO, dense FNO a RealNPU: spolu60nových sietí. Odhad je243GPU-h",
       "vrátane rovnakého záverečného testovania a časovania; plánovať260GPU-h.",
       "Výpočet pri nižšie uvedených predpokladoch dáva minimum19;20je",
       "prehľadné zaokrúhlenie s jedným dodatočným opakovaním.",
       "Tento dodatok by samostatne posilnil tri kľúčové metódy. Nezmení",
       "ostatných sedem architektúr na rovnako rozsiahlo potvrdené metódy.",
       "Pôvodné porovnanie9rodín po5behov ostáva zachované a samostatné.",
       "",
       "AK CHCEME ROVNAKO POSILNIŤ VŠETKÝCH10ARCHITEKTÚR",
       "Pre všetky dvojice a dve primárne L2 metriky vychádza30NOVÝCH behov",
       "na metódu:300nových tréningov. Odhad vrátane TEST je977GPU-h;",
       "konzervatívny tréningový strop a rezerva TEST spolu1312,5GPU-h.",
       "Plán je na90%silu pre20%rozdiel v symetrickej definícii nižšie.",
       "MSE by v tomto NOVOM protokole bola sekundárna a stále zverejnená.",
       "Týmto sa nemení ani nezjednodušuje pôvodná trojmetrická analýza.",
       "Možnosť rozšíriť všetkých10iba oproti TFNO má18L2 kontrastov:",
       "minimum24nových behov/metódu,240modelov,odhad781GPU-h,rezerva1050GPU-h.",
       "",
       "ČO ZNAMENÁ SILA A VEĽKOSŤ ROZDIELU",
       "90%sila znamená:ak by skutočný rozdiel bol aspoň taký veľký, aký",
       "plánujeme zachytiť,a model variability platil,test by ho odhalil",
       "približne v90zo100opakovaných rovnako navrhnutých štúdií.",
       "Nie je to90%pravdepodobnosť,že hypotéza alebo záver je pravdivý.",
       "Je to cieľ každého kontrastu,nie90%šanca zamietnuť naraz všetky testy.",
       "Efekt r sa definuje delta=r*(mean(A)+mean(B))/2. Ide o približne",
       "5%,10%alebo20%relatívny rozdiel chýb,nie o toľko percentuálnych bodov.",
       "Napríklad chyby1,00a0,90majú podľa tejto definície rozdiel10,526%.",
       "Pracovné efekty nie sú nové požiadavky na absolútnu presnosť siete.",
       "",
       "Z ČOHO SA POČÍTALO",
       "Všetkých50kontrolovane trénovaných modelov (45hlavných +5dense FNO),",
       "5párovaných seedových blokov,priemery na pevne daných200TEST prípadoch.",
       "Historická PINN ostáva pomocnou referenciou; nový tréning sa jej",
       "v týchto scenároch nepriraďuje.",
       "Zo starých dát sa odhadovala variabilita,nie pravdivosť pozorovaného",
       "rozdielu. Nejde o nesprávne spätne vypočítanú 'dosiahnutú silu'.",
       "Počet behov sa riešil pomocou necentrálneho t-rozdelenia,obojstranne,",
       "Bonferroni alpha=.05/počet kontrastov. Predpokladom plánovania je",
       "približne normálne rozdelenie rozdielov medzi tréningovými seedmi.",
       "Pri iba5pilotných behoch je odhad SD neistý.Preto odporúčané scenáre",
       "používajú hornú jednostrannú95%hranicu SD:2,3724násobok pilotnej SD.",
       "Táto horná hranica je pre každý kontrast osobitne;nie je súčasnou",
       "95%zárukou pre všetky SD ani zárukou proti inému tvaru rozdelenia.",
       "Simulácia šikmých rozdielov ukazuje,že porušenie predpokladov môže",
       "zmeniť chybovosť testu; čísla sú plánovacie odhady,nie garancie.",
       "",
       "PREHĽAD CIELENÉHO DODATKU —3METÓDY,3METRIKY,9KONTRASTOV",
       "Rozdiel   Sila  Minimum nových behov/metódu  Nové modely  Odhad GPU-h  Strop+rezerva"]
for effect in [.05,.1,.2]:
    for power in [.8,.9]:
        r=scenario("core_three_all_metrics",effect,power);v=r["new_independent_cohort_n"]
        lines.append(f"{effect*100:5.0f}%    {power*100:4.0f}%            {r['maximum_required_n']:4d}                {v['new_models']:4d}       {v['total_expected_GPU_h']:8.1f}       {v['training_cap_plus_TEST_reserve_GPU_h']:8.1f}")
lines+=["",
       "Pre vybraný odporúčaný N20:60modelov,235,05GPU-h odhad tréningov,",
       "8,37GPU-h odhad TEST,spolu243,42GPU-h. Tvrdý tréningový strop240h",
       "a plánovacia rezerva TEST20h dávajú260GPU-h.",
       "",
       "PREČO SA NEODPORÚČA NASLEPO DOROVNAŤ CELÝ REBRÍČEK",
       "Požadovať rovnakú vysokú silu aj pre malé hypotetické rozdiely MSE",
       "v každej dvojici je extrémne drahé,lebo niektoré MSE výrazne kolíšu.",
       "Napríklad všetkých10metód,všetky3metriky,20%efekt a80%sila",
       "pri hornom odhade SD vyžaduje najhorší kontrast1354behov/metódu.",
       "Nie je to potrebný počet na obhajobu dizertácie.Je to cena veľmi",
       "silnej plošnej požiadavky,ktorú pre túto prácu neodporúčame.",
       "Nie je cieľom vynútiť významnosť každého páru za ľubovoľnú cenu.",
       "Súčasné výsledky sa nestávajú neplatné len preto,že plánovanie",
       "malých budúcich rozdielov by vyžadovalo veľa tréningov.",
       "",
       "DVE RÔZNE NEISTOTY",
       "Výpočet N vyššie rieši variabilitu učenia na TOM ISTOM200-prípadovom",
       "benchmarku. Závery budú podmienené týmito dátami,implementáciami",
       "a rovnakým štvorhodinovým rozpočtom na jednu H200.",
       "Pri zovšeobecnení na populáciu geometrií zostáva aj neistota z85",
       "geometrických skupín. Ďalšie tréningy ďalšie geometrie nepridávajú.",
       "Doplnkový dvojfaktorový model naznačuje,že pri viacerých10%efektoch",
       "je geometrická neistota dominantná a samotný počet seedov nestačí.",
       "Tento model je orientačný a pri niektorých kontrastoch dáva iné SE",
       "než pôvodný bootstrap;nejde o presný dôkaz nemožnosti zlepšenia.",
       "Preto nesľubujeme,že60alebo300nových tréningov preukáže všeobecnú",
       "nadradenosť na úplne nových geometriách. Nové CFD sa nenavrhuje.",
       "",
       "NOVÁ NEZÁVISLÁ KOHORTA OPROTI POUŽITIU STARÝCH5",
       "Odporúčaných20na metódu znamená20NOVÝCH seedov. Pôvodných5",
       "už slúžilo na výber metód a plánovanie,a ostanú prieskumnou sadou.",
       "Ak by sa len doplnilo na20vrátane pôvodných5,pri troch metódach",
       "by to bolo45nových tréningov,odhad182,56GPU-h a rezerva195GPU-h.",
       "Takýto spoločný súhrn by bol rozšírenou exploratívnou analýzou;",
       "nevydával by sa za nezávislé20-behové potvrdenie.",
       "Ani nová kohorta nie je nový slepý TEST geometrie;používa tie isté",
       "známe dáta a samostatne preveruje opakovateľnosť nových tréningov.",
       "",
       "AKO SA OVERILI NÁKLADY A VÝPOČET",
       "Načítané boli pôvodné complete,started a allocation_closeout všetkých",
       "50behov aj ich finálne TEST súhrny. Rozlišuje sa zaznamenaný runtime",
       "alokácie od finálneho portálového vyúčtovania a od nákladového stropu.",
       "ELM a SCN sú v odhade lacnejšie,pretože prirodzene skončili skôr.",
       "Tréningový limit zostáva4GPU-h/nový model. TEST rezerva je maximum",
       "doterajšej ceny metódy krát1,25 +3minúty,zaokrúhlené na štvrťhodinu.",
       "Je to plánovacia rezerva,nie absolútna záruka budúceho runtime.",
       "Cena zahŕňa rovnaké200TEST prípadov a1000časovaní/prípad;neplánuje",
       "sa nové ladenie,HPO,LLM generovanie ani nový tréning PINN.",
       "Nezávislý audit preveril96scenárov,113limitujúcich minimálnychN",
       "adaptívnou integráciou a všetky rozpočtové vzorce aj pôvodnéSHA.",
       "Pre veľkéN sa opravila numerická cesta pre NaN zo SciPy1.18.1;",
       "pôvodný neúspešný pokus je zachovaný a neovplyvnil vedecké dáta.",
       "",
       "ČO JE TERAZ HOTOVÉ",
       "Hotová je táto CPU analýza a návrhy rozsahu.ŽIADNE NOVÉ TRÉNINGY",
       "sa nespustili a analýza nespotrebovala GPU hodiny.",
       "Sila nepredstavuje univerzálnu hranicu obhájiteľnosti dizertácie.",
       "Každý prípadný dodatok dostane vopred uzavretý cieľ,metriky,seedy",
       "a počet behov;tréning sa nebude pridávať dovtedy,kým vyjde p<.05.",
       "",
       "SÚBORY A ZDROJE",
       "recommendation.json —presný odporúčaný rozsah a obe hlavné možnosti.",
       "sample_size_design_summary.csv —96scenárov s nákladmi.",
       "sample_size_all_contrasts.csv —všetky5580plánovacích kontrastov.",
       "geometry_uncertainty_limits.csv —oddelená geometrická diagnostika.",
       "TFNO_anchored_effect_sensitivity.csv —alternatívna definícia efektu.",
       "observed_training_and_TEST_costs.csv —pôvodné náklady všetkých50behov.",
       "independent_verification.json —overenie výpočtov a nákladov.",
       "Colas et al.:https://arxiv.org/abs/1806.08295",
       "Hoenig/Heisey:https://doi.org/10.1198/000313001300339897",
       "SciPy:https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nct.html",
       "Owen/Eckles:https://arxiv.org/abs/1106.2125"]
def readable(text):
    fixed=[]
    for line in text.splitlines():
        if "https://" not in line and not line.startswith("Dokončené "):
            line=re.sub(r"(?<=\d)(?=[^\W\d_])"," ",line)
            line=re.sub(r"%(?=[^\W\d_])","% ",line)
            line=re.sub(r"(?<=[^\W\d_])(?=\d)"," ",line)
            line=line.replace("L 2","L2").replace("H 200","H200").replace("FP 32","FP32")
            line=re.sub(r"([;:])(?=\S)",r"\1 ",line)
            line=re.sub(r",(?=[^\W\d_])",", ",line)
            line=re.sub(r"\b(všetkých|všetky|všetkým|každej|každú|spolu|iba|pri|pre|na|po|zo|v)(?=\d)",r"\1 ",line,flags=re.IGNORECASE)
        fixed.append(line)
    return "\n".join(fixed)+"\n"
(out/"VYSLEDKY.txt").write_text(readable("\n".join(lines)),encoding="utf-8")
(out/"README.md").write_text(readable("""# Koľko ďalších tréningov a GPU hodín?

[Čitateľný výsledok a odporúčanie](VYSLEDKY.txt) · [Presné odporúčanie](recommendation.json) · [Všetkých96 plánovacích scenárov](sample_size_design_summary.csv)

Odporúčaný cielený dodatok je **20 nových tréningov každej z metód TFNO, dense FNO a RealNPU**:60 nových modelov, odhad243GPU-h vrátane TEST, plánovacia rezerva260GPU-h. Je navrhnutý na90% silu pre približne10% rozdiel chyby pri troch metrikách, s opravou na9porovnaní a konzervatívnym pilotným odhadom variability.

Rovnomerné nové porovnanie všetkých10 architektúr vo všetkých dvojiciach potrebuje pri90% sile pre20% rozdiely dvoch L2 metrík **30 nových tréningov/metódu**:300modelov, odhad977GPU-h, tréningový strop a TEST rezerva1312,5GPU-h. MSE by v tomto novom protokole zostala sekundárna; pôvodná trojmetrická štúdia sa nemení.

Čísla sú podmienené plánovacími predpokladmi a pevným existujúcim benchmarkom. Nové tréningy nepridávajú geometrie a nemôžu samy odstrániť geometrickú neistotu. Výsledky nie sú univerzálnou hranicou obhájiteľnosti. **Žiadne nové tréningy sa nespustili.**

- [Nezávislá kontrola](independent_verification.json)
- [Náklady pôvodných50 tréningov a TESTov](observed_training_and_TEST_costs.csv)
- [Geometrická neistota](geometry_uncertainty_limits.csv)
- [Podrobná metodika](../../03_methodology/power_analysis_20261124/protokol_v2.txt)
- [Spustiteľný zdroj a nastavenie](../../04_source/power_analysis_20261124)
"""),encoding="utf-8")
print(json.dumps(dict(status="POWER_REPORT_COMPLETE",completed_at=finished,
    focused_new_models=60,focused_expected_GPU_h=recommended["total_expected_GPU_h"],
    focused_budget_GPU_h=recommended["training_cap_plus_TEST_reserve_GPU_h"],
    broad_new_models=300,broad_expected_GPU_h=broad["total_expected_GPU_h"],
    broad_budget_GPU_h=broad["training_cap_plus_TEST_reserve_GPU_h"])),flush=True)
