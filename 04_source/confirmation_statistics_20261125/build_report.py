"""Readable interpretation; preserve primary versus geometry-sensitivity distinction."""
import json,hashlib
from pathlib import Path
R=Path('/mnt/project/perun2601449/dp_llm_cfd/results/architecture_confirmation_statistics_20261125_v1')
def read(n):return json.loads((R/n).read_text())
assert read('independent_verification.json')['status']=='INDEPENDENT_INPUT_PRIMARY_EXACT_SIGN_AND_BOOTSTRAP_AUDIT_PASS'
s=read('summary.json');p=read('paired_t_primary.json');sens=read('crossed_geometry_sensitivity.json');exact=read('exact_sign_sensitivities.json');methods=read('method_summary.json')
labels={'tfno':'TFNO','fno':'dense FNO','realnpu':'RealNPU'}
ml={'all_cell_mse':'MSE u/v/p','fluid_velocity_relative_l2':'relatívna L2 rýchlosti','fluid_pressure_relative_l2':'relatívna L2 tlaku'}
lines=['POTVRDZUJÚCE POROVNANIE TROCH ARCHITEKTÚR – 25.11.2026','',
'Výsledok: na tomto pevnom TEST benchmarku je TFNO presnejšie než denseFNO ajRealNPU vo všetkých troch metrikách. Výsledok platí pre naše konkrétne implementácie, konfigurácie a rovnaký pridelený tréningový rozpočet; nie je univerzálnym poradím všetkých možných sietí.',
'Všetkých60nových tréningov a TESTov je dokončených. Nezávislý CPUaudit znova skontroloval12000celých polí a12000000dvojíc časovacích meraní. Následný nezávislý štatistický audit prešiel.','',
'ČO SA POROVNÁVALO',
'TFNO20,denseFNO20,RealNPU20; vždy4H200GPU-h/model,640TRAIN/160VAL/200TEST,rovnaká celá geometria a rovnaký natívny výstup512×512u/v/p. Počet parametrov ostal odlišný podľa pôvodných implementácií. Nové konfigurácie sa neladili, výber checkpointov bol podľaVAL.',
'Primárna jednotka je tréningové semeno: každý model najprv spriemeruje200pevných TEST prípadov. Používame20nových párovaných semien, nie4000nezávislých prípadov/metódu. Pôvodných5pilotných behov sa nepridáva.',
'Pôvodné porovnanie všetkých architektúr s5behmi ostáva samostatné. Tento cielený dodatok overuje iba tieto3rodiny.','',
'PRIEMERNÉ VÝSLEDKY',
'Metóda | MSE | relatívna L2 rýchlosti[%] | relatívna L2 tlaku[%] | najhoršie per-case GPU p99[ms] | modely spĺňajúce p99<=5ms']
for m in methods:
 lines.append('%s | %.12g | %.6f | %.6f | %.6f | %d/20'%(labels[m['method']],m['all_cell_mse'],100*m['fluid_velocity_relative_l2'],100*m['fluid_pressure_relative_l2'],m['worst_case_GPU_p99_ms'],m['GPU_5ms_pass_models']))
lines+=['','PRIMÁRNE ŠTATISTICKÉ POROVNANIE',
'Dvojstranný párový Studentt cez20semien,df19; Holm cez9vopred určených kontrastov,rodinná alpha0,05. Osem z9rozdielov je významných. Bonferroni t-intervaly s nominálnym95%rodinným pokrytím vylučujú nulu pri rovnakých8kontrastoch.',
'A−B: záporné číslo znamená nižšiu chybuA. Percento je bodový relatívny rozdiel voči priemeruB. Interval je pre absolútny priemerný rozdiel, nie pre percentuálny pomer.',
'Porovnanie | metrika | rozdielA−B | rozdiel vočiB[%] | Holm p | Bonferroni rodinný95% interval | páry s nižšou chybouA/B']
for x in p:
 lines.append('%s–%s | %s | %.12g | %+.6f | %.12g | [%.12g, %.12g] | %d/%d'%(
  labels[x['method_a']],labels[x['method_b']],ml[x['metric']],x['mean_difference_a_minus_b'],x['relative_difference_vs_b_percent'],x['p_Holm_9'],
  *x['Bonferroni_family95_t_CI'],x['A_lower_error_seed_pairs'],x['B_lower_error_seed_pairs']))
lines+=['',
'TFNO oproti denseFNO: nižšia MSE o20,94%,relatívna chyba rýchlosti o5,81% a tlaku o5,52%. Všetky tri rozdiely prešli primárnou Holm korekciou.',
'DenseFNO oproti RealNPU: významne nižšia relatívna chyba rýchlosti ajtlaku. V celkovej MSE je rozdiel približne+1,97%v neprospechFNO, ale Holm p=0,316; rozdiel sa nepreukázal. To nie je dôkaz rovnosti ani ekvivalencie. MSE súhrnne váži absolútne chybyu/v/p, relatívneL2 hodnotí rýchlosť a tlak osobitne, preto nejde o rozpor.','',
'KONTROLY CITLIVOSTI A PREDPOKLADY',
'Exaktné znamienkové vymenovanie všetkých1048576kombinácií potvrdilo8z9rozdielov po vlastnej Holm korekcii. Nezávislý meet-in-the-middle algoritmus dal rovnaké počty. Predpokladom sign-flipu je symetria rozdielov pri nulovej hypotéze.',
'Binomický sign test tiež označil8z9kontrastov. Ten skúma smer/medián rozdielu a nie veľkosť priemerného efektu; neprezentuje sa ako totožná otázka.',
'Žiadne semeno sa nevynechalo. Všetky surové20rozdiely, Shapiro diagnostika, šikmosť,kurtóza a QQbody sú v paired_t_primary.json. Žiadny primárny test sa dodatočne nemenil podľa p.']
ps=[x['distribution_diagnostic']['shapiro_p'] for x in p]
lines.append('Shapiro p v9diagnostikách: rozsah%.6g až%.6g; žiadne pod0,05. Taký výsledok normalitu nedokazuje, najmä pri20pozorovaniach.'%(min(ps),max(ps)))
lines+=['','DÔLEŽITÉ OBMEDZENIE: GEOMETRIE',
'Primárny výsledok sa týka opakovateľnosti na rovnakom pevnom200-prípadovom benchmarku. Nové tréningové semená nepridali nové geometrie;85geometrických skupín sa vyskytuje ajvTRAIN.',
'Pri samostatnom kríženom resamplovaní20semien a85geometrií (200000opakovaní) sa zvyšuje neistota. BasicBonferroni intervaly vylúčili nulu iba pri4z9kontrastov: L2rýchlosti a tlakuTFNO–RealNPU aFNO–RealNPU.',
'Pri TFNO–FNO basicBonferroni intervaly všetkých3metrík nulu obsahujú, zatiaľ čo percentileBonferroni intervaly ju vylučujú; centrované bootstrapHolm p sú0,0908preMSE a0,0306preobeL2. Výsledok tejto približnej citlivosti teda závisí od postupu. Nevyberáme dodatočne priaznivejší interval.',
'Bezpečný záver: výhodaTFNO nadFNO je presvedčivá cez nové tréningové semená na tomto benchmarku. Všeobecnú výhodu na nevidených geometriách tento experiment nepreukazuje.',
'Všetky varianty a intervaly sú v crossed_geometry_sensitivity.json; táto citlivosť nenahrádza dopredu určený primárny test.','',
'RÝCHLOSŤ A JEDNOTLIVÉ PREKROČENIA',
'Všetkých60modelov splnilo GPUajhost per-casep99<=5ms na všetkých200prípadoch. Nie je to záruka, že každé jednotlivé volanie bolo pod5ms.',
'Metóda | jednotlivé CUDAvolania nad5ms zo4000000 | maximálne CUDAmeranie[ms] | najhoršie hostp99[ms]']
for m in methods:lines.append('%s | %d | %.6f | %.6f'%(labels[m['method']],m['individual_GPU_queries_above5ms'],m['maximum_individual_GPU_query_ms'],m['worst_case_host_p99_ms']))
lines+=['RealNPU bolo opisne najrýchlejšie. TFNO poskytlo najnižšie chyby pri splnení nášho p99limitu; denseFNO má menšiu latenciu nežTFNO a lepšieL2chyby nežRealNPU.',
'Meranie:H200,FP32,bezTF32,batch1,celé pole,vstupy naGPU,100zahriatí+1000meraní na prípad. Prenos naGPU,kreslenie a sieťová komunikácia nie sú zahrnuté. CUDA a synchronizovaný meranýhostčas sú samostatné,réžia sa neodpočítava. Nejde o nový test výkonu starej5-seedovej kohorty za rovnakých dátumových podmienok.','',
'ZDROJE A OVERENIE']
cost=read('resource_closeouts.json')
lines.append('60hlavných tréningov:235,0825GPU-h. Samotných12TESTskupín:%.6fGPU-h. Záznamy vybraných súvisiacich kontrol sú osobitne v resource_closeouts.json; nejde o konečný účet portálu.'%sum(x['allocated_GPU_h'] for x in cost['test_v2']['records'] if x['role']=='test'))
lines+=['Štatistika ani jej nezávislé overenie nepoužiliGPU a nespustili nové učenie.',
'Nezávislé overenie:12000vstupných riadkov,180modelových priemerov(60×3),9t-testov cez neúplnúbetafunkciu,Holm a intervaly; všetky exaktné počty znamienok a intervaly zo zachovaných bootstrapreplikátov.',
'Plánz24.11.bol zverejnený pred novými tréningmi, no pôvodnýTEST bol už známy pri výbere tohto dodatku. Skutočná chronológia sa zachováva.',
'Výsledky, váhy a polia sa archivujú priamoPerun→GitHub. Podrobnosti verejného archívu a jeho overenia sú v ARCHIV.txt a publication_receipt.json po dokončení publikácie. Žiadne objemné dáta nevznikli vPC.','',
'REPRODUKČNÉ SÚBORY',
'paired_t_primary.json:9primárnych kontrastov,intervaly,20rozdielov a diagnostiky.',
'exact_sign_sensitivities.json:exaktné znamienkové ajbinomické kontroly.',
'crossed_geometry_sensitivity.json:všetky bootstrapintervaly a odlišný rozsah inferencie.',
'analysis_inputs.npz:kompaktné vstupné metriky,IDprípadov,semien a geometrií.',
'crossed_bootstrap_replicates.npz:všetkých200000resamplingvýsledkov(v úplnomarchíve).',
'input_manifest.json a independent_verification.json:SHA vstupov a nezávislý audit.',
'per_seed_model_means.csv a primary_t_tests.csv:tabuľky na ďalšiu prácu.',
'latency_by_model.json:časovania,počty prekročení a maximá každého modelu.','',
'METODICKÉ ZDROJE',
'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html',
'https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html',
'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html',
'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html']
(R/'VYSLEDKY.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
readme='''# Potvrdenie troch architektúr na20nových tréningoch
[Čitateľný výsledok](VYSLEDKY.txt) · [Všetky primárne testy](paired_t_primary.json) · [Nezávislá kontrola](independent_verification.json).
TFNO má na pevnom200-prípadovom benchmarku nižšie chyby než denseFNO ajRealNPU vo všetkých3metrikách. Osem z9vopred určených párových kontrastov prešloHolm korekciou; celkováMSE denseFNO–RealNPU sa nepreukázala.
20nových semien/metódu; pôvodných5sa nemieša. [Priemery jednotlivých modelov](per_seed_model_means.csv).
**Obmedzenie:**opakovateľnosť cez tréningové semená na známom benchmarku. Pri približnom geometrickom bootstrape sú záveryTFNO–FNO citlivé na druh intervalu; všeobecnú prevahu na nevidených geometriách netvrdíme.
Všetkých60modelov splnilo per-casep99<=5ms naH200. Jednotlivé prekročenia sú priznané v reporte.
[Protokol](../../03_methodology/confirmation_statistics_20261125/protokol.txt) · [Zdroj](../../04_source/confirmation_statistics_20261125).
Úplné váhy, polia, numerické replikáty a zdroje:odkazy pribudnú vARCHIV.txt po overení verejného archívu.
'''
(R/'README.md').write_text(readme,encoding='utf-8')
completion=dict(status='STATISTICS_COMPLETE_INDEPENDENT_AUDIT_PASS_PUBLICATION_IN_PROGRESS',
 primary_contrasts=9,Holm_rejections=8,training_models=60,native_fields_audited=12000,
 old_pilot_models_included=False,GPU_used_for_statistics=False,full_archive_publication_pending=True)
(R/'completion_status.json').write_text(json.dumps(completion,indent=2)+'\n')
(R/'OUTPUT_MANIFEST.json').write_text(json.dumps({p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 for p in R.iterdir() if p.is_file() and p.name!='OUTPUT_MANIFEST.json'},indent=2)+'\n')
print('READABLE_REPORT_AND_COMPLETION_STATUS_READY')
