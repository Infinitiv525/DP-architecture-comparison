# Kde sú dáta a čo je archivované

Zachovaných je 45 hlavných modelov, 5 dodatočne porovnateľne trénovaných dense FNO, 5 pomocných PINN a 4 modely s rozšíreným tréningom mimo hlavného porovnania. V osobitnom vydaní sú všetky existujúce finálne natívne TEST polia týchto 59 modelov: každý má predictions.npy tvaru [200, 512, 512, 3], osi prípad/y/x/pole, kanály u,v,p. Ide o 200 rôznych ustálených testovacích prípadov na model (spolu 11 800 polí), nie o časovú sekvenciu. Váhy ani polia sa nanovo nepočítali.

V tomto repozitári: **587 checkpointových súborov** a **59 vybraných modelov**. Checkpointy zahŕňajú aj priebežné a posledné stavy; ich počet nie je počet nezávislých finálnych sietí.

| Balík | Pôvodných súborov | Archív na GitHube |
|---|---:|---:|
| [Modely a základné výsledky](https://github.com/Infinitiv525/DP-architecture-comparison/releases/tag/research-artifacts-v1) | 2444 | 4.901 GB |
| [Doplnené vedecké záznamy a merania](https://github.com/Infinitiv525/DP-architecture-comparison/releases/tag/scientific-data-supplement-20261118) | 288 | 0.001 GB |
| [Historické numerické výstupy](https://github.com/Infinitiv525/DP-architecture-comparison/releases/tag/historical-numerical-outputs-20261118) | 350 | 2.377 GB |
| [Ďalšie pôvodné diagnostiky a zdroje](https://github.com/Infinitiv525/DP-architecture-comparison/releases/tag/research-records-completion-20261118) | 661 | 0.018 GB |
| [Pôvodné stdout a stderr úloh Slurm](https://github.com/Infinitiv525/DP-architecture-comparison/releases/tag/scheduler-logs-20261118) | 484 | 0.001 GB |
| [Úplné finálne predikčné polia 59 modelov](https://github.com/Infinitiv525/DP-architecture-comparison/releases/tag/native-prediction-fields-20261118) | 59 | 30.790 GB |

Veľké váhy a numerické polia sú **v Releases**, aby sa pri práci so zdrojovým kódom nemuseli automaticky sťahovať. [Zoznam všetkých priamych odkazov a SHA256](PUBLIC_DOWNLOADS.json). Archívy sú členené podľa modelov alebo pôvodných pokusov. Každý balík má manifest s pôvodnými cestami; historické a neúspešné pokusy zostávajú oddelené od finálneho porovnania. V starších nemenných manifestoch sa pôvodné vylúčenia vzťahujú iba na daný balík; dodatočné vydania ich rozširujú podľa tejto tabuľky.

## Ako sa zorientovať

- [Výsledky a tabuľky](.)
- [Vybrané modely a odkazy na váhy](../02_models/README.md)
- [Metodika](../03_methodology/)
- [Zdrojový kód](../04_source/)
- [História pokusov](../05_history/)

## Overenie a rozsah

Každý uložený súbor má SHA256. Overené bolo rozbalenie archívov, serverový SHA256 a úplné spätné prečítanie archívov na Perune. Aktuálne priame odkazy boli samostatne overené bez prihlásenia. Počas dopĺňania sa dáta prenášali priamo Perun → GitHub, bez modelov alebo archívov cez PC, bez nového tréningu a bez nového počítania polí.

Toto je archív vlastných výskumných artefaktov, nie úplná kópia súborového systému Perunu. Neduplikujú sa verejné vstupné FlowBench dáta, obnoviteľné tréningové a vstupné cache, stiahnuté základné LLM váhy ani softvérové prostredia. Nezverejňujú sa prístupové údaje, autentifikačné pomocné súbory ani rozpracovaná dizertácia. Staré surové paralelné/reštartovacie CFD adresáre nie sú úplne zrkadlené; spracované referenčné exporty a diagnostiky sú zahrnuté. Pri čítaní starej reštartovacej metadátovej vetvy sa pozorovalo čakanie na NFS; túto výnimku audit uvádza otvorene. Pôvodné výskumné dáta na Perune neboli zmazané.

Pôvodný [FlowBench](https://huggingface.co/datasets/BGLab/FlowBench) a ďalšie cudzie podklady si ponechávajú svoje autorstvo a licencie; archív im nepriraďuje novú licenciu.

## Súvisiace experimenty

- [Staršie 3D pokusy](https://github.com/Infinitiv525/DP-3D-experiments)
- [Porovnanie LLM](https://github.com/Infinitiv525/DP-LLM-comparison)
- [Porovnanie architektúr](https://github.com/Infinitiv525/DP-architecture-comparison)
