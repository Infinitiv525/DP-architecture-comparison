# Plné uložené predikčné polia

[Stiahnuť polia jednotlivých modelov](https://github.com/Infinitiv525/DP-architecture-comparison/releases/tag/native-prediction-fields-20261118). Je tu všetkých 59 uložených natívnych TEST polí: 45 hlavných sietí, 5 dense FNO, 5 pomocných PINN a 4 samostatné predĺžené siete. Každý model má 200 prípadov, tri kanály u/v/p a mriežku 512×512. Celkom 11 800 polí, 37119597952 B pred kompresiou.

Nejde o časový priebeh ani nové generovanie. Sú to presne pôvodné uložené výstupy, overené proti SHA z originálnych TEST súhrnov. Každý archív obsahuje aj pôvodný výsledkový súhrn a TEST_CASES.json s poradím prípadov, Re a geometriami. Škálovanie a tlaková konvencia zostávajú ako v pôvodnom evaluátore.

Všetko sa balilo a odosielalo priamo na Perune, bez prenosu dát cez PC. Manifest obsahuje SHA originálu aj archívu. Referenčné vstupy FlowBench sú externé a zostávajú v pripnutej verzii datasetu.

| Model | Seed | Archív |
|---|---:|---|
| cascor_s931101 | 931101 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/cascor_s931101__native_TEST_fields.tar.gz) |
| cascor_s931102 | 931102 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/cascor_s931102__native_TEST_fields.tar.gz) |
| cascor_s931103 | 931103 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/cascor_s931103__native_TEST_fields.tar.gz) |
| cascor_s931104 | 931104 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/cascor_s931104__native_TEST_fields.tar.gz) |
| cascor_s931105 | 931105 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/cascor_s931105__native_TEST_fields.tar.gz) |
| chebkan_s931101 | 931101 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/chebkan_s931101__native_TEST_fields.tar.gz) |
| chebkan_s931101_extend8h | 931101 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/chebkan_s931101_extend8h__native_TEST_fields.tar.gz) |
| chebkan_s931102 | 931102 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/chebkan_s931102__native_TEST_fields.tar.gz) |
| chebkan_s931102_extend8h | 931102 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/chebkan_s931102_extend8h__native_TEST_fields.tar.gz) |
| chebkan_s931103 | 931103 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/chebkan_s931103__native_TEST_fields.tar.gz) |
| chebkan_s931104 | 931104 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/chebkan_s931104__native_TEST_fields.tar.gz) |
| chebkan_s931105 | 931105 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/chebkan_s931105__native_TEST_fields.tar.gz) |
| dendrite_s931101 | 931101 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/dendrite_s931101__native_TEST_fields.tar.gz) |
| dendrite_s931102 | 931102 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/dendrite_s931102__native_TEST_fields.tar.gz) |
| dendrite_s931103 | 931103 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/dendrite_s931103__native_TEST_fields.tar.gz) |
| dendrite_s931104 | 931104 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/dendrite_s931104__native_TEST_fields.tar.gz) |
| dendrite_s931105 | 931105 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/dendrite_s931105__native_TEST_fields.tar.gz) |
| elm_s931101 | 931101 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/elm_s931101__native_TEST_fields.tar.gz) |
| elm_s931102 | 931102 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/elm_s931102__native_TEST_fields.tar.gz) |
| elm_s931103 | 931103 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/elm_s931103__native_TEST_fields.tar.gz) |
| elm_s931104 | 931104 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/elm_s931104__native_TEST_fields.tar.gz) |
| elm_s931105 | 931105 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/elm_s931105__native_TEST_fields.tar.gz) |
| fixed_k00 | 150001 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/fixed_k00__native_TEST_fields.tar.gz) |
| fixed_k01 | 150002 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/fixed_k01__native_TEST_fields.tar.gz) |
| fixed_k02 | 150003 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/fixed_k02__native_TEST_fields.tar.gz) |
| fixed_k03 | 150004 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/fixed_k03__native_TEST_fields.tar.gz) |
| fixed_k04 | 150005 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/fixed_k04__native_TEST_fields.tar.gz) |
| fno_s931101 | 931101 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/fno_s931101__native_TEST_fields.tar.gz) |
| fno_s931102 | 931102 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/fno_s931102__native_TEST_fields.tar.gz) |
| fno_s931103 | 931103 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/fno_s931103__native_TEST_fields.tar.gz) |
| fno_s931104 | 931104 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/fno_s931104__native_TEST_fields.tar.gz) |
| fno_s931105 | 931105 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/fno_s931105__native_TEST_fields.tar.gz) |
| pinet_s931101 | 931101 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/pinet_s931101__native_TEST_fields.tar.gz) |
| pinet_s931102 | 931102 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/pinet_s931102__native_TEST_fields.tar.gz) |
| pinet_s931103 | 931103 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/pinet_s931103__native_TEST_fields.tar.gz) |
| pinet_s931104 | 931104 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/pinet_s931104__native_TEST_fields.tar.gz) |
| pinet_s931105 | 931105 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/pinet_s931105__native_TEST_fields.tar.gz) |
| rbf_s931101 | 931101 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/rbf_s931101__native_TEST_fields.tar.gz) |
| rbf_s931101_extend8h | 931101 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/rbf_s931101_extend8h__native_TEST_fields.tar.gz) |
| rbf_s931102 | 931102 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/rbf_s931102__native_TEST_fields.tar.gz) |
| rbf_s931102_extend8h | 931102 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/rbf_s931102_extend8h__native_TEST_fields.tar.gz) |
| rbf_s931103 | 931103 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/rbf_s931103__native_TEST_fields.tar.gz) |
| rbf_s931104 | 931104 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/rbf_s931104__native_TEST_fields.tar.gz) |
| rbf_s931105 | 931105 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/rbf_s931105__native_TEST_fields.tar.gz) |
| realnpu_s931101 | 931101 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/realnpu_s931101__native_TEST_fields.tar.gz) |
| realnpu_s931102 | 931102 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/realnpu_s931102__native_TEST_fields.tar.gz) |
| realnpu_s931103 | 931103 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/realnpu_s931103__native_TEST_fields.tar.gz) |
| realnpu_s931104 | 931104 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/realnpu_s931104__native_TEST_fields.tar.gz) |
| realnpu_s931105 | 931105 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/realnpu_s931105__native_TEST_fields.tar.gz) |
| scn_s931101 | 931101 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/scn_s931101__native_TEST_fields.tar.gz) |
| scn_s931102 | 931102 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/scn_s931102__native_TEST_fields.tar.gz) |
| scn_s931103 | 931103 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/scn_s931103__native_TEST_fields.tar.gz) |
| scn_s931104 | 931104 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/scn_s931104__native_TEST_fields.tar.gz) |
| scn_s931105 | 931105 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/scn_s931105__native_TEST_fields.tar.gz) |
| tfno_s931101 | 931101 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/tfno_s931101__native_TEST_fields.tar.gz) |
| tfno_s931102 | 931102 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/tfno_s931102__native_TEST_fields.tar.gz) |
| tfno_s931103 | 931103 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/tfno_s931103__native_TEST_fields.tar.gz) |
| tfno_s931104 | 931104 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/tfno_s931104__native_TEST_fields.tar.gz) |
| tfno_s931105 | 931105 | [Stiahnuť](https://github.com/Infinitiv525/DP-architecture-comparison/releases/download/native-prediction-fields-20261118/tfno_s931105__native_TEST_fields.tar.gz) |
