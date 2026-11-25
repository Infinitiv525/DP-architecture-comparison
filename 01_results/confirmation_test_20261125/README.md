# Záverečný TEST60nových architektúr
Všetky váhy sú zmrazené. CPU overenie60checkpointov prešlo.
Štyri GPU časti kontrolujú všetkých60modelov naVAL. Až po ich úspechu sa spustí TEST:60modelov×200prípadov=12000natívnych polí.
[Registrácia úloh](submission.json) · [CPU kontrola](cpu_gate.json) · [Zmrazenie](freeze.json).
[Protokol](../../03_methodology/confirmation_test_20261125/protokol.txt) · [Zdrojový kód](../../04_source/confirmation_test_20261125).
H200,FP32,bezTF32,100zahriatí a1000meraní každého celého poľa. Polia a rawtimings zostávajú naPerune, bez sťahovania doPC.
Po dokončeníGPUTESTu je automaticky naplánovaný nezávislýCPUaudit všetkých uložených polí. Párová štatistika20nových semien nasleduje osobitne podľa protokolu24.11.
