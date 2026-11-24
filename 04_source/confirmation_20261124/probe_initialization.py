"""Fingerprint initial networks and sample order without loading training targets."""
import argparse,gc,hashlib,importlib.util,json,pickle,sys
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument("--folder",type=Path,required=True);p.add_argument("--seeds",required=True);p.add_argument("--method",required=True)
a=p.parse_args();folder=a.folder;sys.path.insert(0,str(folder))
spec=importlib.util.spec_from_file_location("isolated_worker_probe",folder/"worker.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
policy,common,training,models,fit=module.dependencies()
ready=json.loads((Path(policy["fixture_root"])/"fixture/ready.json").read_text())
records=[]
for seed in json.loads(a.seeds):
    common.setup("cpu",seed)
    model=models.PackedFieldOperator(a.method,common.recipe(a.method),ready["normalization"])
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        if torch.is_tensor(tensor):
            h.update(str(tuple(tensor.shape)).encode());h.update(str(tensor.dtype).encode())
            h.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            # neuraloperator retains non-tensor extra_state alongside parameters.
            h.update(type(tensor).__name__.encode());h.update(pickle.dumps(tensor,protocol=4))
    generator=torch.Generator(device="cpu").manual_seed(seed+1)
    order=torch.randperm(640,generator=generator).numpy()
    records.append(dict(seed=seed,state_sha256=h.hexdigest(),
        first_epoch_order_sha256=hashlib.sha256(order.tobytes()).hexdigest(),
        parameter_ledger=model.parameter_ledger(),recipe=common.recipe(a.method)))
    del model;gc.collect()
print(json.dumps(dict(method=a.method,records=records,torch=torch.__version__,numpy=np.__version__,
    cuda_initialized=torch.cuda.is_initialized())),flush=True)
