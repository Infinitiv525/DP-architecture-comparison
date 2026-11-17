"""Post-selection wall errors and repeated full-field speed, no TEST inputs."""
import numpy as np
import torch
from ldc_main_common_v32 import save
from ldc_store_v26 import query, conditioning
from audit_ldc_quality_v31 import boundary_audit


def selected_diagnostics(model, store, out, phase):
    out.mkdir(parents=True, exist_ok=False)
    model.eval()
    rows=[]
    speed_ids=store.validation_ids if phase=='main' else store.validation_ids[:2]
    wall_protocol=dict(boundary=dict(outer_points_per_side=256 if phase=='main' else 64, corner_exclusion=1/512))
    for index in store.validation_ids:
        known=store.known(index);fluid=np.asarray(store.fluid_np[index])
        reference=np.moveaxis(store.validation_reference(index),0,2)
        timings=[]
        with torch.inference_mode():
            if index in speed_ids:
                for _ in range(30):query(model,known,store.qgrid)
                torch.cuda.synchronize()
                for _ in range(200):
                    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                    a.record();value=query(model,known,store.qgrid);b.record();b.synchronize()
                    timings.append(a.elapsed_time(b))
            else:
                value=query(model,known,store.qgrid)
            prediction=value.cpu().numpy()
        with torch.no_grad():context=conditioning(model,known)
        bc=boundary_audit(model,context,fluid,prediction,reference,wall_protocol)
        row=dict(sample=index,boundary=bc)
        if timings:
            row['timing']=dict(p50=float(np.percentile(timings,50)),p95=float(np.percentile(timings,95)),
                               p99=float(np.percentile(timings,99)),raw_ms=timings)
        rows.append(row)
    speed_rows=[dict(sample=r['sample'],**{k:v for k,v in r['timing'].items() if k!='raw_ms'})
                for r in rows if 'timing' in r]
    speed=dict(cases=speed_rows,pass_5ms=all(r['p99']<=5 for r in speed_rows),fullfield=[512,512],
        warmup=30,repeats=200,inputs_gpu=True,
        includes='Native geometry pooling,encoder,MLP,spatial interpolation,curl,gauge,mask; no input-dependent cache',
        excludes='Host transfer,rendering,input-independent stencil setup')
    sides=['bottom','top','left','right','obstacle']
    means={side:float(np.mean([r['boundary'][side]['velocity_error_rms'] for r in rows])) for side in sides}
    result=dict(speed=speed,boundary_by_side=means,boundary_mean=float(np.mean(list(means.values()))),
                cases=rows,main_test_opened=False)
    save(out/'summary.json',result)
    model.train()
    return result
