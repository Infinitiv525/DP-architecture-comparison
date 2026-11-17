"""Pure integer layout check, without tensors, model calls or simulation data."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def check(channels,size=512,factor=8):
    packed=size//factor;count=channels*size*size;destinations=set()
    for channel in range(channels):
        for y in range(size):
            for x in range(size):
                c=channel*factor*factor+(y%factor)*factor+x%factor;py,px=y//factor,x//factor
                position=(c*packed+py)*packed+px
                assert 0<=position<count
                original_channel=c//(factor*factor);dy=(c%(factor*factor))//factor;dx=c%factor
                assert (original_channel,py*factor+dy,px*factor+dx)==(channel,y,x)
                destinations.add(position)
    assert len(destinations)==count
    return dict(native_shape=[channels,size,size],packed_shape=[channels*factor*factor,packed,packed],
        values=count,unique_destinations=len(destinations),mapping_is_bijective=True)


def main():
    result=dict(status='INTEGER_LAYOUT_BIJECTION_VERIFIED',geometry=check(2),target=check(3),
        model_forward_calls=0,training_updates=0,simulation_data_read=False,GPU_used=False,
        note='Verification of the layout definition only. Runtime tensors/training/latency remain unverified.')
    (HERE/'layout_arithmetic.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(result['status'])


if __name__=='__main__':main()
