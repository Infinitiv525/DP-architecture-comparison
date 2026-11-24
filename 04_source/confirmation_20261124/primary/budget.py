"""Budget clock counts allocated GPU wall time, including worker startup."""
import os
import re
import subprocess
import time


def seconds(value):
    days=0
    if '-' in value:
        d,value=value.split('-',1);days=int(d)
    fields=[int(x) for x in value.split(':')]
    if len(fields) not in [2,3]:raise ValueError('Unexpected Slurm duration '+value)
    return days*86400+sum(x*60**i for i,x in enumerate(reversed(fields)))


class Budget:
    def __init__(self,limit_seconds,initial_elapsed_seconds,clock=time.monotonic):
        self.limit=float(limit_seconds);self.initial=float(initial_elapsed_seconds)
        if not 0<=self.initial<self.limit:raise ValueError('No allocation budget remaining')
        self.clock=clock;self.start=clock()
    @property
    def used(self):return self.initial+self.clock()-self.start
    @property
    def remaining(self):return max(0.,self.limit-self.used)
    def can_start(self,reserve_seconds,estimated_work_seconds=0.):
        return self.remaining>reserve_seconds+estimated_work_seconds


def allocation_clock(limit_seconds):
    job=os.environ.get('SLURM_JOB_ID')
    if not job:raise RuntimeError('Primary timing requires a real Slurm allocation')
    output=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
    runtime=re.search(r'\bRunTime=(\S+)',output)
    if not runtime:raise RuntimeError('Cannot read current allocated runtime')
    return Budget(limit_seconds,seconds(runtime.group(1))),dict(job_id=job,initial_scontrol=output)


def self_check():
    assert seconds('01:02')==62 and seconds('01:02:03')==3723
    assert seconds('2-03:04:05')==183845
    now=[10.];b=Budget(60,5,clock=lambda:now[0])
    assert b.used==5 and b.remaining==55 and b.can_start(10,44)
    now[0]+=40
    assert b.used==45 and b.remaining==15 and not b.can_start(10,5)
    now[0]+=30
    assert b.remaining==0 and not b.can_start(0)
    return 'BUDGET_CLOCK_CHECK_COMPLETE'


if __name__=='__main__':print(self_check())
