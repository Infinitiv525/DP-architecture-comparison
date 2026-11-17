"""Separate gradient truncation from optimizer-update cadence."""
import torch


class RolloutFailure(RuntimeError):
    def __init__(self,record):
        super().__init__(record['reason']);self.record=record


def matched_window(previous,current,step_function,loss_function,count,gradient_horizon):
    """Same <=32 forward steps and 8-step objectives; detach differs ONLY.

    Caller does exactly one backward/optimizer step after this function.
    Returned states are own final predictions detached for the next update.
    loss_function receives a block offset, its prefix, and own predictions.
    No reference reader or future-state-reset callback is accepted.
    """
    if count<1 or count>32 or gradient_horizon not in [8,32]:raise ValueError('Invalid matched-window lengths')
    terms=[]
    for offset in range(0,count,8):
        first_previous,first_current=previous,current;predictions=[]
        for _ in range(min(8,count-offset)):
            predicted=step_function(previous,current)
            previous,current=current,predicted;predictions.append(predicted)
        terms.append(loss_function(offset,first_previous,first_current,predictions)*(len(predictions)/count))
        if gradient_horizon==8:previous,current=previous.detach(),current.detach()
    return previous.detach(),current.detach(),torch.stack(terms).sum()
