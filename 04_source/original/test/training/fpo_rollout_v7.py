"""Own-state rollout with truncated gradients, never reference-state resets."""


def own_chunks(previous,current,step_function,horizon,chunk_length):
    if horizon<1 or chunk_length<1:raise ValueError('Positive horizon and chunk length required')
    for start in range(0,horizon,chunk_length):
        first_previous,first_current=previous,current;predictions=[]
        for _ in range(min(chunk_length,horizon-start)):
            predicted=step_function(previous,current)
            previous,current=current,predicted;predictions.append(predicted)
        yield start,first_previous,first_current,predictions
        # The caller may backpropagate and update weights at yield. Carry own
        # numerical states onward, but discard the previous block's graph.
        previous,current=previous.detach(),current.detach()
