"""Restore the pilot's original checkpoints, including neuralop's metadata envelope.

models.py from the frozen source revision must be on sys.path. This helper does
not change a checkpoint or relax strict checking of any tensor parameter.
"""
def restore_model(checkpoint):
    import copy
    from models import PointPipeline, AuthorTFNOPipeline, SCN, CascadeCorrelation
    name, config, normalization = checkpoint['model'], checkpoint['config'], checkpoint['normalization']
    if name == 'tfno':
        model = AuthorTFNOPipeline(normalization, config['tfno'])
    else:
        model = PointPipeline(name, normalization, config)
        if name == 'scn':
            model.core = SCN(model.features.inputs, checkpoint['dynamic']['scn_units'])
        if name == 'cascor':
            model.core = CascadeCorrelation(model.features.inputs, checkpoint['dynamic']['cascor_units'])
    state = copy.copy(checkpoint['state'])
    if name == 'tfno':
        # neuralop 2.0 BaseModel.state_dict inserts this dictionary into the
        # parent's destination. The parent nn.Module loader does not call the
        # child's public load_state_dict (which normally consumes the envelope).
        envelope = state.pop('_metadata')
        if envelope != model.core._init_kwargs:
            raise ValueError('TFNO constructor/version metadata differs from the saved model')
    model.load_state_dict(state, strict=True)
    return model.eval()
