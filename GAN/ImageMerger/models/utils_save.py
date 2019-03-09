import torch


def save_model(model, path, optimizer=None, params={}, verbose=True):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': None if optimizer is None else optimizer.state_dict(),
        **params
    }, path)
    if verbose:
        print('saved checkpoint to path: {}' .format(path))


def load_model(model, path, optimizer=None, verbose=True):
    if verbose:
        print('loading checkpoint from path: {}'.format(path))
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    params = checkpoint.copy()
    params.pop('model_state_dict')
    if params.__contains__('optimizer_state_dict'):
        params.pop('optimizer_state_dict')
    return params
