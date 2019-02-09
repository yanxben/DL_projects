import torch


def save_model(model, path, optimizer=None, params={}):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': None if optimizer is None else optimizer.state_dict(),
        **params
    }, path)
    print('saved checkpoint to path: {}' .format(path))


def load_model(model, path, optimizer=None):
    print('loading checkpoint from path: {}'.format(path))
    checkpoint = torch.load(path, map_location='cpu' if not torch.cuda.is_available() else None)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    params = checkpoint.copy()
    params.pop('model_state_dict')
    if params.__contains__('optimizer_state_dict'):
        params.pop('optimizer_state_dict')
    return params
