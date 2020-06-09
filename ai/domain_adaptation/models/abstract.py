import os
import torch


class AbstractModel:
    def __init__(self, name):
        super(AbstractModel, self).__init__()
        self.name = name
        self.save_path = os.path.join('../saved_models', f'{self.name}.pt')

    def save(self):
        try:
            torch.save({
                'model_state_dict': self.state_dict()},
                self.save_path)
        except OSError as e:
            print(f'OSError encountered.{e}')

    def load_if_exists(self):
        if os.path.exists(self.save_path):
            checkpoint = torch.load(self.save_path)
            self.load_state_dict(checkpoint['model_state_dict'])
            print(f'model loaded from {self.save_path}')

    def load_from_path_if_exists(self, path):
        path = self.resolve_checkpoint_surfix(path)
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print(f'model loaded from {path}')
        elif 'none' not in path:
            raise ValueError(f'checkpoint not found in {path}')

    def resolve_checkpoint_surfix(self, path):
        if path.endswith('.pt') is not True:
            return path + '.pt'
        else:
            return path

    def filter_checkpoint_dict_by_keyword(self, checkpoint, keyword='classifier'):
        checkpoint['model_state_dict'] = {k: checkpoint['model_state_dict'][k]
                                          for k in checkpoint['model_state_dict'] if keyword not in k}
        return checkpoint

    def freeze(self, layer):
        for p in layer.parameters():
            p.requires_grad = False

    def unfreeze(self, layer):
        for p in layer.parameters():
            p.requires_grad = True
