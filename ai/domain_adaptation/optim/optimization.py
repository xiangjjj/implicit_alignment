from torch import optim
from ai.domain_adaptation.utils.config import parse_yaml_to_dict

class INVScheduler(object):
    def __init__(self, gamma, decay_rate, init_lr=0.001):
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.init_lr = init_lr

    def next_optimizer(self, optimizer, num_iter):
        lr = self.init_lr * (1 + self.gamma * num_iter) ** (-self.decay_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * param_group['lr_scale']
        return optimizer


def get_optimizer(optimizer_name, param_groups, opt_params):
    optimization_type = getattr(optim, optimizer_name, None)
    if optimization_type is None:
        raise ValueError(f'wrong optimizer {optimizer_name}')

    optimizer = optimization_type(param_groups, **opt_params)
    return optimizer


def get_optimizer_from_yaml(yaml_file, param_groups):
    opt_cfg = parse_yaml_to_dict(yaml_file)
    optimizer = get_optimizer(opt_cfg.optim.type, param_groups, opt_cfg.optim.params)
    optimizer.zero_grad()
    return optimizer, opt_cfg


def get_schedular_from_yaml(yaml_file, opt_cfg):
    cfg = parse_yaml_to_dict(yaml_file)
    assert cfg.lr_scheduler.type == 'INVScheduler', 'Scheduler type not supported!'
    lr_scheduler = INVScheduler(gamma=cfg.lr_scheduler.gamma,
                                decay_rate=cfg.lr_scheduler.decay_rate,
                                init_lr=opt_cfg.optim.params.lr)
    return lr_scheduler
