import torch


def train_batch(model_instance, inputs_source, labels_source, inputs_target, optimizer, args, step_num=0):
    inputs = torch.cat((inputs_source, inputs_target), dim=0)
    total_loss, train_stats = model_instance.get_loss(inputs, labels_source, step_num)
    total_loss[args.train_loss].backward()
    optimizer.step()
    return train_stats
