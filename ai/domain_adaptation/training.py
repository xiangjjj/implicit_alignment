import tqdm
from torch.autograd import Variable

from ai.domain_adaptation.evaluator import evaluate
from ai.domain_adaptation.optim import optimization
from ai.domain_adaptation.datasets.data_provider import DataLoaderManager, get_dataloader_from_image_filepath
from ai.domain_adaptation.trainer import base_train
from ai.domain_adaptation.utils import logger


def load_data_and_train(model_instance, args):
    data_loader_manager = DataLoaderManager(args)

    param_groups = model_instance.get_parameter_list()

    base_optimizer, opt_cfg = optimization.get_optimizer_from_yaml(args.optimizer_config, param_groups)
    base_scheduler = optimization.get_schedular_from_yaml(args.scheduler_config, opt_cfg)

    model_summary = logger.get_model_summary_str(args)
    tensorboard = logger.create_tensorboard(args, model_summary)
    train(model_instance, data_loader_manager, max_iter=args.train_steps,
          base_optimizer=base_optimizer,
          lr_scheduler=base_scheduler,
          tensorboard=tensorboard,
          save=not args.not_save, args=args)


def load_data_and_test(model_instance, args):
    test_loader = get_dataloader_from_image_filepath(args.test_file, args, batch_size=32, is_train=False)
    eval_stats, prob_hat = evaluate.evaluate_from_dataloader(model_instance, test_loader[0])
    print(eval_stats)


def train(model_instance, data_loader_manager,
          max_iter, base_optimizer, lr_scheduler, tensorboard, save=True, args=None):
    model_instance.set_train(True)
    train_stats = {}
    iter_num = 0
    total_progress_bar = tqdm.tqdm(desc='Train iter', total=args.train_steps, ncols=100)
    for i_th_iter in range(args.train_steps):
        if args.self_train is True and i_th_iter % args.yhat_update_freq == 0:
            data_loader_manager.update_self_training_labels(model_instance)

        source_loader, target_loader = data_loader_manager.get_train_source_target_loader()
        _, inputs_source, labels_source = next(iter(source_loader))
        _, inputs_target, labels_target = next(iter(target_loader))

        base_optimizer = lr_scheduler.next_optimizer(base_optimizer, iter_num / 5)
        base_optimizer.zero_grad()

        inputs_source, inputs_target, labels_source = Variable(inputs_source).cuda(), Variable(
            inputs_target).cuda(), Variable(labels_source).cuda()

        # evaluate
        if iter_num % args.eval_interval == 0 or iter_num == args.train_steps - 1:
            eval_result, prob_hat = evaluate.evaluate_from_dataloader(model_instance,
                                                                      data_loader_manager.get_test_target_loader())
            total_progress_bar.set_description(
                desc=f'Acc {eval_result["accuracy"]:.4f}, cls avg acc {eval_result["test_balanced_acc"]:.4f}')
            train_stats.update(eval_result)

            if save is True:
                model_instance.c_net.save()

            if 'train_stats' in locals():
                for k in train_stats:
                    tensorboard.add_scalar(k, train_stats[k], iter_num)

        train_stats = base_train.train_batch(model_instance, inputs_source, labels_source, inputs_target,
                                   base_optimizer, args)
        train_stats['target_unique_label'] = labels_target.unique().shape[0]
        iter_num += 1
        total_progress_bar.update(1)
