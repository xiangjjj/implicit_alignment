from tensorboardX import SummaryWriter
from datetime import datetime


def create_tensorboard(args, model_summary):
    tensorboard = SummaryWriter(f'{args.tensorboard_dir}/{args.group_name}/{model_summary}')
    tb_log_all_config(tensorboard, args)
    return tensorboard


def get_model_summary_str(args):
    model_summary = f'{abbreviate_domain_name(args.src_address)}2{abbreviate_domain_name(args.tgt_address)}'
    if args.n_way is not None:
        model_summary += f'.{args.n_way}way'
    if args.k_shot is not None:
        model_summary += f'.{args.k_shot}shot'
    if args.freeze_backbone is True:
        model_summary += '.freezebackbone'
    if args.mask_classifier is True:
        model_summary += '.maskC'
    if args.mask_divergence is True:
        model_summary += '.maskD'
    model_summary += f".{datetime.today().strftime('%b.%d')}"
    model_summary += f'.{args.name}'
    model_summary += f'.seed{args.seed}'
    model_summary += f'.{args.machine[:15]}'
    print(f'model summary: {model_summary}')

    return model_summary


def abbreviate_domain_name(domain_name):
    return domain_name.split('/')[-1][0]


def tb_log_all_config(tensorboard, args):
    if args.disable_prompt is not True:
        print(
            'why do you run this model? [enter reason below] (This will be logged in Tensorboard for future reference)')
        goal = input()
        tensorboard.add_text('goal', goal)
    arg_dict = vars(args)
    for k in arg_dict:
        tensorboard.add_text(k, str(arg_dict[k]))
