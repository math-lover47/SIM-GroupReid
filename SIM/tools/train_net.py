import sys
import numpy as np
import logging

from methods.SIM.fastreid.engine import DefaultTrainer, default_argument_parser, default_setup

sys.path.append('.')

from methods.SIM.fastreid.data.samplers.triplet_sampler import set_seed
from methods.SIM.fastreid.config import get_cfg
from fastreid.engine import launch
from methods.SIM.fastreid.utils.checkpoint import Checkpointer


# 读取配置文件
def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    return cfg


def main(args):
    set_seed(args.seed)  # 设置随机种子
    logger = logging.getLogger(__name__)
    logger.info(f"Using random seed: {args.seed}")

    cfg = setup(args)
    # 模型测试
    if args.eval_only:  # args.eval_only = True  # 是否测试模型,False表示训练模型，True表示测试模型
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = DefaultTrainer.build_model(cfg)
        # 加载预训练模型
        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

        res = DefaultTrainer.test(cfg, model)
        return res
    # 模型训练
    trainer = DefaultTrainer(cfg)

    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    # args = default_argument_parser().parse_args()

    # 调试使用，使用的时候删除下面代码
    # ---
    # args.config_file = "./configs/Market1501/bagtricks_R50.yml"  # config路径
    # args.eval_only = True  # 是否测试模型,False表示训练模型，True表示测试模型
    # ---

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args, ),
    )
