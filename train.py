import argparse
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn
import torch.cuda
import torch.nn
import torch.utils.data
from torchpack import distributed as dist
from torchpack.callbacks import InferenceRunner, MaxSaver, Saver
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs
from torchpack.utils.logging import logger

from core import builder
from core.callbacks import MeanIoU
from core.trainers import SemanticKITTITrainer

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')
    parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    parser.add_argument('--model-init', default=None, metavar='FILE', help='model location')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    if configs.distributed:
        dist.init()

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)

    # seed
    if ('seed' not in configs.train) or (configs.train.seed is None):
        configs.train.seed = torch.initial_seed() % (2 ** 32 - 1)

    seed = configs.train.seed + dist.rank(
    ) * configs.workers_per_gpu * configs.num_epochs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    dataset = builder.make_dataset()
    
    logger.info(' '.join([sys.executable] + sys.argv))
    logger.info(f'Experiment started: "{args.run_dir}".' + '\n' + f'{configs}')

    dataflow = {}
    for split in dataset:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset[split],
            num_replicas=dist.size(),
            rank=dist.rank(),
            shuffle=(split == 'train'))
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=configs.batch_size,
            sampler=sampler,
            num_workers=configs.workers_per_gpu,
            pin_memory=True,
            collate_fn=dataset[split].collate_fn)

    def cal_weights():
        sample_classes = np.zeros(configs.data.num_classes)
        for feed_dict in dataflow['train']:
            for i in range(configs.data.num_classes):
                sample_classes[i] += torch.sum(feed_dict['targets'].F == i).item()

    model = builder.make_model(args.model_init).cuda()
    # model.load_state_dict()
    if configs.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[dist.local_rank()], find_unused_parameters=True)

    wights_ce = torch.FloatTensor([1/58.25, 1]).cuda()
    criterion = builder.make_criterion()
    optimizer = builder.make_optimizer(model)
    scheduler = builder.make_scheduler(optimizer)

    trainer = SemanticKITTITrainer(model=model,
                                   criterion=criterion,
                                   optimizer=optimizer,
                                   scheduler=scheduler,
                                   num_workers=configs.workers_per_gpu,
                                   seed=seed,
                                   amp_enabled=configs.amp_enabled)
    trainer.train_with_defaults(
        dataflow['train'],
        num_epochs=configs.num_epochs,
        callbacks=[
            # ! _trigger_epoch
            InferenceRunner(
                dataflow[split],
                callbacks=[
                    MeanIoU(name=f'{split}/iou',
                            num_classes=configs.data.num_classes,
                            ignore_label=configs.data.ignore_label)
                ],
            ) for split in ['test']
        ] + [
            MaxSaver('test/iou'),
            Saver(),
        ])


if __name__ == '__main__':
    main()
