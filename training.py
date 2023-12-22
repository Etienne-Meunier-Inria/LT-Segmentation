from Models.CoherenceNet import CoherenceNet
from Models.CoherenceNets.MethodeB import MethodeB
from Models.LitSegmentationModel import LitSegmentationModel

from argparse import ArgumentParser
import pytorch_lightning as pl
from csvflowdatamodule.CsvDataset import CsvDataModule
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path
from utils.ExperimentalFlag import ExperimentalFlag as Ef
from utils.ExperimentalFlag import *
from utils.Callbacks import ResultsLogger

import wandb, yaml, os, sys, torch, json

# ------------
# args
# ------------

def ArgParser() :
    exp_flags = yaml.safe_load(open('utils/ExperimentalFlags.yaml'))
    parser = ArgumentParser()
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--experimental_flag', default = [],
                    choices=list(exp_flags.keys()), nargs='*')
    parser.add_argument('--save_dir', type=str, default=None)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitSegmentationModel.add_specific_args(parser)
    parser = CoherenceNet.add_specific_args(parser)
    parser = CsvDataModule.add_specific_args(parser)
    parser = MethodeB.add_specific_args(parser)
    return parser

def get_model(args) :
    if args.model_type == 'coherence_B' :
        model = MethodeB(**vars(args))
    else :
        raise(f'Model type : {args.model_type} not known')


    if args.finetune != 'none' :
        print('\n=== Finetuning ===')
        finetune = torch.load(args.finetune)
        st =  finetune['state_dict']
        del st['grid']
        if finetune['hyper_parameters']['nAffineMasks'] != args.nAffineMasks :
            print(f"Warning : different number of queries\
                    {finetune['hyper_parameters']['nAffineMasks']} \
                    vs {args.nAffineMasks} - reset queries")
            if finetune['hyper_parameters']['maskformer.transformer.queries_type'] == 'embeddings' :
                del st['backbone_model.model.model.Ta.queries.queriesInit.queries.weight']

        print(f'Loading Initial Weights from {args.finetune}')
        print(model.load_state_dict(st, strict=False))
        print('=======\n')
        model.init_param_model(args.img_size, args.len_seq, args.param_model)
    return model

def main(args) :
    Ef.set(args.experimental_flag)

    pl.seed_everything(args.seed)
    if torch.cuda.is_available() :
        args.gpus = 1
        args.auto_select_gpus = True
    print('Start Training')

    # ------------
    # model
    # ------------
    model = get_model(args)

    # ------------
    # data
    # ------------
    args.data_path  = f'DataSplit/{args.data_file}_'+'{}.csv'
    dm = CsvDataModule(request=model.request, **vars(args))
    # ------------
    # logger and callbacks
    # ------------
    logger = pl.loggers.CSVLogger(args.save_dir)


    # ------------
    # log model
    # ------------

    path = Path(args.save_dir+'/checkpoints/')
    path.mkdir(parents=True, exist_ok=True)
    # We save the model with the lowest validation loss.
    args.callbacks = [pl.callbacks.ModelCheckpoint(args.save_dir+'/checkpoints/',
                                                   monitor='val/loss',
                                                   filename='{epoch}-epoch_val_loss:{val/loss:.5f}',
                                                   mode='min',
                                                   auto_insert_metric_name=False,
                                                   save_top_k=1),
                      ResultsLogger(args.save_dir+'/results.csv')]


    # ------------
    # training
    # ------------
    args.gradient_clip_val=5.0
    print('\n\n\n')
    print(args)
    print('\n\n\n')

    trainer = pl.Trainer.from_argparse_args(args,  logger=logger)
    trainer.tune(model, dm)
    trainer.fit(model, dm)

if __name__ == '__main__' :
    print('main')
    parser = ArgParser()
    args = parser.parse_args()
    main(args)
