import os
import numpy as np
import yaml
import argparse




if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='/home/krishna/Krishna/keyword-detection/egs/speech_commands/conf/transformer_v2_35.yaml')
    parser.add_argument('-n', '--ngpu', type=int, default=1)
    parser.add_argument('-g', '--gpus', type=str, default='0')
    parser.add_argument('-se', '--seed', type=int, default=1234)
    parser.add_argument('-p', '--parallel_mode', type=str, default='dp')
    parser.add_argument('-r', '--local_rank', type=int, default=0)
    parser.add_argument('-l', '--logging_level', type=str, default='info', choices=['info','debug'])
    parser.add_argument('-lg', '--log_file', type=str, default=None)
    parser.add_argument('-mp', '--mixed_precision', action='store_true', default=False)
    parser.add_argument('-ct', '--continue_training', action='store_true', default=False)
    parser.add_argument('-dir', '--expdir', type=str, default=None)
    parser.add_argument('-im', '--init_model', type=str, default=None)
    parser.add_argument('-ios', '--init_optim_state', type=str, default=None)
    parser.add_argument('-debug', '--debug', action='store_true', default=False)
    parser.add_argument('-knpt', '--keep_last_n_chkpt', type=int, default=30)
    parser.add_argument('-tfs', '--from_step', type=int, default=0)
    parser.add_argument('-tfe', '--from_epoch', type=int, default=0)
    parser.add_argument('-vb', '--verbose', type=int, default=0)
    parser.add_argument('-ol', '--opt_level', type=str, choices=['O0', 'O1', 'O2', 'O3'], default='O1')
    cmd_args = parser.parse_args()




    with open(cmd_args.config) as parameters:
        params = yaml.safe_load(parameters)


        if cmd_args.expdir is not None:
        expdir = os.path.join(cmd_args.expdir, params['train']['save_name'])
    else:
        expdir = os.path.join('egs', params['data']['name'], 'exp', params['train']['save_name'])
    if not os.path.exists(expdir):
        os.makedirs(expdir)

    shutil.copy(cmd_args.config, os.path.join(expdir, 'config.yaml'))

    logging_level = {
        'info': logging.INFO,
        'debug': logging.DEBUG
    }

    if cmd_args.log_file is not None:
        log_file = cmd_args.log_file
    else:
        log_file = cmd_args.config.split('/')[-1][:-5] + '.log'
    
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging_level[cmd_args.logging_level], format=LOG_FORMAT)
    logger = logging.getLogger(__name__)

    if cmd_args.ngpu > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cmd_args.gpus)
        logger.info('Set CUDA_VISIBLE_DEVICES as %s' % cmd_args.gpus)

    if cmd_args.parallel_mode == 'ddp':
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ["OMP_NUM_THREADS"] = '1'
        torch.cuda.set_device(cmd_args.local_rank)