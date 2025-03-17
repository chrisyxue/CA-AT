import argparse

from core.attacks import ATTACKS
from core.data import DATASETS
from core.models import MODELS
from .train import SCHEDULERS
from .utils import str2bool, str2float

GRAD_OP = ['avg','thres','orth','thres_layer','avg_ablation']
ADVER_LOSS = ['adver','mart','trades','clp']
ADVER_CRITERION = ['ce','ce_ls','dlr','dlr_softmax']

def parser_train():
    """
    Parse input arguments (train.py).
    """
    parser = argparse.ArgumentParser(description='Standard + Adversarial Training.')

    parser.add_argument('--augment', type=str2bool, default=True, help='Augment training set.')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size for training.')
    parser.add_argument('--batch-size-validation', type=int, default=512, help='Batch size for testing.')
    
    parser.add_argument('--data-dir', type=str, default='/cluster/home/rarade/data/')
    parser.add_argument('--log-dir', type=str, default='/cluster/scratch/rarade/test/')
    
    parser.add_argument('-d', '--data', type=str, default='cifar10s', choices=DATASETS, help='Data to use.')
    parser.add_argument('--desc', type=str, required=True, 
                        help='Description of experiment. It will be used to name directories.')

    parser.add_argument('-m', '--model', choices=MODELS, default='wrn-28-10-swish', help='Model architecture to be used.')
    parser.add_argument('--normalize', type=str2bool, default=False, help='Normalize input.')
    # parser.add_argument('--pretrained-file', type=str, default=None, help='Pretrained weights file name.')

    parser.add_argument('-na', '--num-adv-epochs', type=int, default=400, help='Number of adversarial training epochs.')
    parser.add_argument('--adv-eval-freq', type=int, default=25, help='Adversarial evaluation frequency (in epochs).')
    
    parser.add_argument('--lr', type=float, default=0.4, help='Learning rate for optimizer (SGD).')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Optimizer (SGD) weight decay.')
    parser.add_argument('--scheduler', choices=SCHEDULERS, default='cosinew', help='Type of scheduler.')
    parser.add_argument('--nesterov', type=str2bool, default=True, help='Use Nesterov momentum.')

    parser.add_argument('-a', '--attack', type=str, choices=ATTACKS, default='linf-pgd', help='Type of attack.')
    parser.add_argument('--attack-eps', type=str2float, default=8/255, help='Epsilon for the attack.')
    parser.add_argument('--attack-step', type=str2float, default=2/255, help='Step size for PGD attack.')
    parser.add_argument('--attack-iter', type=int, default=10, help='Max. number of iterations (if any) for the attack.')

    parser.add_argument('--debug', action='store_true', default=False, 
                        help='Debug code. Run 1 epoch of training and evaluation.')

    parser.add_argument('--beta', default=0, type=float, help='TradeOff')
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')
    parser.add_argument('--adver_loss',choices=ADVER_LOSS,default='adver')
    parser.add_argument('--adver_criterion',choices=ADVER_CRITERION,default='ce',help='the loss to generate adv examples')
    parser.add_argument('--grad_op',choices=GRAD_OP,default='avg')
    parser.add_argument('--gpu',default=0,help='the id of gpu device')
    parser.add_argument('--grad_thres',default=0.5,type=float,help='thres for projection')
    parser.add_argument('--grad_track',action='store_true',help='track gradients or not')
    parser.add_argument('--grad_track_class',action='store_true',help='track gradients for targeted attack on each class')
    parser.add_argument('--grad_track_multiattack',action='store_true',help='track gradients for different attacks')
    parser.add_argument('--clip_grad', type=float, default=None, help='Gradient norm clipping.')
    parser.add_argument('--grad_norm',action='store_true',help='Normalize adv/clean gradient before operation')
    parser.add_argument('--classifier',type=str,default='linear',choices=['linear','cosine'],help='The type of classifier')
    parser.add_argument('--use_pretrain',action='store_true',help='Use pretrained model as the initialization')
    parser.add_argument('--use_ls', action='store_true', default=False, help='use label smooth or not')
    parser.add_argument('--ls_factor', type=float, default=0.1, help='factor of label smooth')

    # adaptive strategy of thres
    parser.add_argument('--adaptive_thres',default='No',type=str,choices=['No','StdAccLin','StdAccExp'],help='adaptive strategy of thres in gradient surgery')
    # use pretrained paramter
    parser.add_argument('--pretrain_type',default='No',choices=['No','In-Domain','ImageNet'])

    return parser


def parser_eval():
    """
    Parse input arguments (eval-adv.py, eval-corr.py, eval-aa.py).
    """
    parser = argparse.ArgumentParser(description='Robustness evaluation.')

    parser.add_argument('--data-dir', type=str, default='/cluster/home/rarade/data/')
    parser.add_argument('--log-dir', type=str, default='/cluster/scratch/rarade/test/')
        
    parser.add_argument('--desc', type=str, required=True, help='Description of model to be evaluated.')
    parser.add_argument('--num-samples', type=int, default=1000, help='Number of test samples.')
    
    # eval-aa.py
    parser.add_argument('--train', action='store_true', default=False, help='Evaluate on training set.')
    parser.add_argument('-v', '--version', type=str, default='standard', choices=['custom', 'plus', 'standard'], 
                        help='Version of AA.')

    # eval-adv.py
    parser.add_argument('--source', type=str, default=None, help='Path to source model for black-box evaluation.')
    parser.add_argument('--wb', action='store_true', default=False, help='Perform white-box PGD evaluation.')
    
    # eval-rb.py
    parser.add_argument('--threat', type=str, default='corruptions', choices=['corruptions', 'Linf', 'L2'],
                        help='Threat model for RobustBench evaluation.')
    
    parser.add_argument('--rep_analysis', action='store_true', help='Use Representation Analysis or Not')
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')
    parser.add_argument('--classifier',type=str,default='linear',choices=['linear','cosine'],help='The type of classifier')
    return parser

