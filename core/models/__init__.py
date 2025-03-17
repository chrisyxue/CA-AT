import torch

from .resnet import Normalization
from .preact_resnet import preact_resnet
from .resnet import resnet
from .wideresnet import wideresnet

from .preact_resnetwithswish import preact_resnetwithswish
from .wideresnetwithswish import wideresnetwithswish

from core.data import DATASETS



MODELS = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 
          'preact-resnet18', 'preact-resnet34', 'preact-resnet50', 'preact-resnet101', 
          'wrn-28-10', 'wrn-34-10', 'wrn-34-20', 
          'preact-resnet18-swish', 'preact-resnet34-swish',
          'wrn-28-10-swish', 'wrn-34-20-swish', 'wrn-70-16-swish', 'tiny-resnet18','tiny-renset50']


def create_model(name, normalize, info, device, args):
    """
    Returns suitable model from its name.
    Arguments:
        name (str): name of resnet architecture.
        normalize (bool): normalize input.
        info (dict): dataset information.
        device (str or torch.device): device to work on.
    Returns:
        torch.nn.Module.
    """
    if info['data'] in ['tiny-imagenet']:
        if 'preact-resnet' in name and 'swish' not in name:
            from .ti_preact_resnet import ti_preact_resnet
            backbone = ti_preact_resnet(name, num_classes=info['num_classes'], device=device)
            normalize = True
        elif name == 'tiny-renset50':
            backbone = resnet(name, num_classes=info['num_classes'], pretrained=False, device=device, classifier=args.classifier)
            checkpoint_path = '/home/zhiyuxue/Adver/adversarial_robustness_pytorch/pretrained_models/tiny_resnet50_2.pth'
            weights = torch.load(checkpoint_path)['model']
            backbone.load_state_dict(weights)
            normalize = True
            # import pdb; pdb.set_trace()
            # backbone.load_state_dict(weights.get_state_dict(progress=True, check_hash=True),strict=False)
        elif name == 'tiny-resnet18':
            backbone = resnet(name, num_classes=info['num_classes'], pretrained=False, device=device, classifier=args.classifier)
            checkpoint_path = '/home/zhiyuxue/Adver/adversarial_robustness_pytorch/pretrained_models/tiny_resnet18.pth'
            weights = torch.load(checkpoint_path)['model']
            backbone.load_state_dict(weights)
            normalize = True
        else:
            raise ValueError('Invalid model name {}!'.format(name))
        # import pdb; pdb.set_trace()
        # assert 'preact-resnet' in name, 'Only preact-resnets are supported for this dataset!'
        # from .ti_preact_resnet import ti_preact_resnet
        # backbone = ti_preact_resnet(name, num_classes=info['num_classes'], device=device)
    
    elif info['data'] in DATASETS and info['data'] not in ['tiny-imagenet']:
        if 'preact-resnet' in name and 'swish' not in name:
            backbone = preact_resnet(name, num_classes=info['num_classes'], pretrained=False, device=device)
        elif 'preact-resnet' in name and 'swish' in name:
            backbone = preact_resnetwithswish(name, dataset=info['data'], num_classes=info['num_classes'])
        elif 'resnet' in name and 'preact' not in name:
            backbone = resnet(name, num_classes=info['num_classes'], pretrained=False, device=device, classifier=args.classifier)
        elif 'wrn' in name and 'swish' not in name:
            backbone = wideresnet(name, num_classes=info['num_classes'], device=device)
        elif 'wrn' in name and 'swish' in name:
            backbone = wideresnetwithswish(name, dataset=info['data'], num_classes=info['num_classes'], device=device)
        else:
            raise ValueError('Invalid model name {}!'.format(name))
    else:
        raise ValueError('Models for {} not yet supported!'.format(info['data']))
    
    # import pdb; pdb.set_trace()
        
    if normalize:
        model = torch.nn.Sequential(Normalization(info['mean'], info['std']), backbone)
    else:
        model = torch.nn.Sequential(backbone)
    
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    return model
