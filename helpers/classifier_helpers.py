import sys, os
import torch
import torch as ch
import numpy
import itertools
import torch.nn as nn
from collections import OrderedDict
from robustness.model_utils import make_and_restore_model
import dill

def eval_accuracy(model, loader, alt=False, normalize=None):
    labels, preds = [], []
    with ch.no_grad():
        for _, (im, targ) in enumerate(loader):
            if normalize:
                im = normalize(im.cuda())
            if alt:
                op, _ = model(im.cuda())
            else:
                op = model(im.cuda()) #model(normalizer(im))
            preds.append(op.argmax(dim=1).cpu())
            labels.append(targ)
    return ch.cat(preds), ch.cat(labels)

def load_clip(arch='clip_RN50'):
    from clip.clip import _download, _MODELS, _transform_unnorm
    from clip.model_editing import build_model
    
    arch = arch.split('_')[1]
    
    model_path = _download(_MODELS[arch])
    model = torch.jit.load(model_path, map_location="cpu").eval()    
    model = build_model(model.state_dict()).cuda()
    
    model.cache_dir = f'/data/theory/robustopt/editing/model_info/other/clip/{arch}'
    if not os.path.exists(model.cache_dir): os.makedirs(model.cache_dir)
    model.zeroshot_classifier()
    return model, _transform_unnorm(model.visual.input_resolution)

def load_checkpoint(model, dataset, resume_path, 
                    new_code=True, arch='resnet50'):
    
    if arch.startswith('clip'):
        return load_clip(arch)
    
    checkpoint = ch.load(resume_path, pickle_module=dill)
    if isinstance(model, str):
        model = dataset.get_model(model, False).cuda()
        model.eval()
        pass
    
        
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict= checkpoint['state_dict']
    else:
        state_dict = checkpoint
    sd = {}
    
    for i, (k, v) in enumerate(state_dict.items()):
        if arch == 'resnet50':
            if k[len('module.'):].startswith('model'):
                kk = k[len('module.model.'):]
                if arch == 'resnet50':
                    if kk.startswith('conv') or kk.startswith('bn'):
                        kk = 'layer0.' + kk
                    elif kk.startswith('fc'):
                        kk = 'layer5.' + kk

                    if kk.startswith('layer0'):
                        rep_key = 'layer0'
                        kk = kk.replace(rep_key, model.sequence_dict[rep_key])
                    elif kk.startswith('layer5'):
                        rep_key = 'layer5'
                        kk = kk.replace(rep_key, model.sequence_dict[rep_key])
                    else:
                        rep_key = kk[:len('layer1.0')]
                        kk = kk.replace(rep_key, model.sequence_dict[rep_key])
                        if new_code:
                            if 'downsample' in kk:
                                kks = kk.split('.')
                                kks.insert(1, "residual")
                            else:
                                kks = kk.split('.')
                                kks.insert(-1, "module")
                            kk = '.'.join(kks) 
                        for rr in ['.conv3.',  '.bn3.', '.relu.', '.residual.']:
                            if rr in kk:
                                kk = kk.replace(rr, '.final'+rr)
                sd[kk] = v
            else:
                continue
        elif arch == 'resnet18':
            kk = k
            if kk.startswith('conv') or kk.startswith('bn'):
                kk = 'layer0.' + kk
            elif kk.startswith('fc'):
                kk = 'layer5.' + kk

            if kk.startswith('layer0'):
                rep_key = 'layer0'
                kk = kk.replace(rep_key, model.sequence_dict[rep_key])
            elif kk.startswith('layer5'):
                rep_key = 'layer5'
                kk = kk.replace(rep_key, model.sequence_dict[rep_key])
            else:
                rep_key = kk[:len('layer1.0')]
                kk = kk.replace(rep_key, model.sequence_dict[rep_key])
                if new_code:
                    if 'downsample' in kk:
                        kks = kk.split('.')
                        kks.insert(1, "residual")
                    else:
                        kks = kk.split('.')
                        kks.insert(-1, "module")
                    kk = '.'.join(kks) 
                for rr in ['.conv2.',  '.bn2.', '.relu.', '.residual.']:
                    if rr in kk:
                        kk = kk.replace(rr, '.final'+rr)
            sd[kk] = v
        elif arch == 'vgg16_bn':
            if k[len('module.model.'):].startswith('model'):
                kk = k[len('module.model.model.'):]
                kks = kk.split('.')
                p, s = '.'.join(kks[:-1]), kks[-1]
                if p in model.sequence_dict:
                    kk = f"{model.sequence_dict[p]}.{s}"
                else:
                    kk = f"{p}.{s}"
                sd[kk] = v
            else:
                continue
        elif arch == 'vgg16' and k.startswith('model'):
            kk = k[len('model.'):]
            kks = kk.split('.')
            p, s = '.'.join(kks[:-1]), kks[-1]
            if p in model.sequence_dict:
                kk = f"{model.sequence_dict[p]}.{s}"
            else:
                kk = f"{p}.{s}"
            sd[kk] = v
        elif arch == 'vgg16' and k.startswith('features'):
            suffix = k.split(".")[-1]
            kk = f'layer{i // 2}.conv.{suffix}'
            sd[kk] = v
        elif arch == 'vgg16' and k.startswith('classifier'):
            layer= {'fc6': 0,
                    'fc7': 3,
                    'fc8a': 6,
                   }[k.split('.')[1]]
            suffix = k.split(".")[-1]
            kk = f'classifier.{layer}.{suffix}'
            sd[kk] = v
        else: 
            raise ValueError('unrecognizable checkpoint')
            
    model.load_state_dict(sd)
    model.eval()
    pass

    return model

def load_classifier(model_path, model_class, arch, dataset, layernum):
    
    preprocess = None
    mod = load_checkpoint(model=model_class, 
                            dataset=dataset,
                            resume_path=model_path,
                            arch=arch)
    if arch.startswith('clip'):
        mod, preprocess = mod
    mod = mod.cuda()
    
    con_mod, Nfeatures = get_context_model(mod, layernum, arch)
    if arch.startswith('vgg'):
        targ_mod = mod[layernum + 1]
    elif arch.startswith('clip'):
        targ_mod = mod.visual[layernum + 1].final
    else:
        targ_mod = mod[layernum + 1].final
        
    if not arch.startswith('clip'):
        return mod, con_mod, targ_mod
    else:
        return mod, con_mod, targ_mod, preprocess