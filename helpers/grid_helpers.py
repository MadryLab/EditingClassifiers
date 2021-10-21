import torch, copy, os, json, warnings, sys, cox, random, string, dill, git
warnings.filterwarnings("ignore")
sys.path.append('..')
import torch as ch
import torchvision
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from argparse import Namespace
import webdataset as wds

from robustness import datasets
from robustness.main import setup_store_with_metadata
from robustness.tools.helpers import has_attr
from robustness import __version__

import helpers.context_helpers as coh
import helpers.load_helpers as lh
import helpers.match_helpers as mh
import helpers.gen_helpers as gh
from helpers.seg_helpers import LABEL_DICT
from utils import nethook
from utils.custom_folder import ImageFolder

REQ = 'REQUIRED'


CONFIG_ARGS = [['dataset-name', str, 'name of dataset', REQ],
               ['out-dir', str, 'output dir', REQ],
               ['concepts', str, 'mode of segmentations', REQ],
               ['concept-dir', str , 'source of concepts', '/mnt/nfs/projects/editing/datasets/segmentations/'],
               ['styles', list, 'style names', REQ],
               ['style-dir', str, 'path to stylized images', '/mnt/nfs/projects/editing/datasets'],
               ['arch', str, 'network architecture', REQ],
               ['epsilon', float, 'type of model', 0],
               ['layernum', int, 'layer to edit', REQ],
               ['random-seed', int, 'random seed', 0],
               ['num-workers', int, 'num workers', 10],
               ['batch-size', int, 'batch size', 50],
               ['anno-filename', str, 'annotation path', '/mnt/nfs/projects/editing/results/style_eval/web_results/'],
               ['anno-min-classes', int, 'min classes', 3],
               ['anno-keys', list, 'yes/maybe/no', ['yes']],
               ['conf-thresh', float, 'min confidence of seg', 0.8],
               ['pixel-thresh', int, 'min pixels in seg', 100],
               ['cache-dir', str, 'model cache', '/data/theory/robustopt/editing/model_info/predictions/info/'],
               ['cov-dir', str, 'cov cache', '/data/theory/robustopt/editing/model_info/covariances_custom'],
               ['save-checkpoint', bool, 'save weights', False],
               ['expt-name', str, 'experiment name', ''],
]
 
REWRITE_ARGS = [    
    ['mode-rewrite', str, 'editing/finetune/finetune_local', REQ],
    ['nconcept', int, '# examples to use for concept matching', 3],
    ['ntrain', int, '# examples to use for training', 1],
    ['nsteps', int, '# editing steps', REQ],
    ['nsteps-proj', int, '# projection steps', 10],
    ['use-mask', bool, 'use mask for editing', bool],
    ['lr', float, 'learning rate', REQ],
    ['mode-concept', str, 'zca/cov', 'zca'],
    ['restrict-rank', bool, 'restrict rank while editing', True],  
    ['rank', int, 'rank while editing', 1]] 

DATA_INFO_SCHEMA = {'target_label': int, 
                    'style_name': str,
                    'concept_name': str,
                    'idx_train': cox.store.PICKLE,
                    'idx_test': cox.store.PICKLE,
                    'labels_train': cox.store.PICKLE,
                    'labels_test': cox.store.PICKLE}

RESULTS_SCHEMA = {'pre_acc': float,
                  'post_acc': float,
                  'train_pre_imgs': cox.store.PICKLE, 
                  'train_post_imgs': cox.store.PICKLE, 
                  'train_pre_manip_imgs': cox.store.PICKLE, 
                  'train_post_manip_imgs': cox.store.PICKLE,
                  'test_pre_imgs': cox.store.PICKLE, 
                  'test_post_imgs': cox.store.PICKLE,
                  'test_pre_manip_imgs_same': cox.store.PICKLE, 
                  'test_post_manip_imgs_same': cox.store.PICKLE,
                  'test_pre_manip_imgs_diff': cox.store.PICKLE, 
                  'test_post_manip_imgs_diff': cox.store.PICKLE,
                  }

def setup_store_with_metadata(args, store_name=None):
    '''
    Sets up a store for training according to the arguments object. See the
    argparse object above for options.
    '''
    # Add git commit to args
    try:
        repo = git.Repo(path=os.path.dirname(os.path.realpath(__file__)),
                            search_parent_directories=True)
        version = repo.head.object.hexsha
    except git.exc.InvalidGitRepositoryError:
        version = __version__
    args.version = version

    # Create the store
    store = cox.store.Store(args.out_dir) if store_name is None else \
            cox.store.Store(args.out_dir, store_name)
    args_dict = args.__dict__
    schema = cox.store.schema_from_dict(args_dict)
    store.add_table('metadata', schema)
    store['metadata'].append_row(args_dict)

    return store

def init_args(args):
    args = cox.utils.Parameters(args.__dict__)
    args = check_and_fill_args(args, CONFIG_ARGS)
    args = check_and_fill_args(args, REWRITE_ARGS)
    args = Namespace(**args.__dict__['params'])
    return args

def check_and_fill_args(args, arg_list):

    for arg_name, _, _, arg_default in arg_list:
        name = arg_name.replace("-", "_")
        if has_attr(args, name): 
            continue
        elif arg_default == REQ: raise ValueError(f"{arg_name} required")
        elif arg_default is not None: 
            setattr(args, name, arg_default)
    return args

def find_impacted_classes(args, concept_name):
    
    anno_path = os.path.join(args.anno_filename, args.dataset_name, args.concepts, 'dimitris')
    
    anno_map = {}
    for a in sorted(os.listdir(anno_path)):
        match = a.split('_')[1:-1]
        anno_map[' '.join(match)] = a
        
    if concept_name not in anno_map:
        return None
    
    with open(os.path.join(anno_path, anno_map[concept_name])) as f: 
        res = json.load(f)
    
    count, val = 0, Counter(list(res.values()))
    for k in args.anno_keys:
        count += val[k]

    if count < args.anno_min_classes: 
        return None
        
    return [int(k.split('_')[1]) for k, v in res.items() if v in args.anno_keys] 
    

def load_concepts(args, test_loader, cache_file=None):
            
    # Only keep reasonable concepts
    all_concepts_dict = {}
    for k, v in LABEL_DICT[args.concepts].items():
        vc = find_impacted_classes(args, v) 
        if vc is None: continue
        all_concepts_dict[v] = (k, vc)   
    
    print(f"----Number of valid concepts: {len(all_concepts_dict)}")
    print(cache_file)
    
    # Load concept segmentations
    if args.concepts != 'LVIS':
        load_dir = f"{args.concept_dir}/{args.dataset_name}_{args.concepts}_webd/dest.tar"
    else:
        load_dir = f"{args.concept_dir}/test_{args.dataset_name}_{args.concepts}/dest.tar"
        
        
    dataset_concept = (
        wds.Dataset(load_dir)
        .decode("rgb")
    )
    
    print(f"----GOT WDSS---")
    
    if cache_file is not None and os.path.exists(cache_file):
        return {'concepts': all_concepts_dict,
            'dataset': dataset_concept, 
            'indices': np.load(cache_file, allow_pickle=True).item()}
    
    loader_concept = ch.utils.data.DataLoader(dataset_concept, 
                                              num_workers=args.num_workers,
                                              batch_size=args.batch_size, 
                                              shuffle=False)
    
    valid_indices = {}
    count = 0
    for bi, (batch, sample) in tqdm(enumerate(zip(test_loader, loader_concept)), total=len(test_loader)):

        indices = bi * args.batch_size + np.arange(batch[0].shape[0])
        count += len(indices)

        if args.concepts != 'LVIS':
            segs, confs = sample['input.pyd'], sample['output.pyd']
            segs = segs.unsqueeze(1).repeat(1, 3, 1, 1)
            confs = confs.unsqueeze(1).repeat(1, 3, 1, 1)
        else:
            segs, confs = sample['class.pyd'], sample['score.pyd']
            
        for v in all_concepts_dict:
            if args.concepts == 'LVIS':
                masks = ((segs == all_concepts_dict[v][0]) & (confs > args.conf_thresh)).max(axis=1)[0]
                masks = masks.unsqueeze(1).repeat(1, 3, 1, 1).float()
            else:
                masks = ((segs == all_concepts_dict[v][0]) & (confs > args.conf_thresh)).float()
                
            pix = masks.sum((1,2,3)).int().numpy() // 3
            idx = np.where(np.logical_and(np.isin(batch[1].numpy(), all_concepts_dict[v][1]),
                                      pix > args.pixel_thresh))[0]
            if len(idx) == 0: 
                continue
            if v not in valid_indices: valid_indices[v] = []
            valid_indices[v].extend(indices[idx])
            
    print("Total:", count)
    if cache_file is not None:
        np.save(cache_file, valid_indices)
        
    return {'concepts': all_concepts_dict,
            'dataset': dataset_concept, 
            'indices': valid_indices}

def preprocess_imgs(imgs, preprocess):
    if preprocess is None:
        return imgs
    else:
        return ch.stack([preprocess(torchvision.transforms.ToPILImage()(im)) for im in imgs])

def load_single_concept(args, concept_dict, concept_name, test_loader, preprocess=None, cache_file=None):
    
    if cache_file is not None and os.path.exists(cache_file):
        try:
            return ch.load(cache_file)
        except:
            print("Error loading cached data...Recomputing")

    concept_num = concept_dict['concepts'][concept_name][0]
    dataset_concept = concept_dict['dataset']
    valid_idx = concept_dict['indices'][concept_name]
    assert len(valid_idx) > 0
    
    loader_concept = ch.utils.data.DataLoader(dataset_concept, 
                                              num_workers=args.num_workers,
                                              batch_size=args.batch_size, 
                                              shuffle=False)
    
    all_indices, all_ims, all_labels, all_masks = [], [], [], []

    for bi, (batch, sample) in tqdm(enumerate(zip(test_loader, loader_concept)), total=len(test_loader)):

        indices = bi * args.batch_size + np.arange(batch[0].shape[0])
        
        idx = np.isin(indices, valid_idx)
        if len(idx) == 0:
            continue
        
        if args.concepts != 'LVIS':
            segs = sample['input.pyd'].unsqueeze(1).repeat(1, 3, 1, 1)
            confs = sample['output.pyd'].unsqueeze(1).repeat(1, 3, 1, 1)
            masks = ((segs == concept_num) & (confs > args.conf_thresh)).float()
        else:
            segs, confs = sample['class.pyd'], sample['score.pyd']
            masks = ((segs == concept_num) & (confs > args.conf_thresh)).max(axis=1)[0]
            masks = masks.unsqueeze(1).repeat(1, 3, 1, 1).float()
        
        for k, v in zip([all_indices, all_ims, all_labels, all_masks], 
                        [indices, batch[0], batch[1], masks]):
            k.append(v[idx])

    all_ims, all_labels = ch.cat(all_ims), ch.cat(all_labels)
    all_indices, all_masks = np.concatenate(all_indices), ch.cat(all_masks)
    
    assert np.array_equal(valid_idx, all_indices)
    
    ret = {'idx': all_indices, 
            'imgs': preprocess_imgs(all_ims, preprocess=preprocess), 
            'labels': all_labels, 
            'masks': preprocess_imgs(all_masks, preprocess=preprocess)}
    
    if cache_file is not None:
        ch.save(ret, cache_file)
    return ret

def load_style_images(args, style_name, idx_to_style, preprocess=None):
    print("----Loading style images----")
    style_path = os.path.join(args.style_dir, f'{args.dataset_name}/all_styles', style_name) 
    
    dataset_style = ImageFolder(style_path, 
                                transform=torchvision.transforms.ToTensor(),
                                img_mapping=idx_to_style)

    loader_style = ch.utils.data.DataLoader(dataset_style, num_workers=args.num_workers,
                                            batch_size=args.batch_size, shuffle=False)
        
    imgs_style, labels_style = [], []

    for _, style in tqdm(enumerate(loader_style), total=len(loader_style)):
        imgs_style.append(style[0])
        labels_style.append(style[1])

    return preprocess_imgs(ch.cat(imgs_style), preprocess=preprocess), ch.cat(labels_style)

def interpolate(imgs, masks, stylized, batch_size=25):
    interpolated = []
    batch_count = int(np.ceil(len(imgs) / batch_size))
    
    for b in range(batch_count):
        interpolated.append(imgs[b*batch_size:(b+1)*batch_size] * (1 - masks[b*batch_size:(b+1)*batch_size]) + \
                            stylized[b*batch_size:(b+1)*batch_size] * masks[b*batch_size:(b+1)*batch_size])
    return ch.cat(interpolated)

def obtain_train_test_splits(args, concept, class_dict, style_name, preprocess=None, rng=None):
    
    all_imgs, all_labels = concept['imgs'], concept['labels']
    
    label_counter = Counter(all_labels.numpy())
    allowed_labels = [k for k, v in label_counter.items() if v >= args.ntrain]
    target_label = rng.choice(allowed_labels, 1)[0]
    
    print("Target label: ", target_label)
    print("Examples of relevant classes:")
    print([(k, class_dict[k], v) for ii, (k, v) in enumerate(label_counter.items()) if ii < 5])
    
    rel_idx = np.where(all_labels.numpy() == target_label)[0]
        
    idx_train = rel_idx[rng.choice(len(rel_idx), args.nconcept, replace=False)]
    idx_test = np.array(list(set(np.arange(len(all_labels))) - set(idx_train)))
    
    style_path = os.path.join(args.style_dir, f'{args.dataset_name}/all_styles', style_name)
    Nstyles = len(os.listdir(style_path))
    train_style = rng.choice(Nstyles, 1)[0]
    assert train_style < Nstyles
    style_number_test = rng.choice(list(set(np.arange(Nstyles)) - set([train_style])), len(idx_test))
    assert train_style not in style_number_test
    
    idx_all = concept['idx'][np.concatenate([idx_train, idx_test])]
    style_number_same = np.concatenate([[train_style] * len(idx_train), [train_style] * len(idx_test)])
    style_number_diff = np.concatenate([[train_style] * len(idx_train), style_number_test])
    
    idx_to_style = {k: v for k, v in zip(idx_all, style_number_same)}
    stylized_img_same, style_labels_same = load_style_images(args, 
                                                             style_name, 
                                                             idx_to_style,
                                                             preprocess=preprocess)
    assert np.array_equal(style_labels_same.numpy(), np.array(list(idx_to_style.keys())))
    
    idx_to_style = {k: v for k, v in zip(idx_all, style_number_diff)}
    stylized_img_diff, style_labels_diff = load_style_images(args, 
                                                             style_name, 
                                                             idx_to_style,
                                                             preprocess=preprocess)
   
    
    assert np.array_equal(style_labels_diff.numpy(), np.array(list(idx_to_style.keys())))
    
    data_dict, data_info_dict = {}, {}

    # Training data
    data_dict['train_data'] = {k: v[idx_train] for k, v in concept.items()}
    
    check_labels = np.unique(data_dict['train_data']['labels'].numpy())
    assert len(check_labels) == 1
    assert check_labels[0] == target_label

        
    data_dict['train_data']['manip_imgs'] = interpolate(data_dict['train_data']['imgs'], 
                                                        data_dict['train_data']['masks'], 
                                                        stylized_img_same[:len(idx_train)])
    
    # Test data
    data_dict['test_data'] = {k: concept[k][idx_test] for k in data_dict['train_data']
                              if k != 'manip_imgs'}
    data_dict['test_data']['manip_imgs_same'] = interpolate(data_dict['test_data']['imgs'], 
                                                        data_dict['test_data']['masks'], 
                                                        stylized_img_same[len(idx_train):])
    data_dict['test_data']['manip_imgs_diff'] = interpolate(data_dict['test_data']['imgs'], 
                                                        data_dict['test_data']['masks'], 
                                                        stylized_img_diff[len(idx_train):])

    
    
    data_info_dict = {'target_label': target_label, 
                      'idx_train': data_dict['train_data']['idx'], 
                      'idx_test': data_dict['test_data']['idx'], 
                      'labels_train': data_dict['train_data']['labels'], 
                      'labels_test': data_dict['test_data']['labels'], 
                     }
    return data_dict, data_info_dict


def edit_model(args, data_dict, 
               context_model, 
               target_model=None, 
               ZM_k=None):
                
    assert args.ntrain <= len(data_dict['train_data']['imgs'])
    cp_imgs = ch.cat([data_dict['train_data']['imgs'][:args.ntrain], 
                      data_dict['train_data']['manip_imgs'][:args.ntrain]]).float()
    cp_masks = ch.cat([data_dict['train_data']['masks'][:args.ntrain], 
                       data_dict['train_data']['masks'][:args.ntrain]]).float()
    
    Nims = len(cp_imgs)

    if args.mode_rewrite == 'editing':
        assert (target_model is not None) and (ZM_k is not None)
        context_k = coh.get_context_key(data_dict['train_data']['manip_imgs'].float(), 
                                        data_dict['train_data']['masks'], 
                                        context_model, ZM_k, rank=args.rank)
    
        with ch.no_grad(): context_model(cp_imgs.cuda())

        kstar = coh.features['pre']
        vstar = coh.features['post'][:Nims//2].detach().clone()
        kstar = (kstar[0][Nims//2:].detach().clone(), kstar[1][Nims//2:].detach().clone()) if not args.arch.startswith('vgg') \
                else kstar[Nims//2:].detach().clone()
        
        mstar = ch.max(cp_masks[:Nims//2], dim=1, keepdims=True)[0]
        print(f"--In here {mstar.shape}")

        mh.edit_classifier_weights(target_model, kstar, vstar, 
                                   context_k, niter=args.nsteps, 
                                   piter=args.nsteps_proj, lr=args.lr, 
                                   low_rank_insert=args.restrict_rank, 
                                   mask=mstar.cuda() if args.use_mask else None)
    else:
        if args.arch == 'resnet50':
            first_layer = f'layer{args.layernum + 1}.final.conv3'  
        elif args.arch == 'resnet18':
            first_layer = f'layer{args.layernum + 1}.final.conv2'  
        elif args.arch.startswith('vgg'):
            first_layer = f'layer{args.layernum}.conv'
        else:
            first_layer = f'visual.layer{args.layernum + 1}.final'  
                    
                
        if args.mode_rewrite == 'finetune_local':
            edit_params = [mh.target_weights(target_model)]
        else:
            edit_model = nethook.subsequence(context_model, 
                                             first_layer=first_layer,
                                             share_weights=True)
            
            edit_params = edit_model.parameters()
            
            if args.arch.startswith('clip'):
                edit_params = []
                for name, param in edit_model.named_parameters():
                    if 'visual' in name:
                        edit_params.append(param)

        optimizer = ch.optim.SGD(edit_params, lr=args.lr)
        compute_loss = torch.nn.CrossEntropyLoss()
        pbar = tqdm(range(args.nsteps))
        
        imgs = data_dict['train_data']['manip_imgs'][:args.ntrain].float()
        target_label = np.unique(data_dict['train_data']['labels'][:args.ntrain].numpy())
        assert len(target_label) == 1
        
        tgts = ch.tensor([target_label[0]] * len(imgs))
        
        with torch.enable_grad():
            for i in pbar:
                loss = compute_loss(context_model(imgs.cuda()), tgts.cuda())
                optimizer.zero_grad()
                loss.backward()
                pbar.set_description(str(loss))
                optimizer.step()
        loss.detach()
       
    return context_model

def save_checkpoint(args, model):
    if not args.save_checkpoint:
        return
    
    sd_info = {
        'model': model.state_dict()
    }

    ckpt_save_path = os.path.join(args.out_dir if not store else \
                              store.path, filename)
    ch.save(sd_info, ckpt_save_path, pickle_module=dill)
        