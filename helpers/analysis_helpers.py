import os
from pathlib import Path
from tqdm import tqdm
import torch
import torch as ch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set_style('darkgrid')

def eval_accuracy(model, loader, workers=10, batch_size=128, cache_file=None):
    if cache_file and os.path.exists(cache_file):
        df = ch.load(cache_file)
        targets, preds = df['targets'], df['preds']
        
    else:
        targets, preds = [], []
        for _, (im, targ) in tqdm(enumerate(loader), total=len(loader)):
            with ch.no_grad():
                op = model(im.cuda())
                pred = op.argmax(dim=1)
                targets.append(targ)
                preds.append(pred.cpu())

        targets, preds = ch.cat(targets), ch.cat(preds)
        
        if cache_file:
            ch.save({'targets': targets, 'preds': preds}, cache_file)
    
    acc = 100 * ch.mean(targets.eq(preds).float()).item()
    print(f"Accuracy: {acc}")
    
    return targets.cpu(), preds, acc

def get_preds(model, imgs, BS=50, use_tqdm=False):
    top_preds = []
    
    batch_count = int(np.ceil(len(imgs) / BS))
    with ch.no_grad():
        it = range(batch_count)
        if use_tqdm:
            it = tqdm(it)
        for bc in it:
            op = model(imgs[bc*BS:(bc+1)*BS].cuda())
            pred = ch.argmax(op, dim=1)
            top_preds.append(pred.detach().cpu())
            
    return ch.cat(top_preds)
  
def calculate_metric(errors_pre, errors_post, tol=1e-3, base_value=np.nan):
    if errors_pre == 0:
        ratio = base_value
    else:
        ratio = (errors_pre - errors_post) / max(tol, errors_pre)
        ratio = 100 * max(ratio, -1)
    return ratio


def evaluate_rewrite_effect(d, r, random_seed=0, base_value=np.nan):
    
    RENAME_DICT = {'imgs': 'clean',
               'modified_imgs': 'id',
               'modified_imgs_same': 'id',
               'modified_imgs_diff': 'ood'}
    
    target_label = np.unique(d['train_data']['labels'].numpy())
    assert len(target_label) == 1
    target_label = target_label[0]
    test_labels = d['test_data']['labels'].numpy()
    
    rng = np.random.RandomState(random_seed)
    is_correct_clean = r['test_pre_imgs']['preds'] == test_labels
    is_target = test_labels == target_label     
    same_class = is_correct_clean & is_target
    
    for k in ['modified_imgs_same', 'modified_imgs_diff']:
        wrong_pre_manip = (r[f'test_pre_{k}']['preds'] != test_labels)
        wrong_post_manip = (r[f'test_post_{k}']['preds'] != test_labels)
                
        # For target label class
        r[f'errors_{RENAME_DICT[k]}_target'] = np.sum(wrong_pre_manip[same_class])
        r[f'ratio_{RENAME_DICT[k]}_target'] = calculate_metric(np.sum(wrong_pre_manip[same_class]), 
                                                               np.sum(wrong_post_manip[same_class]),
                                                               base_value=base_value)
                
        r[f'errors_{RENAME_DICT[k]}_other'] = np.sum(wrong_pre_manip[~same_class])                                             
        r[f'ratio_{RENAME_DICT[k]}_other'] = calculate_metric(np.sum(wrong_pre_manip[~same_class]), 
                                                              np.sum(wrong_post_manip[~same_class]),
                                                              base_value=base_value)
            
    return r

def get_fig_key(k):
    if 'post_acc' in k: return 'Test set'
    elif '_id_target' in k: return 'Target class + train style'
    elif '_ood_target' in k: return 'Target class + held-out styles'
    elif '_id_other' in k: return 'Other classes + train style'
    elif '_ood_other' in k: return 'Other classes + held-out styles'
    
def plot_improvement_bar(overall_log, args):
    
    colors = sns.color_palette('tab10', 4)
    KEYS = ['ratio_id_target', 'ratio_ood_target', 'ratio_id_other', 'ratio_ood_other']

    res = {'key': [], 'gain': [], 'error': []}

    label = f'{args.mode_rewrite} w/ #Train: {args.ntrain}'

    for ki, k in enumerate(KEYS):                

        res['key'].append(get_fig_key(k).replace('+', "+\n"))
        res['gain'].append(overall_log[k])
        res['error'].append(overall_log[k.replace('ratio', 'errors')])

    fig, axarr = plt.subplots(1, 2, figsize=(20, 4))
    res = pd.DataFrame(res)
    
    sns.barplot(x="key", y="error", data=res, ax=axarr[0], palette=colors)
    sns.barplot(x="key", y="gain", data=res, ax=axarr[1], palette=colors)
    
    for ai, ax in enumerate(axarr):
        labels = [item.get_text() for item in ax.get_xticklabels()]   
        ax.set_xticklabels(labels, rotation=0,  fontsize=12)
        ax.set_xlabel('')
        ax.set_ylabel('% Errors corrected' if ai == 1 else '# Errors to correct', fontsize=20)
        if ai == 1: ax.set_ylim([-20, 110])
    plt.subplots_adjust(wspace=0.2)
    plt.show()

