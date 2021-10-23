import torch as ch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import auc 
from sklearn.metrics import roc_curve
from robustness.tools.vis_tools import get_axis


def show_image_row(xlist, ylist=None, fontsize=12, size=(2.5, 2.5), 
                   title=None, tlist=None, filename=None):
    from robustness.tools.vis_tools import get_axis

    H, W = len(xlist), len(xlist[0])
    fig, axarr = plt.subplots(H, W, figsize=(size[0] * W, size[1] * H))
    for w in range(W):
        for h in range(H):
            ax = get_axis(axarr, H, W, h, w)  

            ax.imshow(xlist[h][w].permute(1, 2, 0))
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            if ylist and w == 0: 
                ax.set_ylabel(ylist[h], fontsize=fontsize)
            if tlist:
                ax.set_title(tlist[h][w], fontsize=fontsize)
        
    if title is not None:
        fig.suptitle(title, fontsize=fontsize)
        
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.show()
      
