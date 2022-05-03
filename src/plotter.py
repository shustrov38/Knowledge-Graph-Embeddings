import torch
import numpy as np
import scipy.sparse as sp
from sklearn.manifold import TSNE  
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from typing import List, Dict, Union, Tuple

from datasets.wordnet.wordnet import wordnet_preprocessed


def reduce_embeddings_dimension(matrix: torch.tensor, random_state: int=0) -> Tuple[List[float], List[float]]:
    vectors = matrix.detach().numpy()

    tsne = TSNE(n_components=2, random_state=random_state, init='pca')
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    
    return x_vals, y_vals


def plot(filename: str, wn: wordnet_preprocessed, x_vals: List[float], y_vals: List[float], \
             region_synsets: List[int], geographical_data: Dict[str, Dict[str, Union[str, List[int]]]]) -> None:
    
    plt.figure(figsize=(18, 18))
    plt.scatter(x_vals, y_vals, c='w')
    
    patches = []
    
    for region_name in geographical_data.keys():
        color = geographical_data[region_name]['color']
        patches.append(mpatches.Patch(label=region_name, color=color))
        for idx in geographical_data[region_name]['idxs']:            
            content = wn.synset2con[wn.idx2synset[region_synsets[idx]]]
            content = content.lstrip('_')
            content = content[:content.rfind('_')]
            content = content[:content.rfind('_')]
            
            plt.text(x_vals[idx]-1.0, y_vals[idx], content, color=color, fontsize=14)
    
    plt.legend(handles=patches, fontsize=16, loc='lower left')
    plt.axis('off')
    
    plt.savefig(f'./pictures/{filename}')
    