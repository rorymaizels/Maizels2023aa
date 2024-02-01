from typing import NamedTuple
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

class colors(NamedTuple):
    celltype = {
        'NMP':'#00afff',
        'Mesoderm':'#20c505',
        'Early_Neural':'#db00ff',
        'Neural':'#ff00a7',
        'p3':'#ffab02',
        'V3':'#ffd966',
        'pMN':'#ff0000',
        'MN':'#ff8080',
        'FP':'#a52a2a',
    }
    
    timepoint = {
        'D3': '#0000cc',
        'D3.2': '#0d00ff',
        'D3.4': '#4200ff',
        'D3.6': '#7800ff',
        'D3.8': '#ad18e7',
        'D4': '#e23ac5',
        'D5': '#ff5ca3',
        'D6': '#ff7e81',
        'D7': '#ffa05f',
        'D8': '#ffc23d'
    }
    
    replicate = {
        'r1': '#ff0000', 
        'r2': '#0000ff', 
        'r3': '#008000', 
        'r4': '#ffa500'
    }
    
    cmap_cont1 = 'inferno'
    cont1 = [mcolors.rgb2hex(color) for color in plt.get_cmap(cmap_cont1).colors]
    
    cmap_cont2 = 'viridis'
    cont2 = [mcolors.rgb2hex(color) for color in plt.get_cmap(cmap_cont2).colors]
    
    cmap_cat1 = 'colorblind'
    cat1 = [mcolors.rgb2hex(color) for color in sns.color_palette(cmap_cat1)]

    cmap_cat2 = 'tab20c'
    cat2 = [mcolors.rgb2hex(color) for color in plt.get_cmap(cmap_cat2).colors]
    
colorpalette = colors()