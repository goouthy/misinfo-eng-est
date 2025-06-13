# import warnings
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
import seaborn as sns
# import textwrap

############################################################################################################################
def cos_sim_plot(data, color='#284F8C', output=None):
    """
    Plot histogram of cosine similarity scores with a specified color.
    Saves the plot as PNG and SVG formats.
    """
    plt.rcParams['font.family'] = 'Times New Roman'
    font = 16
    alpha = 0.9

    plt.figure(figsize=(6, 5))

    ax = sns.histplot(data=data, x='Cosine Similarity Score', bins=100, color=color, edgecolor='none', alpha=alpha, kde=False)

    plt.xlabel('Similarity scores', fontsize=font)
    plt.ylabel('Estonian news articles', fontsize=font)
    ax.tick_params(axis='both', labelsize=font-2)

    x_ticks = np.arange(0.75, 1, 0.05) 
    plt.xticks(ticks=x_ticks)

    sns.despine() 

    if output:
        plt.savefig(f'../{output}.svg', format='svg')
        plt.savefig(f'../{output}.png', format='png')

    plt.show()

############################################################################################################################
def counts_plot(data, color='#FF3131', output=None):
    """
    Plot histogram of counts above threshold with a specified color.
    Saves the plot as PNG and SVG formats.
    """
    plt.rcParams['font.family'] = 'Times New Roman'
    font = 16
    alpha = 0.9

    plt.figure(figsize=(6, 5)) 

    ax = sns.histplot(data=data, x='Count_Over_Threshold', bins=100, color=color, edgecolor='none', alpha=alpha, kde=False)

    plt.xlabel('Matches above 50% similarity', fontsize=font)
    plt.ylabel('Estonian news articles', fontsize=font)
    ax.tick_params(axis='both', labelsize=font-2)

    sns.despine() 

    if output:
        plt.savefig(f'../{output}.svg', format='svg')
        plt.savefig(f'../{output}.png', format='png')

    plt.show()

############################################################################################################################
# def wrap_labels(ax, width, break_long_words=False):
#     labels = []
#     for label in ax.get_xticklabels():
#         text = label.get_text()
#         wrapped_label = textwrap.fill(text, width=width, break_long_words=break_long_words)
#         labels.append(wrapped_label)
#     ax.set_xticklabels(labels, rotation=0)

############################################################################################################################
# def annt_plot(data, color='#284F8C'):
#     font = 20
#     similarity_order = ['Similar', 'Similar (Subtopic)', 'Somewhat similar', 'Dissimilar']
#     plt.figure(figsize=(16, 6))
#     gs = gridspec.GridSpec(1, 3, width_ratios=[2.5, 1.5, 2]) 

#     ax1 = plt.subplot(gs[0])
#     sns.countplot(x='Similarity', data=data, color=color, order=similarity_order)
#     plt.title('Similarity', fontsize=font)
#     wrap_labels(ax1, 12)
#     plt.xlabel(' ', fontsize=font)
#     plt.ylabel(' ', fontsize=font)
#     plt.tick_params(axis='both', labelsize=font-2)

#     ax2 = plt.subplot(gs[1])
#     sns.countplot(x='Stance', data=data, color=color)
#     plt.title('Stance', fontsize=font)
#     wrap_labels(ax2, 12)
#     plt.xlabel(' ', fontsize=font)
#     plt.ylabel(' ', fontsize=font)
#     plt.tick_params(axis='both', labelsize=font-2)

#     ax3 = plt.subplot(gs[2])
#     sns.countplot(x='True Label', data=data, color=color)
#     plt.title('Veracity', fontsize=font)
#     wrap_labels(ax3, 12)
#     plt.xlabel(' ', fontsize=font)
#     plt.ylabel(' ', fontsize=font)
#     plt.tick_params(axis='both', labelsize=font-2)

#     plt.tight_layout()
#     plt.show()