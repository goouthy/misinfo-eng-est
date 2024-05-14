import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

def cos_sim_plot(data):
    sns.set_theme(style="white")
    font = 16

    plt.figure(figsize=(5, 4)) 

    ax = sns.histplot(data=data, x='Cosine Similarity Score', bins=80, element='step', color='#284F8C')

    plt.xlabel('Cosine Similarity Score', fontsize=font)
    plt.ylabel('Articles', fontsize=font)
    ax.tick_params(axis='both', labelsize=font-2)

    plt.show()

def counts_plot(data):
    sns.set_theme(style="white")
    font = 16

    plt.figure(figsize=(5, 4)) 

    ax = sns.histplot(data=data, x='Count_Over_Threshold', bins=50, element='step', color='#284F8C')

    plt.xlabel('Similarity Matches', fontsize=font)
    plt.ylabel('Articles', fontsize=font)
    ax.tick_params(axis='both', labelsize=font-2)

    plt.show()

def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        wrapped_label = textwrap.fill(text, width=width, break_long_words=break_long_words)
        labels.append(wrapped_label)
    ax.set_xticklabels(labels, rotation=0)

def annt_plot(data, color='#284F8C'):
    font = 20
    similarity_order = ['Similar', 'Similar (Subtopic)', 'Somewhat similar', 'Dissimilar']
    plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[2.5, 1.5, 2]) 

    ax1 = plt.subplot(gs[0])
    sns.countplot(x='Similarity', data=data, color=color, order=similarity_order)
    plt.title('Similarity', fontsize=font)
    wrap_labels(ax1, 12)
    plt.xlabel(' ', fontsize=font)
    plt.ylabel(' ', fontsize=font)
    plt.tick_params(axis='both', labelsize=font-2)

    ax2 = plt.subplot(gs[1])
    sns.countplot(x='Stance', data=data, color=color)
    plt.title('Stance', fontsize=font)
    wrap_labels(ax2, 12)
    plt.xlabel(' ', fontsize=font)
    plt.ylabel(' ', fontsize=font)
    plt.tick_params(axis='both', labelsize=font-2)

    ax3 = plt.subplot(gs[2])
    sns.countplot(x='True Label', data=data, color=color)
    plt.title('Veracity', fontsize=font)
    wrap_labels(ax3, 12)
    plt.xlabel(' ', fontsize=font)
    plt.ylabel(' ', fontsize=font)
    plt.tick_params(axis='both', labelsize=font-2)

    plt.tight_layout()
    plt.show()