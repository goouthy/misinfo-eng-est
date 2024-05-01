import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

def stats_plot(data):
    sns.set_theme(style="white")

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    sns.lineplot(data=data, x='Epoch', y='Training Loss', ax=ax[0], label='Training Loss', color='salmon', linewidth=2.5)
    sns.lineplot(data=data, x='Epoch', y='Validation Loss', ax=ax[0], label='Validation Loss', color='lightblue', linewidth=2.5)
    # ax[0].set_title('Training and Validation Loss')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epoch')

    sns.lineplot(data=data, x='Epoch', y='Validation Accuracy', ax=ax[1], label='Validation Accuracy', color='mediumseagreen', linewidth=2.5)
    # ax[1].set_title('Validation Accuracy')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_xlabel('Epoch')

    plt.tight_layout()
    plt.show()

def distr_plot(data, ontop=1):
    sns.set_theme(style="white")

    plt.figure(figsize=(6, 5)) 

    if ontop == 1:
        hue_order = [1, 0]
    else: 
        hue_order = [0, 1]

    ax = sns.histplot(data=data, x='Confidence', hue='Prediction', bins=100, element='step', 
                      alpha=0.6, palette=['salmon', 'lightblue'], hue_order=hue_order)

    # plt.title('Prediction Confidence')
    plt.xlabel('Confidence', fontsize=10)
    plt.ylabel('Predictions', fontsize=10)

    ax.tick_params(axis='both', labelsize=10)

    legend_handles = [Patch(facecolor='salmon', label='Misinformation'),
                      Patch(facecolor='lightblue', label='Not Misinformation')]
    
    plt.legend(handles=legend_handles, loc='upper left', fontsize=10) # , title='Prediction'

    plt.show()
