import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

def stats_plot(data):
    sns.set_theme(style="white")
    font = 18

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    sns.lineplot(data=data, x='Epoch', y='Training Loss', ax=ax[0], label='Training Loss', color='#FF3131', linewidth=2.5)
    sns.lineplot(data=data, x='Epoch', y='Validation Loss', ax=ax[0], label='Validation Loss', color='#284F8C', linewidth=2.5)
    ax[0].set_ylabel('Loss', fontsize=font)
    ax[0].set_xlabel('Epoch', fontsize=font)

    ax[0].tick_params(axis='both', labelsize=font-2)
    ax[0].legend(fontsize=font) 

    sns.lineplot(data=data, x='Epoch', y='Validation Accuracy', ax=ax[1], label='Validation Accuracy', color='#284F8C', linewidth=2.5)
    ax[1].set_ylabel('Accuracy', fontsize=font)
    ax[1].set_xlabel('Epoch', fontsize=font)
    
    ax[1].tick_params(axis='both', labelsize=font-2)
    ax[1].legend(fontsize=font) 

    plt.tight_layout()
    plt.show()

def distr_plot(data, ontop=1):
    sns.set_theme(style="white")
    font = 16

    plt.figure(figsize=(6, 5))

    if ontop == 1:
        hue_order = [1, 0]
    else:
        hue_order = [0, 1]

    ax = sns.histplot(data=data, x='Confidence', hue='Prediction', bins=100, element='step', 
                      alpha=0.6, palette=['#FF3131', '#284F8C'], hue_order=hue_order)

    plt.xlabel('Confidence', fontsize=font)
    plt.ylabel('Predictions', fontsize=font)

    ax.tick_params(axis='both', labelsize=font-2)

    legend_handles = [Patch(facecolor='#FF3131', label='Misinformation'),
                      Patch(facecolor='#284F8C', label='Not Misinformation')]

    plt.legend(handles=legend_handles, loc='upper left', fontsize=font) 

    plt.show()

def out_distr(data):
    sns.set_theme(style="white")
    font_size = 16

    df = data.copy()
    df['Label'] = df['Label'].map({0: 'Not Misinformation', 1: 'Misinformation'})

    grouped = df.groupby(['Website', 'Label']).size().reset_index(name='Count')

    mask = grouped['Count'] < 20
    small_groups = grouped[mask].groupby('Label', as_index=False).agg({'Count': 'sum'})
    small_groups['Website'] = 'Other'
    grouped = pd.concat([grouped[~mask], small_groups])

    plt.figure(figsize=(9, 5))
    sns.barplot(data=grouped, x='Website', y='Count', hue='Label', palette=['#FF3131', '#284F8C'])

    plt.xlabel('Websites', fontsize=font_size)
    plt.ylabel('Articles', fontsize=font_size)

    labels = [label.get_text().replace('.ee', '').replace('.', ' ').title() for label in plt.gca().get_xticklabels()]
    plt.xticks(ticks=plt.gca().get_xticks(), labels=labels, rotation=0)
    plt.tick_params(axis='both', labelsize=font_size-2)

    plt.legend(title='', fontsize=font_size)
    plt.show()
