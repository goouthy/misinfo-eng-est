import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

def stats_plot(data):
    font_family = 'Times New Roman'
    label_size = 16
    tick_size = 16
    legend_size = 20

    sns.set_theme(style="white")
    plt.rc('font', family=font_family)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    sns.lineplot(data=data, x='Epoch', y='Training Loss', ax=ax[0], label='Training Loss', color='#FF3131', linewidth=2.5)
    sns.lineplot(data=data, x='Epoch', y='Validation Loss', ax=ax[0], label='Validation Loss', color='#284F8C', linewidth=2.5)
    ax[0].set_ylabel('Loss', fontsize=label_size, fontfamily=font_family)
    ax[0].set_xlabel('Epoch', fontsize=label_size, fontfamily=font_family)
    ax[0].tick_params(axis='both', labelsize=tick_size, labelcolor='black')
    ax[0].legend(fontsize=legend_size, frameon=False)

    sns.lineplot(data=data, x='Epoch', y='Validation Accuracy', ax=ax[1], label='Validation Accuracy', color='#284F8C', linewidth=2.5)
    ax[1].set_ylabel('Accuracy', fontsize=label_size, fontfamily=font_family)
    ax[1].set_xlabel('Epoch', fontsize=label_size, fontfamily=font_family)
    ax[1].tick_params(axis='both', labelsize=tick_size, labelcolor='black')
    ax[1].legend(fontsize=legend_size, frameon=False)

    for axis in ax:
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.spines['bottom'].set_linewidth(1)
        axis.spines['left'].set_linewidth(1)

    plt.tight_layout()
    plt.show()

def distr_plot(data, ontop=1):
    font_family = 'Times New Roman'
    label_size = 16
    tick_size = 16
    legend_size = 20
    bin_width = 50

    sns.set_theme(style="white")

    plt.figure(figsize=(6, 5))

    hue_order = [1, 0] if ontop == 1 else [0, 1]

    ax = sns.histplot(data=data, x='Confidence', hue='Prediction', bins=bin_width, element='step', 
                      alpha=0.6, palette=['#FF3131', '#284F8C'], hue_order=hue_order)

    plt.rc('font', family=font_family)

    plt.xlabel('Confidence', fontsize=label_size, fontfamily=font_family)
    plt.ylabel('Predictions', fontsize=label_size, fontfamily=font_family)
    ax.tick_params(axis='both', labelsize=tick_size, labelcolor='black')
    
    x_ticks = [0.5, 0.75, 1]
    x_labels = ['0.5', '0.75', '1']
    plt.xticks(ticks=x_ticks, labels=x_labels, fontsize=tick_size, fontfamily=font_family)

    num_yticks = 4
    y_ticks = plt.yticks()[0]
    if len(y_ticks) > num_yticks:
        yticks_to_display = y_ticks[::len(y_ticks) // num_yticks]
        plt.yticks(ticks=yticks_to_display, labels=[f'{int(tick)}' for tick in yticks_to_display], fontsize=tick_size, fontfamily=font_family)

    legend_handles = [Patch(facecolor='#FF3131', label='Misinformation'),
                      Patch(facecolor='#284F8C', label='Not Misinformation')]
    plt.legend(handles=legend_handles, loc='upper left', fontsize=legend_size, frameon=False)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)

    plt.show()

def out_distr(data):
    font_family = 'Times New Roman'
    font_size = 16
    tick_size = 16

    sns.set_theme(style="white")
    plt.rc('font', family=font_family)

    df = data.copy()
    df['Label'] = df['Label'].map({0: 'Not Misinformation', 1: 'Misinformation'})

    grouped = df.groupby(['Website', 'Label']).size().reset_index(name='Count')

    mask = grouped['Count'] < 20
    small_groups = grouped[mask].groupby('Label', as_index=False).agg({'Count': 'sum'})
    small_groups['Website'] = 'Other'
    grouped = pd.concat([grouped[~mask], small_groups])

    plt.figure(figsize=(9, 5))
    sns.barplot(data=grouped, x='Website', y='Count', hue='Label', palette=['#FF3131', '#284F8C'])

    plt.xlabel('Websites', fontsize=font_size, fontfamily=font_family)
    plt.ylabel('Articles', fontsize=font_size, fontfamily=font_family)

    labels = [label.get_text().replace('.ee', '').replace('.', ' ').title() for label in plt.gca().get_xticklabels()]
    plt.xticks(ticks=plt.gca().get_xticks(), labels=labels, rotation=0, fontsize=tick_size, fontfamily=font_family)
    plt.tick_params(axis='both', labelsize=tick_size, labelcolor='black')

    plt.legend(title='', fontsize=font_size, frameon=False)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)

    plt.show()
