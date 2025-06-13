import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
# from matplotlib.font_manager import FontProperties
# import matplotlib.colors as mcolors
from sklearn.metrics import f1_score, confusion_matrix

############################################################################################################################
def stats_plot(epochstats, title, highlight=None, output=None):
    font = 16
    title_size = font
    label_size = font
    tick_size = font - 2

    plt.rcParams['font.family'] = 'Times New Roman'

    plt.figure(figsize=(6, 5))

    loss_ax = plt.gca()
    sns.lineplot(data=epochstats, x='Epoch', y='Training Loss', ax=loss_ax, color='#FF3131', linewidth=2)
    sns.lineplot(data=epochstats, x='Epoch', y='Validation Loss', ax=loss_ax, color='#284F8C', linewidth=2)
    sns.lineplot(data=epochstats, x='Epoch', y='Validation Accuracy', ax=loss_ax, color='#3CB371', linewidth=2)

    loss_ax.set_ylabel('Loss / Accuracy', fontsize=label_size)
    loss_ax.set_xlabel('Epoch', fontsize=label_size)

    loss_ax.tick_params(axis='both', labelsize=tick_size, left=True, bottom=True, width=1, length=3)

    loss_ax.set_ylim(-0.05, 1.05)
    loss_ax.set_xlim(0, 13)
    loss_ax.set_xticks(range(0, 13, 4))
    loss_ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    sns.despine()

    if highlight:
        loss_ax.axvline(x=highlight, color='black', linestyle='--', linewidth=1)

    final_epoch = epochstats['Epoch'].iloc[-1]
    loss_ax.text(final_epoch + 0.2, epochstats['Training Loss'].iloc[-1] - 0.03, 'Training loss', color='#FF3131', fontsize=label_size)
    loss_ax.text(final_epoch + 0.2, epochstats['Validation Loss'].iloc[-1] - 0.03, 'Validation loss', color='#284F8C', fontsize=label_size)
    loss_ax.text(final_epoch + 0.2, epochstats['Validation Accuracy'].iloc[-1] - 0.03, 'Validation accuracy', color='#3CB371', fontsize=label_size)

    loss_ax.set_title(f'{title}', fontsize=title_size)

    for axis in ['top', 'bottom', 'left', 'right']:
        loss_ax.spines[axis].set_linewidth(0.75)

    if output:
        plt.savefig(f'../{output}.svg', format='svg')
        plt.savefig(f'../{output}.png', format='png')

    plt.show()

############################################################################################################################
def distr_plot(data, title, ontop=0, y_adj=0.8, output=None):
    font = 16
    title_size = font
    label_size = font
    tick_size = font - 2
    legend_size = font
    bin_width = 50
    alpha = 0.8

    plt.rcParams['font.family'] = 'Times New Roman'

    plt.figure(figsize=(6, 5))

    ax = plt.gca()

    class_order = [1, 0] if ontop == 1 else [0, 1]
    class_colors = {1: '#FF3131', 0: '#284F8C'} 

    for cls in class_order:
        ax.hist(
            data[data['Prediction'] == cls]['Confidence'],
            bins=bin_width,
            alpha=alpha,
            color=class_colors[cls],
            label='Misinformation' if cls == 1 else 'Not misinformation',
            edgecolor='none'
        )

    ax.set_xlabel('Confidence', fontsize=label_size)
    ax.set_ylabel('Predictions', fontsize=label_size)
    ax.tick_params(axis='both', labelsize=tick_size, labelcolor='black', width=1, length=3)

    counts, _ = np.histogram(data['Confidence'], bins=bin_width)
    max_y_value = counts.max()
    ax.set_ylim(0, max_y_value * y_adj)
    ax.set_xticks([0.5, 0.75, 1])
    ax.set_xticklabels(['0.5', '0.75', '1'], fontsize=tick_size)

    ax.set_title(f'{title}', fontsize=title_size)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)

    legend_handles = [Patch(facecolor='#FF3131', label='Misinformation', alpha=alpha),
                      Patch(facecolor='#284F8C', label='Not misinformation', alpha=alpha)]
    ax.legend(handles=legend_handles, loc='upper left', fontsize=legend_size, frameon=False)

    if output:
        plt.savefig(f'../{output}.svg', format='svg')
        plt.savefig(f'../{output}.png', format='png')

    plt.show()

############################################################################################################################
# def out_distr(data):
#     """
#     Plots the distribution of misinformation and non-misinformation articles across different websites.

#     Parameters:
#     - data (pd.DataFrame): DataFrame containing website information and prediction labels.
#     """
#     font_family = 'Times New Roman'
#     font_size = 16
#     tick_size = 16

#     sns.set_theme(style="white")
#     plt.rc('font', family=font_family)

#     df = data.copy()
#     df['Label'] = df['Label'].map({0: 'Not Misinformation', 1: 'Misinformation'})

#     grouped = df.groupby(['Website', 'Label']).size().reset_index(name='Count')

#     mask = grouped['Count'] < 20
#     small_groups = grouped[mask].groupby('Label', as_index=False).agg({'Count': 'sum'})
#     small_groups['Website'] = 'Other'
#     grouped = pd.concat([grouped[~mask], small_groups])

#     plt.figure(figsize=(9, 5))
#     sns.barplot(data=grouped, x='Website', y='Count', hue='Label', palette=['#FF3131', '#284F8C'])

#     plt.xlabel('Websites', fontsize=font_size, fontfamily=font_family)
#     plt.ylabel('Articles', fontsize=font_size, fontfamily=font_family)

#     labels = [label.get_text().replace('.ee', '').replace('.', ' ').title() for label in plt.gca().get_xticklabels()]
#     plt.xticks(ticks=plt.gca().get_xticks(), labels=labels, rotation=0, fontsize=tick_size, fontfamily=font_family)
#     plt.tick_params(axis='both', labelsize=tick_size, labelcolor='black')

#     plt.legend(title='', fontsize=font_size, frameon=False)

#     ax = plt.gca()
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_linewidth(1)
#     ax.spines['left'].set_linewidth(1)

#     plt.show()

############################################################################################################################
# def f1_score_plot(datasets, titles=None):
#     font_family = 'Times New Roman'
#     label_size = 16
#     tick_size = 16
#     line_width = 1.5  

#     sns.set_theme(style="white")

#     thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999]

#     if titles is None:
#         titles = [f'Dataset {i+1}' for i in range(len(datasets))]

#     colors = list(mcolors.LinearSegmentedColormap.from_list("red_blue", ["red", "blue"])(np.linspace(0, 1, len(datasets))))
    
#     plt.figure(figsize=(12, 7))  

#     max_f1_scores = []

#     for i, data in enumerate(datasets):
#         f1_scores = []
#         for threshold in thresholds:
#             filtered_data = data[data['Confidence'] >= threshold]
#             if len(filtered_data) > 0:
#                 f1 = f1_score(filtered_data['True Label'], filtered_data['Prediction'], zero_division=0)
#             else:
#                 f1 = np.nan
#             f1_scores.append(f1)

#         plt.plot(thresholds, f1_scores, color=colors[i], linewidth=line_width, label=titles[i])

#         max_f1 = round(np.nanmax(f1_scores), 3)
#         max_f1_threshold = thresholds[np.nanargmax(f1_scores)]

#         plt.plot([0.5, max_f1_threshold], [max_f1, max_f1], color=colors[i], linestyle='--', linewidth=1)

#         max_f1_scores.append(max_f1)

#     ax = plt.gca()
#     ax.spines['top'].set_color('white')
#     ax.spines['right'].set_color('white')
#     ax.spines['bottom'].set_color('black')
#     ax.spines['left'].set_color('black')
    
#     plt.yticks(np.arange(0, 1.1, 0.1), fontsize=tick_size, family=font_family)
#     plt.ylabel('F1 Score', fontsize=label_size, fontfamily=font_family)
    
#     plt.xlabel('Confidence Threshold', fontsize=label_size, fontfamily=font_family)
#     plt.xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize=tick_size, fontfamily=font_family)
    
#     plt.grid(False)
    
#     plt.xlim(0.5, 1.0) 

#     plt.subplots_adjust(left=0.15) 

#     font_props = FontProperties(family=font_family, size=tick_size)
#     plt.legend(prop=font_props, frameon=False)

#     plt.title('F1 Score across Confidence Thresholds', fontsize=label_size+2, fontfamily=font_family, fontweight='bold')

#     plt.show()

############################################################################################################################
def f1_score_plot(data, title, output=None):
    plt.rc('font', family='Times New Roman')

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.996, 0.997, 0.998, 0.999]

    f1_scores = []
    prediction_counts = []

    for threshold in thresholds:
        filtered_data = data[data['Confidence'] >= threshold]
        prediction_count = len(filtered_data)
        prediction_counts.append(prediction_count)
        f1 = f1_score(filtered_data['True Label'], filtered_data['Prediction'], zero_division=0) if prediction_count > 0 else np.nan
        f1_scores.append(f1)

    plt.figure(figsize=(6, 5))
    ax = plt.gca()
    ax.plot(thresholds, f1_scores, color='#FF3131', linewidth=3)

    ax.set_xlabel('Confidence threshold', fontsize=16, fontfamily='Times New Roman')
    ax.set_ylabel('F1 score', fontsize=16, fontfamily='Times New Roman')
    ax.tick_params(axis='both', labelsize=14, labelcolor='black', bottom=True, width=1, length=3)

    max_f1 = round(np.nanmax(f1_scores), 3)
    max_f1_threshold = thresholds[np.nanargmax(f1_scores)]

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlim(0.46, 1.04)
    ax.set_ylim(0, 1)

    ax.axvline(x=max_f1_threshold, color='black', linestyle='--', linewidth=0.75, ymin=0, ymax=max_f1)
    xlim = ax.get_xlim()
    x_min_fraction = (0.5 - xlim[0]) / (xlim[1] - xlim[0])
    x_max_fraction = (max_f1_threshold - xlim[0]) / (xlim[1] - xlim[0])
    ax.axhline(y=max_f1, color='black', linestyle='--', linewidth=0.75, xmin=x_min_fraction, xmax=x_max_fraction)

    ax.scatter(max_f1_threshold, max_f1, color='#FF3131', s=50, edgecolor='none', zorder=5)
    ax.text(max_f1_threshold, max_f1 + 0.02, f'{max_f1_threshold}', color='black', fontsize=16, ha='center', fontfamily='Times New Roman', fontweight='bold')

    ax2 = ax.twinx()
    ax2.fill_between(thresholds, prediction_counts, color='#3CB371', alpha=0.3)
    ax2.set_ylabel('Predictions', fontsize=16, fontfamily='Times New Roman')
    ax2.tick_params(axis='both', labelsize=14, labelcolor='black', width=1, length=3)

    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    ax.set_title(title, fontsize=16)

    for axis in ['bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.75)
        ax2.spines[axis].set_linewidth(0.75)

    if output:
        plt.savefig(f'../{output}.svg', format='svg')
        plt.savefig(f'../{output}.png', format='png')

    plt.show()

############################################################################################################################
def conf_mat_plot(datasets, titles):
    """
    Plots confusion matrices for one or more datasets. If only one dataset is provided, a normal plot is used.
    If multiple datasets are provided, subplots are created. The confusion matrix for each dataset is 
    visualized with a heatmap.

    Args:
    datasets (list or pandas.DataFrame): List of datasets or a single dataset. Each dataset should have 
                                         'True Label' and 'Prediction' columns.
    titles (list or str): List of titles corresponding to each dataset or a single title for one dataset.

    Returns:
    None: Displays the confusion matrix plot(s).
    """
    font_family = 'Times New Roman'
    label_size = 14
    tick_size = 14
    title_size = 16

    if isinstance(datasets, pd.DataFrame):
        datasets = [datasets]
        titles = [titles]

    num_datasets = len(titles)

    if num_datasets == 1:
        df = datasets[0]
        title = titles[0]
        cm = confusion_matrix(df["True Label"], df["Prediction"])

        plt.figure(figsize=(3, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['0', '1'], yticklabels=['0', '1'],
                    annot_kws={"size": 14})

        plt.xlabel('Predicted Label', fontsize=label_size, fontfamily=font_family)
        plt.ylabel('True Label', fontsize=label_size, fontfamily=font_family)
        plt.title(f'{title}', fontsize=title_size, fontfamily=font_family)
        plt.tick_params(axis='both', labelsize=tick_size, labelcolor='black')
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)

        plt.tight_layout()
        plt.show()

    else:
        num_rows = (num_datasets + 2) // 3  
        fig, axs = plt.subplots(num_rows, 3, figsize=(18, 6 * num_rows), gridspec_kw={'hspace': 0.5, 'wspace': 0.3})

        axs = axs.flatten()

        for i, (df, title) in enumerate(zip(datasets, titles)):
            cm = confusion_matrix(df["True Label"], df["Prediction"])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['0', '1'], yticklabels=['0', '1'],
                        ax=axs[i], annot_kws={"size": 14})

            axs[i].set_xlabel('Predicted Label', fontsize=label_size, fontfamily=font_family)
            axs[i].set_ylabel('True Label', fontsize=label_size, fontfamily=font_family)
            axs[i].set_title(f'{title}', fontsize=title_size, fontfamily=font_family)
            axs[i].tick_params(axis='both', labelsize=tick_size, labelcolor='black')
            axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=0, ha='right')
            axs[i].set_yticklabels(axs[i].get_yticklabels(), rotation=0)

        num_plots = num_datasets
        for j in range(num_plots, len(axs)):
            fig.delaxes(axs[j])

        plt.show()

############################################################################################################################
def len_plot(data, column, cutoff, xmax, output=None):
    """Generates a histogram of sequence lengths with a cutoff line."""
    plt.rcParams['font.family'] = 'Times New Roman'
    font = 16
    tick_font = font - 2  

    total_count = data.shape[0]
    above_cutoff = data[data[column] > cutoff]
    below_cutoff = data[data[column] <= cutoff]

    count_above = above_cutoff.shape[0]
    count_below = below_cutoff.shape[0]

    percent_above = (count_above / total_count) * 100
    percent_below = (count_below / total_count) * 100

    truncated_lengths = above_cutoff[column] - cutoff
    median_truncated_length = truncated_lengths.median() if not truncated_lengths.empty else 0

    print(f"Count above cutoff ({cutoff}): {count_above} ({percent_above:.1f}%)")
    print(f"Count below or equal to cutoff ({cutoff}): {count_below} ({percent_below:.1f}%)")
    print(f"Median truncated length for articles above cutoff: {median_truncated_length:.2f}")

    plt.figure(figsize=(6, 5))

    data[column].hist(bins=200, edgecolor='none', color='#284F8C')
    plt.axvline(x=cutoff, color='black', linestyle='--', linewidth=1)

    plt.xlabel('Article length (tokens)', fontsize=font)
    plt.ylabel('Articles', fontsize=font)

    plt.xlim([0, xmax])
    plt.grid(False)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(axis='both', labelsize=tick_font, labelcolor='black', width=1, length=3)

    if output:
        plt.savefig(f'../{output}.svg', format='svg')
        plt.savefig(f'../{output}.png', format='png')

    plt.show()


