import matplotlib.pyplot as plt
import seaborn as sns

def stats_plot(data):
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    sns.lineplot(data=data, x='Epoch', y='Training Loss', ax=ax[0], label='Training Loss')
    sns.lineplot(data=data, x='Epoch', y='Validation Loss', ax=ax[0], label='Validation Loss')
    ax[0].set_title('Training and Validation Loss')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epoch')

    sns.lineplot(data=data, x='Epoch', y='Validation Accuracy', ax=ax[1], color='green', label='Validation Accuracy')
    ax[1].set_title('Validation Accuracy')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_xlabel('Epoch')

    plt.tight_layout()
    plt.show()

def distr_plot(data):
    sns.set_theme(style="whitegrid")

    sns.histplot(data=data, x='Confidence', hue='Prediction', bins=100, element='step', alpha=0.4)
    
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.legend(title='Prediction') 

    plt.show()