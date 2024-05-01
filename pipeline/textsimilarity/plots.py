import matplotlib.pyplot as plt
import seaborn as sns

def cos_sim_plot(data):
    sns.set_theme(style="white")

    plt.figure(figsize=(5, 4)) 

    ax = sns.histplot(data=data, x='Cosine Similarity Score', bins=80, element='step', color='mediumseagreen')

    plt.xlabel('Cosine Similarity Score', fontsize=10)
    plt.ylabel('Texts', fontsize=10)

    plt.show()

def counts_plot(data):
    sns.set_theme(style="white")

    plt.figure(figsize=(5, 4)) 

    ax = sns.histplot(data=data, x='Count_Over_Threshold', bins=50, element='step', color='mediumseagreen')

    plt.xlabel('Similarity Matches', fontsize=10)
    plt.ylabel('Texts', fontsize=10)

    plt.show()