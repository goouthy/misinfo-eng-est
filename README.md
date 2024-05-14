# Cross-Lingual Misinformation Detection: Aligning English and Estonian Fake Health News

## Institution
University of Tartu 

## Author
Li Merila
Data Science, MSc

## Abstract
This thesis focuses on identifying fake health news in Estonian news articles by leveraging a pre-labelled dataset in another language. The primary objective is to develop a reliable system for generating ground truth labels for fake health news in Estonian, contributing to the broader field of fake news detection. The proposed approach employs a hybrid two-phase methodology involving semantic similarity measurements, manual annotation, classification, and confidence sampling to create a novel fake health news dataset in Estonian.

## Repository Structure

The repository is organized as follows:

eng-dataset/: Collected English articles from previous works.
Sources:
\begin{table}[H]
\centering
\caption{Previously Annotated English Fake News Datasets}
\label{tab:engarticles}
\begin{tabular}{@{}lllll@{}}
\toprule
\textbf{Dataset} & \textbf{Articles} & \textbf{Date Range} & & \
\midrule
Med-MMHL \cite{sun2023med} & 6059 & Jan 2017 -- May 2023\textsuperscript{} & & \
Monant Medical Misinformation Dataset \cite{srba2022monant} & 5680 & Apr 2001 -- Jan 2022 & & \
FNID: Fake News Inference Dataset \cite{sadeghi2022fake} & 2988 & Aug 2007 -- Apr 2020 & & \
ISOT Fake News Dataset \cite{ahmed2018detecting} & 4756 & Apr 2015 -- Feb 2018 & & \
ReCOVery \cite{zhou2020recovery} & 1910 & Jan 2020 -- May 2020 & & \
\midrule
\textbf{Total} & \textbf{21393} & \textbf{Apr 2001 -- May 2023} & & \
\bottomrule
\end{tabular}
\textsuperscript{}Claimed by authors, date variable not included in the dataset.
\end{table}
est-dataset/: Collected Estonian articles and code for web scraping.
pipeline/:
textsimilarity/: Phase-I methodology including similarity calculation and analysis.
classification/: Phase-II methodology including pipeline and classifier comparison notebooks.
models/: Training and prediction files for the compared classifiers.
data/: Contains classifier validation metrics, classifier predictions, gold standard dataset, silver standard dataset, and final dataset.





