# Cross-Lingual Misinformation Detection: Aligning English and Estonian Fake Health News

## Institution
University of Tartu 

## Author
Li Merila
Data Science, MSc

## Abstract
This thesis focuses on identifying fake health news in Estonian news articles by leveraging a pre-labelled dataset in another language. The primary objective is to develop a reliable system for generating ground truth labels for fake health news in Estonian, contributing to the broader field of fake news detection. The proposed approach employs a hybrid two-phase methodology involving semantic similarity measurements, manual annotation, classification, and confidence sampling to create a novel fake health news dataset in Estonian.

## Repository Structure

- **eng-dataset/**: Contains collected English articles from previously annotated fake news datasets, including:
  - [Med-MMHL](https://github.com/styxsys0927/Med-MMHL)
  - [Monant Medical Misinformation Dataset](https://github.com/kinit-sk/medical-misinformation-dataset)
  - [FNID: Fake News Inference Dataset](https://ieee-dataport.org/open-access/fnid-fake-news-inference-dataset)
  - [ISOT Fake News Dataset](https://onlineacademiccommunity.uvic.ca/isot/?utm_medium=redirect&utm_source=%2Fdatasets%2Ffake-news%2Findex.php&utm_campaign=redirect-usage)
  - [ReCOVery](https://github.com/apurvamulay/ReCOVery)

- **est-dataset/**: Contains collected Estonian articles and the code used for web scraping these articles.

- **pipeline/**:
  - **textsimilarity/**: Implements Phase-I of the methodology, which includes the calculation of semantic similarity and similarity analysis between English and Estonian articles.
  - **classification/**: Implements Phase-II of the methodology, which includes pipeline and classifier comparison notebooks.
    - **models/**: Contains training and prediction files for various classifiers.
    - **data/**: Contains classifier validation metrics, classifier predictions, gold standard dataset, silver standard dataset, and final dataset.

## Contact

For any questions or further information, please contact Li Merila at [li.merila@ut.ee].





