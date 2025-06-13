# CAPS: A Cross-Lingual Methodology for Detecting Misinformation in Estonian Health News

This repository contains resources and methodologies from the research paper titled "CAPS: A Cross-Lingual Methodology for Detecting Misinformation in Estonian Health News".

## Abstract

Health misinformation undermines public trust and hampers adherence to health guidelines. Automated detection is essential, especially for low-resource languages like Estonian, where dedicated resources are scarce. The proposed Cross-lingual Alignment and Confident Prediction Sampling (CAPS) approach leverages English-labeled datasets to create a comprehensive annotated Estonian misinformation dataset efficiently. The resulting dataset includes 8,795 annotated news articles, significantly advancing misinformation detection capabilities for the Estonian language.


## Repository Structure

- **eng-dataset/**: Contains collected English articles from previously annotated fake news datasets, including:
  - [Med-MMHL](https://github.com/styxsys0927/Med-MMHL)
  - [Monant Medical Misinformation Dataset](https://github.com/kinit-sk/medical-misinformation-dataset)
  - [FNID: Fake News Inference Dataset](https://ieee-dataport.org/open-access/fnid-fake-news-inference-dataset)
  - [ISOT Fake News Dataset](https://onlineacademiccommunity.uvic.ca/isot/?utm_medium=redirect&utm_source=%2Fdatasets%2Ffake-news%2Findex.php&utm_campaign=redirect-usage)
  - [ReCOVery](https://github.com/apurvamulay/ReCOVery)

- **est-dataset/**: Contains collected Estonian articles and the code used for web scraping these articles.

- **pipeline/**:
  - **textsimilarity/**: Phase I methodology: semantic similarity analysis between English and Estonian articles.
  - **classification/**: Phase II methodology: classifier training, validation, and evaluation notebooks.
    - **models/**: Classifier models and scripts.
    - **data/**: Validation metrics, predictions, and finalized datasets.

- **est-fake-news-dataset/**: Final labeled dataset containing 8,795 Estonian health news articles annotated for misinformation.

## Conclusion

CAPS demonstrates effectiveness in detecting misinformation in low-resource languages by combining cross-lingual alignment and confidence sampling. The developed dataset and methodology are adaptable for broader applications beyond health misinformation, providing essential groundwork for future research and practical interventions.

## Contact

For any questions or further information, please contact [Li Tetsmann](li.tetsmann@outlook.com).





