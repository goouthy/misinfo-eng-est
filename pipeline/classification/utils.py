from urllib.parse import urlparse

import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.metrics import confusion_matrix, f1_score, precision_score, accuracy_score, recall_score

# Includes: compute_metrics, test_predictions, extract_website_name, distr, print_values, flabel, conf_mat, get_error_counts, run_test

############################################################################################################################
def compute_metrics(data, threshold, pred_col='Prediction', true_col='True Label', conf_col='Confidence', confidence=0.95):
    if (conf_col in data.columns):
        filtered_data = data[data[conf_col] >= threshold]
    else:
        filtered_data = data
    
    n_samples = len(filtered_data)
    
    if n_samples > 0:
        f1 = f1_score(filtered_data[true_col], filtered_data[pred_col], pos_label=1)
        precision = precision_score(filtered_data[true_col], filtered_data[pred_col], pos_label=1)
        recall = recall_score(filtered_data[true_col], filtered_data[pred_col], pos_label=1)
        accuracy = accuracy_score(filtered_data[true_col], filtered_data[pred_col])
        
        z_value = scipy.stats.norm.ppf((1 + confidence) / 2.0)
        f1_ci_length = z_value * np.sqrt((f1 * (1 - f1)) / n_samples)
        precision_ci_length = z_value * np.sqrt((precision * (1 - precision)) / n_samples)
        recall_ci_length = z_value * np.sqrt((recall * (1 - recall)) / n_samples)
        accuracy_ci_length = z_value * np.sqrt((accuracy * (1 - accuracy)) / n_samples)
        
        if threshold > 0:
            print(f"Threshold {threshold}:")
        print(f"Sample Size: {n_samples}/{len(data)}")
        print(f"F1 Score: {f1:.4f} ± {f1_ci_length:.4f}")
        print(f"Precision: {precision:.4f} ± {precision_ci_length:.4f}")
        print(f"Recall: {recall:.4f} ± {recall_ci_length:.4f}")
        print(f"Accuracy: {accuracy:.4f} ± {accuracy_ci_length:.4f}\n")
    else:
        print("No samples meet the confidence threshold.")

############################################################################################################################
def test_predictions(data, threshold):
    print(f'Threshold {threshold}:')
    testpreds = data[data['Confidence'] > threshold]
    
    if testpreds.empty:
        print('No predictions exceed the threshold.')
        return

    predictions_count = testpreds['Prediction'].value_counts().sort_index()

    for pred_class in predictions_count.index:
        print(f'Class {pred_class}: {predictions_count[pred_class]}')

    total_count = predictions_count.sum()
    print(f'Total predictions exceeding threshold: {total_count}')

############################################################################################################################
def extract_website_name(url):
    """
    Extracts the domain name from a given URL.

    Parameters:
    - url (str): The URL string from which to extract the domain name.

    Returns:
    - str: The domain name extracted from the URL.
    """
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    if domain.startswith('www.'):
        domain = domain[4:]
    return domain

############################################################################################################################
def distr(data):
    """
    Displays the distribution of predictions across the different classes in the data.

    Parameters:
    - data (pd.DataFrame): DataFrame containing prediction data.

    Prints:
    - The count of predictions for each class.
    """
    predictions_count = data['Prediction'].value_counts()
    
    count_0 = predictions_count.get(0, 0)
    count_1 = predictions_count.get(1, 0)
    index_0 = predictions_count.index[0]
    index_1 = predictions_count.index[1]

    print('Distribution between classes:')
    print(f'{index_0}: {count_0}')
    print(f'{index_1}: {count_1}')

############################################################################################################################
def print_values(data, column_name):
    """
    Prints the counts and percentages of unique values in a specified column of the DataFrame.

    Parameters:
    - data (pd.DataFrame): DataFrame containing the data.
    - column_name (str): The column name for which to print value counts and percentages.

    Prints:
    - The count and percentage of each unique value in the specified column.
    """
    if column_name in data.columns:
        value_counts = data[column_name].value_counts()
        total_counts = value_counts.sum()

        for label, count in value_counts.items():
            percentage = (count / total_counts) * 100
            print(f'{label}: {count} ({percentage:.1f}%)')
    else:
        print(f"Column '{column_name}' not found in the DataFrame.")

############################################################################################################################
def flabel(inp, join='data/gold-annotation.parquet'):
    inpdf = pd.read_parquet(inp).rename(columns={'Label': 'Prediction'})
    joindf = pd.read_parquet(join).rename(columns={'Label': 'True Label'})
    outp = pd.merge(inpdf, joindf[['EstText Index', 'True Label']], on='EstText Index', how='left')
    return outp

############################################################################################################################
def conf_mat(df):
    cm = confusion_matrix(df["True Label"], df["Prediction"])
    tn, fp, fn, tp = cm.ravel()
    
    print("Confusion Matrix:")
    print(f"{'':<15}{'  Predicted 0':<15}{'  Predicted 1':<15}")
    print(f"{'Actual 0':<15}| {tn:<15}{fp:<15}")
    print(f"{'Actual 1':<15}| {fn:<15}{tp:<15}")

############################################################################################################################
def get_error_counts(true_labels, predictions):
    cm = confusion_matrix(true_labels, predictions)
    
    TN, FP, FN, TP = cm.ravel()
    return TN, FP, FN, TP

############################################################################################################################
def run_test(contingency_table, error_type="Type I"):
    if any(cell < 5 for row in contingency_table for cell in row):
        # Fisher's Exact Test
        _, p_value = fisher_exact(contingency_table)
        print(f"{error_type}: Fisher's Exact Test p-value = {p_value} (no test statistic or df)")
    else:
        # Chi-square test
        chi2_stat, p_value, dof, _ = chi2_contingency(contingency_table)
        print(f"{error_type}: Chi-square statistic = {chi2_stat}, df = {dof}, p-value = {p_value}")