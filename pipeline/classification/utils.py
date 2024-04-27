import numpy as np
import scipy.stats
from sklearn.metrics import f1_score, precision_score, accuracy_score
from urllib.parse import urlparse

def compute_metrics(data, threshold, pred_col='Prediction', true_col='True Label', conf_col='Confidence', confidence=0.95):
    if conf_col in data.columns:
        filtered_data = data[data[conf_col] >= threshold]
    else:
        filtered_data = data
    
    n_samples = len(filtered_data)
    
    if n_samples > 0:
        f1 = f1_score(filtered_data[true_col], filtered_data[pred_col])
        precision = precision_score(filtered_data[true_col], filtered_data[pred_col])
        accuracy = accuracy_score(filtered_data[true_col], filtered_data[pred_col])
        
        z_value = scipy.stats.norm.ppf((1 + confidence) / 2.0)
        f1_ci_length = z_value * np.sqrt((f1 * (1 - f1)) / n_samples)
        precision_ci_length = z_value * np.sqrt((precision * (1 - precision)) / n_samples)
        accuracy_ci_length = z_value * np.sqrt((accuracy * (1 - accuracy)) / n_samples)
        
        if threshold > 0:
            print(f"Threshold {threshold}:")
        print(f"Sample Size: {n_samples}/{len(data)}")
        print(f"F1 Score: {f1:.4f} ± {f1_ci_length:.4f}")
        print(f"Precision: {precision:.4f} ± {precision_ci_length:.4f}")
        print(f"Accuracy: {accuracy:.4f} ± {accuracy_ci_length:.4f}\n")
    else:
        print("No samples meet the confidence threshold.")

def test_predictions(data, threshold):
    print(f'Threshold {threshold}:')
    testpreds = data[data['Confidence'] > threshold]
    predictions_count = testpreds['Prediction'].value_counts()
    
    count_0 = predictions_count.get(0, 0)
    count_1 = predictions_count.get(1, 0)
    index_0 = predictions_count.index[0]
    index_1 = predictions_count.index[1]

    print(f'{index_0}: {count_0}')
    print(f'{index_1}: {count_1}')
    print(f'Adding to the dataset: {count_0 + count_1}')

def extract_website_name(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    if domain.startswith('www.'):
        domain = domain[4:]
    return domain

def distr(data):
    predictions_count = data['Prediction'].value_counts()
    
    count_0 = predictions_count.get(0, 0)
    count_1 = predictions_count.get(1, 0)
    index_0 = predictions_count.index[0]
    index_1 = predictions_count.index[1]

    print('Distribution between classes:')
    print(f'{index_0}: {count_0}')
    print(f'{index_1}: {count_1}')

def print_values(data, column_name):
    if column_name in data.columns:
        value_counts = data[column_name].value_counts()
        total_counts = value_counts.sum()

        for label, count in value_counts.items():
            percentage = (count / total_counts) * 100
            print(f'{label}: {count} ({percentage:.1f}%)')
    else:
        print(f"Column '{column_name}' not found in the DataFrame.")