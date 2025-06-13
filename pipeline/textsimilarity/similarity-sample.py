import pandas as pd

### Similarity results for every possible combination of articles
results = pd.read_parquet('similarity-results.parquet')
# Disregarding 0 and 1 since these are likely noise
results = results[(results['Cosine Similarity Score'] > 0) & (results['Cosine Similarity Score'] < 1)]


### To find the distribution of the best match between Est text and Eng text
maxscoreidx = results.groupby('EstText Index')['Cosine Similarity Score'].idxmax()
maxscores = results.loc[maxscoreidx].reset_index(drop=True)

maxscores.to_parquet('similarity-distr.parquet', index=False)


### Sampling n=10 similarity comparisons for every bin of similarity score from 0 to 1 by 0.1

# results['Similarity Bin'] = pd.cut(results['Cosine Similarity Score'], 
#                                    bins=np.arange(0, 1.1, 0.1), 
#                                    labels=range(1, 11), 
#                                    include_lowest=True)

# sampled_results = results.groupby('Similarity Bin').apply(
#     lambda x: x.sample(n=min(10, len(x)), random_state=0)
# )

# sampled_results.reset_index(drop=True, inplace=True)

# sampled_results.to_parquet('similarity-sample.parquet', index=False)