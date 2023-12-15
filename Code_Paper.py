import json
import re
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import random
from itertools import combinations
import math
from collections import Counter

random.seed(1)
np.random.seed(1)

# Specify the path to JSON file
json_file_path = '/Downloads/CSfBA individual assignment.json'

with open("TVs-all-merged.json", 'r') as file:
    data = json.load(file)


def extract_top10kvp():
    # Extract all keys in the featuresMap of each product
    all_keys = [key for item_list in data.values() for item in item_list for key in item["featuresMap"].keys()]
    # Count the number of occurrences of each key in the dataset
    key_counts = Counter(all_keys)
    # Find the top 10 most occurring keys
    top_keys = key_counts.most_common(10)
    print("Top 10 most occurring keys:")
    for key, count in top_keys:
        print(f"{key}: {count} occurrences")

    return top_keys


# Create lists for the modelID's, titles and the top 10 most occurring key-value pairs in the dataset
model_ids = []
titles = []
brands = []
max_resolutions = []
aspect_ratio = []
upc = []
v_chip = []
screen_size_diag = []
usb_port = []
tv_type = []
vertical_resolution = []
screen_size_class = []

# Iterates over the data which is a dictionary
for model_id, model_list in data.items():
    # Iterate over tv's which are in the same list (duplicates)
    for model in model_list:
        # extract information from each model
        model_ids.append(model_id)
        titles.append(model['title'])
        brands.append(model['featuresMap'].get('Brand', None))
        max_resolutions.append(model['featuresMap'].get('Maximum Resolution', None))
        aspect_ratio.append(model['featuresMap'].get('Aspect Ratio', None))
        upc.append(model['featuresMap'].get('UPC', None))
        v_chip.append(model['featuresMap'].get('V-Chip', None))
        screen_size_diag.append(model['featuresMap'].get('Screen Size (Measured Diagonally)', None))
        usb_port.append(model['featuresMap'].get('USB Port', None))
        tv_type.append(model['featuresMap'].get('TV Type', None))
        vertical_resolution.append(model['featuresMap'].get('Vertical Resolution', None))
        screen_size_class.append(model['featuresMap'].get('Screen Size Class', None))

# create a DataFrame df with the modelIDs, title and the 10 attributes
df = pd.DataFrame({
    'model_id': model_ids,
    'title': titles,
    'brand': brands,
    'maximum resolution': max_resolutions,
    'aspect ratio': aspect_ratio,
    'UPC': upc,
    'v chip': v_chip,
    'screen size diag': screen_size_diag,
    'usb port': usb_port,
    'tv type': tv_type,
    'vertical resolution': vertical_resolution,
    'screen size class': screen_size_class
})


# Function to clean the title
def clean_title(title):
    # Define the different variations of 'inch'
    inch_patterns = ['Inch', 'inches', '"', '-inch', ' inch', 'inch']

    # Define the different variations of 'hertz'
    hertz_patterns = ['Hertz', 'hertz', 'Hz', 'HZ', ' hz', '-hz', 'hz']

    # Replace inch patterns with 'inch'
    for pattern in inch_patterns:
        title = re.sub(pattern, 'inch', title)

    # Replace hertz patterns with 'hz'
    for pattern in hertz_patterns:
        title = re.sub(pattern, 'hz', title)

    # Replace upper-case characters with lower-case characters
    title = title.lower()

    # Remove spaces and non-alphanumeric tokens in front of the units
    title = re.sub(r'\W+', ' ', title)

    return title


# Apply the cleaning function to the 'title' column
df['cleaned_title'] = df['title'].apply(clean_title)


# Returns list with model words of 1 TV title
def extract_model_words(row):
    cleaned_title = row['cleaned_title']
    # regex = r'\b(?:[a-zA-Z]+\d+|\d+[a-zA-Z]+)\b'
    regex = '[a-zA-Z]+[0-9]+[a-zA-Z0-9]*|[a-zA-Z0-9]*[0-9]+[a-zA-Z]+|[0-9]+[.][0-9]+[a-zA-Z]*'

    model_words = set(re.findall(regex, cleaned_title))
    # List of columns to include
    attributes_to_include = ['brand', 'maximum resolution', 'aspect ratio', 'UPC', 'v chip',
                             'screen size diag', 'usb port', 'tv type', 'vertical resolution', 'screen size class']
    # Append the values of the kvp's to the set of model words (only if the value of the kvp is not 'None')
    model_words.update(value for col, value in row[attributes_to_include].items() if value)
    mw_and_kvp = model_words

    return mw_and_kvp  # From now on, model words is defined as the model words from the title and the values from the kvp's


# Function which extracts all model words and values of the kvp's from the data set
def extract_all_model_words(dataframe):
    # set with ALL model words of all TVS
    all_model_words = set()

    # Loop through each row in the DataFrame
    for index, row in dataframe.iterrows():
        # Extract model words from the title
        model_words = extract_model_words(row)
        # Update the set of distinct model words
        all_model_words.update(model_words)

    return all_model_words


# Function which adds a column to the dataset containing the model words of each TV
def add_model_words_column(dataframe):
    # Initialize an empty list to store the sets of model words
    model_words_list = []

    # Loop through each row in the DataFrame and extract the model words from that product
    for index, row in dataframe.iterrows():
        model_words = extract_model_words(row)
        model_words_list.append(set(model_words))

    # Add a new column 'model_words' to the DataFrame
    dataframe['model_words'] = model_words_list


def obtain_binary_vector(product, all_model_words):
    binary_vector = {mw: 1 if mw in extract_model_words(product) else 0 for mw in all_model_words}
    return binary_vector


def create_binary_matrix(dataframe, all_model_words):
    binary_matrix = []

    # Loop through each row in the DataFrame and return the binary matrix
    for index, row in dataframe.iterrows():
        binary_vector = obtain_binary_vector(row, all_model_words)
        binary_matrix.append(list(binary_vector.values()))
    transposed_matrix = np.array(binary_matrix).T
    return transposed_matrix


# ###################################################### MINHASHING ###################################################

# Function performing the minhashing on the binary matrix and returns the signature matrix
def minhash(binary_matrix, num_permutations):
    num_products = binary_matrix.shape[1]
    num_model_words = binary_matrix.shape[0]

    np.random.seed(1)
    # Initialize the signature matrix
    signature_matrix = np.zeros((num_permutations, num_products), dtype=int)

    # Generates random permutations, 'n_permutations' in total
    permutations = [np.random.permutation(num_model_words) for _ in range(num_permutations)]

    # Performs minhashing for each permutation and for each product
    for i in range(num_permutations):
        permutation = permutations[i]

        for j in range(num_products):
            # Find the index of the first '1' in the permuted binary vector
            index_first_one = np.argmax(binary_matrix[permutation, j] == 1)
            signature_matrix[i, j] = index_first_one

    return signature_matrix


# ###################################################### LSH ###########################################################


def locality_sensitive_hashing(signature_matrix, bands, rows):
    num_products = signature_matrix.shape[1]

    # Initialize a hash table (dictionary) to store the candidate pairs
    hash_table = defaultdict(list)

    # divide the signature matrix into bands and define the start and end of a band
    for band in range(bands):
        band_start = band * rows
        band_end = (band + 1) * rows

        # Hash each product in the band to a bucket and store the hashed_value and the modelID as kvp in the hash_table
        for product_id in range(num_products):
            hashed_value = hash(tuple(signature_matrix[band_start:band_end, product_id]))
            hash_table[hashed_value].append(product_id)

    # Identify the candidate pairs, use set to store pairs such that only unique pairs are considered
    candidate_pairs = set()
    # Bucket is the key, products_buckets are the value(s)
    for bucket, products_in_bucket in hash_table.items():
        # Only buckets with more than 1 tv's are considered
        if len(products_in_bucket) > 1:
            # If there are at least two products in the same bucket, consider them as candidate pairs,
            # 'if p1 < p2' ensures that a candidate pair is only added once
            candidate_pairs.update([(p1, p2) for p1 in products_in_bucket for p2 in products_in_bucket if p1 < p2])

    return list(candidate_pairs)


# ################################################# Classification #################################################

# Function which computes the Jaccard Similarity between 2 sets
def jaccard_similarity(set_a, set_b):
    if len(set_a.union(set_b)) == 0 or len(set_a.intersection(set_b)) == 0:
        jaccard_sim = 0
    else:
        jaccard_sim = len(set_a.intersection(set_b)) / len(set_a.union(set_b))
    return jaccard_sim


# Function which finds the duplicates after LSH is performed, the classification with Jaccard Similarity method is used
def classification_to_find_duplicates(dataframe, candidate_pairs, num_tvs, jaccard_threshold=0.8):
    sim_matrix = np.zeros((num_tvs, num_tvs,))  # similarity matrix

    # Set the similarities between the products in the similarity matrix
    for i, j in candidate_pairs:
        set_a = dataframe.loc[i, 'model_words']  # set of mw of 1 element of pair
        set_b = dataframe.loc[j, 'model_words']  # set of mw of the other element of pair
        similarity = jaccard_similarity(set_a, set_b)
        sim_matrix[i, j] = similarity
        sim_matrix[j, i] = similarity  # Since similarity is symmetric

    # Find the duplicates pairs by using the Jaccard similarity.
    duplicates_found = []
    for i in range(sim_matrix.shape[0]):
        for j in range(i + 1, sim_matrix.shape[1]):  # Avoid duplicates and the diagonal
            similarity = sim_matrix[i, j]

            # Candidate pairs which have a similarity higher than the Jaccard threshold will be classified as duplicates
            if similarity > jaccard_threshold:
                duplicates_found.append((i, j))

    return duplicates_found


# Function which returns the performance metrics
def obtain_performance_metrics(candidate_pairs, duplicates_found, dataframe):
    # candidates obtained from lsh+classification
    n_true_positives = 0
    n_false_positives = 0

    # candidates obtained from lsh, true positives and false positives
    n_lsh_tp = 0
    n_lsh_fp = 0

    n_comparisons = len(candidate_pairs)

    # compute the number of real duplicates
    n_true_duplicates = 0
    unique_model_ids = dataframe['model_id'].unique()
    for model_ID in unique_model_ids:
        indices_duplicates = np.where(dataframe['model_id'] == model_ID)[0]
        duplicates_combinations = len(list(combinations(indices_duplicates, 2)))
        n_true_duplicates += duplicates_combinations

    for k, l in candidate_pairs:  # Pairs obtained from LSH only (what if we would have a perfect classifier)
        model_id_k = dataframe.at[k, 'model_id']
        model_id_l = dataframe.at[l, 'model_id']
        if model_id_l == model_id_k:
            n_lsh_tp += 1
        else:
            n_lsh_fp += 1

    for i, j in duplicates_found:  # Pairs obtained from classification with lsh as preselection
        model_id_i = dataframe.at[i, 'model_id']
        model_id_j = dataframe.at[j, 'model_id']
        if model_id_i == model_id_j:
            n_true_positives += 1
        else:
            n_false_positives += 1

    precision = n_true_positives / (n_false_positives + n_true_positives)
    recall = n_true_positives / n_true_duplicates
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = (2 * (precision * recall)) / (precision + recall)

    pair_quality = n_lsh_tp / n_comparisons
    pair_completeness = n_lsh_tp / n_true_duplicates

    if (pair_quality + pair_completeness) == 0:
        f1_star = 0
    else:
        f1_star = (2 * pair_quality * pair_completeness) / (pair_quality + pair_completeness)

    metrics = [precision, recall, f1, f1_star, pair_quality, pair_completeness]
    return metrics


# List to store the measures for all bootstraps (test)
all_pc_measures = []
all_pq_measures = []
all_fraction_of_comparisons = []
all_f1_measures = []
all_f1star_measures = []

# List to store the measures for all bootstraps (train)
all_pc_measures_train = []
all_pq_measures_train = []
all_fraction_of_comparisons_train = []
all_f1_measures_train = []
all_f1star_measures_train = []


# Function for the bootstrapping and running of the algorithms
def bootstrapping(df):
    n_bootstraps = 5

    for bootstrap in range(n_bootstraps):

        np.random.seed(1 + bootstrap)

        # List to store the measures for one bootstrap (test)
        f1_measures = []
        f1star_measures = []
        pq_measures = []
        pc_measures = []
        fraction_of_comparisons = []
        t_thresholds = []

        # List to store the measures for one bootstrap (train)
        f1_measures_train = []
        f1star_measures_train = []
        pq_measures_train = []
        pc_measures_train = []
        fraction_of_comparisons_train = []

        num_tvs = len(df)

        # Create a bootstrap sample for training (63% of the original data)
        train_indices = np.random.choice(range(num_tvs), size=int(0.63 * num_tvs), replace=True)
        train_df = df.iloc[train_indices].reset_index(drop=True)

        num_tvs_train = len(train_df)
        add_model_words_column(train_df)
        all_model_words_train = extract_all_model_words(train_df)
        num_permutations_train = 600  # Approximately 50% of the total number of model words
        binary_matrix_train = create_binary_matrix(train_df, all_model_words_train)
        signature_matrix_train = minhash(binary_matrix_train, num_permutations_train)

        # Not all possible combinations are considered due the computation time
        possible_b_values = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 50, 60, 150, 600]
        possible_r_values = [600, 300, 200, 150, 120, 100, 75, 60, 50, 40, 30, 20, 12, 10, 4, 1]

        best_f1 = -1
        optimal_t = 0
        optimal_n_bands = 0
        optimal_n_rows = 0

        total_possible_comparisons = math.comb(len(train_df), 2)

        # Try different values of b and r
        for b_candidate, r_candidate in zip(possible_b_values, possible_r_values):

            # Perform LSH with the bands, and rows on the training set, perform classification and obtain metrics
            candidate_pairs_train = locality_sensitive_hashing(signature_matrix_train, b_candidate, r_candidate)
            duplicates_found_train = classification_to_find_duplicates(train_df, candidate_pairs_train,
                                                                       num_tvs_train)
            performance_metrics = obtain_performance_metrics(candidate_pairs_train, duplicates_found_train,
                                                             train_df)

            comparison_fraction = len(candidate_pairs_train) / total_possible_comparisons
            f1 = performance_metrics[2]
            f1_star = performance_metrics[3]
            pair_quality = performance_metrics[4]
            pair_completeness = performance_metrics[5]
            t_treshold = (1 / b_candidate) ** (1 / r_candidate)

            pc_measures_train.append(pair_completeness)
            pq_measures_train.append(pair_quality)
            f1_measures_train.append(f1)
            f1star_measures_train.append(f1_star)
            fraction_of_comparisons_train.append(comparison_fraction)

            print(f" For {b_candidate} bands and {r_candidate} rows (TRAINING):")
            print(f"PairQuality: {pair_quality}")
            print(f"PairComplete: {pair_completeness}")
            print(f"f1: {f1}")
            print(f"f1_star: {f1_star}")
            print(f"t_threshold: {t_treshold}")
            print()

            # Update optimal values if a better F1-measure is found
            if f1 > best_f1:
                best_f1 = f1
                optimal_n_bands = b_candidate
                optimal_n_rows = r_candidate
                optimal_t = (1 / optimal_n_bands) ** (1 / optimal_n_rows)

        print(f"Optimal Parameters for Bootstrap {bootstrap + 1} (Training):")
        print(f"  Threshold (t): {optimal_t}")
        print(f"  Bands: {optimal_n_bands}")
        print(f"  Rows: {optimal_n_rows}")
        print(f"  best f1: {best_f1}")

        all_pq_measures_train.append(pq_measures_train)
        all_pc_measures_train.append(pc_measures_train)
        all_f1_measures_train.append(f1_measures_train)
        all_f1star_measures_train.append(f1star_measures_train)
        all_fraction_of_comparisons_train.append(fraction_of_comparisons_train)

        # Perform LSH with the optimal threshold, bands, and rows on the test set
        # Create a test sample (37% of the original data)
        test_indices = np.setdiff1d(range(num_tvs), train_indices)
        test_df = df.iloc[test_indices].reset_index(drop=True)

        num_tvs_test = len(test_df)

        add_model_words_column(test_df)
        all_model_words_test = extract_all_model_words(test_df)
        num_permutations_test = 600
        binary_matrix_test = create_binary_matrix(test_df, all_model_words_test)
        signature_matrix_test = minhash(binary_matrix_test, num_permutations_test)

        total_possible_comparisons_test = math.comb(len(test_df), 2)

        for b_candidate_test, r_candidate_test in zip(possible_b_values, possible_r_values):
            candidate_pairs_test = locality_sensitive_hashing(signature_matrix_test, b_candidate_test, r_candidate_test)
            duplicates_found_test = classification_to_find_duplicates(test_df, candidate_pairs_test, num_tvs_test)
            performance_metrics_test = obtain_performance_metrics(candidate_pairs_test, duplicates_found_test,
                                                                  test_df)

            f1_test = performance_metrics_test[2]
            f1_star_test = performance_metrics_test[3]
            pair_quality_test = performance_metrics_test[4]
            pair_completeness_test = performance_metrics_test[5]
            t_treshold_test = (1 / b_candidate_test) ** (1 / r_candidate_test)
            comparison_fraction = len(candidate_pairs_test) / total_possible_comparisons_test

            # Save results for current b and r combination
            f1_measures.append(f1_test)
            f1star_measures.append(f1_star_test)
            pc_measures.append(pair_completeness_test)
            pq_measures.append(pair_quality_test)
            t_thresholds.append(t_treshold_test)
            fraction_of_comparisons.append(comparison_fraction)

        # Save all results for the current bootstrap
        all_pq_measures.append(pq_measures)
        all_pc_measures.append(pc_measures)
        all_f1_measures.append(f1_measures)
        all_f1star_measures.append(f1star_measures)
        all_fraction_of_comparisons.append(fraction_of_comparisons)


bootstrapping(df)

# Calculate averages over all bootstraps (test)
avg_pq_measures = np.mean(all_pq_measures, axis=0)
avg_fraction_of_comparisons = np.mean(all_fraction_of_comparisons, axis=0)
avg_pc_measures = np.mean(all_pc_measures, axis=0)
avg_f1_measures = np.mean(all_f1_measures, axis=0)
avg_f1star_measures = np.mean(all_f1star_measures, axis=0)

# Calculate averages over all bootstraps (train)
avg_pq_measures_train = np.mean(all_pq_measures_train, axis=0)
avg_pc_measures_train = np.mean(all_pc_measures_train, axis=0)
avg_f1_measures_train = np.mean(all_f1_measures_train, axis=0)
avg_f1star_measures_train = np.mean(all_f1star_measures_train, axis=0)
avg_fraction_of_comparisons_train = np.mean(all_fraction_of_comparisons_train, axis=0)

print(f"Average PQ over all bootstraps: {avg_pq_measures}")
print(f"Average PC over all bootstraps: {avg_pc_measures}")
print(f"Average F1 over all bootstraps: {avg_f1_measures}")
print(f"Average F1* over all bootstraps: {avg_f1star_measures}")
print(f"Average fraction of comparisons over all bootstraps: {avg_fraction_of_comparisons}")
print()
print(f"Average PQ train over all bootstraps: {avg_pq_measures_train}")
print(f"Average PC train over all bootstraps: {avg_pc_measures_train}")
print(f"Average F1 train over all bootstraps: {avg_f1_measures_train}")
print(f"Average F1* train over all bootstraps: {avg_f1star_measures_train}")
print(f"Average fraction of comparisons over all bootstraps: {avg_fraction_of_comparisons_train}")

plt.subplot(2, 4, 1)
plt.plot(avg_fraction_of_comparisons, avg_pq_measures, label='Average Pair Quality (PQ)')
plt.xlabel('Fraction of Comparisons')
plt.ylabel('Pair Quality (PQ)')

plt.subplot(2, 4, 2)
plt.plot(avg_fraction_of_comparisons, avg_pc_measures, label='Average Pair Completeness (PC)')
plt.xlabel('Fraction of Comparisons')
plt.ylabel('Pair Completeness (PC)')

plt.subplot(2, 4, 3)
plt.plot(avg_fraction_of_comparisons, avg_f1_measures, label='F1')
plt.xlabel('Fraction of Comparisons')
plt.ylabel('F1')

plt.subplot(2, 4, 4)
plt.plot(avg_fraction_of_comparisons, avg_f1star_measures, label='F1*')
plt.xlabel('Fraction of Comparisons')
plt.ylabel('F1*')

# Paper only considers the test plots
plt.subplot(2, 4, 5)
plt.plot(avg_fraction_of_comparisons_train, avg_pq_measures_train, label='Avg PQ (Train)')
plt.xlabel('Fraction of Comparisons (Train)')
plt.ylabel('Pair Quality (PQ) (Train)')

plt.subplot(2, 4, 6)
plt.plot(avg_fraction_of_comparisons_train, avg_pc_measures_train, label='Avg PC (Train)')
plt.xlabel('Fraction of Comparisons (Train)')
plt.ylabel('Pair Completeness (PC) (Train)')

plt.subplot(2, 4, 7)
plt.plot(avg_fraction_of_comparisons_train, avg_f1_measures_train, label='F1 (Train)')
plt.xlabel('Fraction of Comparisons (Train)')
plt.ylabel('F1 (Train)')

plt.subplot(2, 4, 8)
plt.plot(avg_fraction_of_comparisons_train, avg_f1star_measures_train, label='F1* (Train)')
plt.xlabel('Fraction of Comparisons (Train)')
plt.ylabel('F1* (Train)')

plt.tight_layout()
plt.show()
