"""
Data Profiler Module
"""

import numpy as np
# Library imports
import pandas as pd
from scipy.stats import entropy, kurtosis, skew
from sklearn.preprocessing import KBinsDiscretizer, scale
from sklearn.random_projection import (SparseRandomProjection,
                                       johnson_lindenstrauss_min_dim)


def generate_basic_data_profile(X: pd.DataFrame) -> str:
    """
    Generate a basic data profile for the DataFrame X.

    Parameters:
        X (pd.DataFrame): The DataFrame to profile.

    Returns:
        str: A string representation of the basic data profile.
    """

    # 1. Get the field names (column names)
    field_names = X.columns

    # 2. Get the data type of each field
    data_types = X.dtypes

    # 3. Get the top 10 rows of the dataset
    top_10_rows = X.head(10)

    # Function to check if a column has exactly two unique values
    def has_exactly_two_uniques(values):
        """
        Checks if a given iterable has exactly two unique values.
        This version is efficient as it stops as soon as a third unique value is found.
        """
        uniques = set()
        for v in values:
            uniques.add(v)
            if len(uniques) > 2:
                return False
        return len(uniques) == 2

    # Apply the function to each column
    binary_fields = X.apply(has_exactly_two_uniques)

    # Create result strings
    result_field_names = "--- Field Names ---\n" + str(field_names)
    result_data_types = "\n" + "="*30 + "\n" + \
        "--- Data Types of Each Field ---\n" + str(data_types)
    result_top_10_rows = "\n" + "="*30 + "\n" + \
        "--- Top 10 Rows of the Dataset ---\n" + str(top_10_rows)
    result_binary_fields = "\n" + "="*30 + "\n" + \
        "--- Checking Each Field for Exactly Two Unique Values ---\n" + \
        str(binary_fields)

    # Final result output
    final_output = (
        result_field_names +
        result_data_types +
        result_top_10_rows +
        result_binary_fields
    )

    # Return the final result
    return final_output


def generate_feature_profile(X: pd.DataFrame, target_column: str = None) -> pd.DataFrame:
    """
    Generate a complete data profile for the DataFrame X, including statistics for the target variable.

    Parameters:
        X (pd.DataFrame): The DataFrame to profile.
        target_column (str): The name of the target column, if any.

    Returns:
        pd.DataFrame: A DataFrame containing the complete data profile.
    """

    profile = {}

    for column in X.columns:
        if column == target_column:
            continue
        col_data = X[column]
        dtype = str(col_data.dtypes)
        col_profile = {}

        if pd.api.types.is_bool_dtype(col_data):
            vc = col_data.value_counts(dropna=False)
            col_profile = {
                'dtype': dtype,
                'value_counts': dict(vc),
                'missing_percent': col_data.isna().mean() * 100,
                'is_constant': vc.nunique() == 1
            }

        elif pd.api.types.is_numeric_dtype(col_data):
            values = col_data.dropna().values
            if len(values) > 0:
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1)
                median_val = np.median(values)
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                iqr = q3 - q1
                outlier_percent = np.sum(
                    (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)) / len(values) * 100
                cv = std_val / mean_val if mean_val != 0 else np.nan
            else:
                mean_val = std_val = median_val = q1 = q3 = iqr = outlier_percent = cv = np.nan

            col_profile = {
                'dtype': dtype,
                'mean': mean_val,
                'std_dev': std_val,
                'median': median_val,
                'q1': q1,
                'q3': q3,
                'skewness': skew(values) if len(values) > 1 else np.nan,
                'kurtosis': kurtosis(values) if len(values) > 1 else np.nan,
                'missing_percent': col_data.isna().mean() * 100,
                'zero_percent': (col_data == 0).mean() * 100,
                'unique_values': col_data.nunique(),
                'outlier_percent': outlier_percent,
                'coefficient_of_variation': cv,
                'is_constant': col_data.nunique() == 1
            }

        elif pd.api.types.is_categorical_dtype(col_data) or pd.api.types.is_object_dtype(col_data):
            vc = col_data.value_counts(dropna=False)
            cardinality = col_data.nunique(dropna=True)
            ent = entropy(vc.values, base=2) if cardinality > 1 else 0
            if pd.api.types.is_string_dtype(col_data):
                str_vals = col_data.dropna().astype(str)
                col_profile['lowercase_consistent'] = (
                    str_vals.str.lower().nunique() == cardinality)
                col_profile['avg_str_length'] = str_vals.str.len().mean()

            col_profile.update({
                'dtype': dtype,
                'value_counts': dict(vc.head(10)),
                'cardinality': cardinality,
                'entropy': ent,
                'missing_percent': col_data.isna().mean() * 100,
                'high_cardinality': cardinality > 50,
                'is_constant': cardinality == 1
            })

        elif pd.api.types.is_datetime64_any_dtype(col_data):
            col_profile = {
                'dtype': dtype,
                'min_timestamp': col_data.min(),
                'max_timestamp': col_data.max(),
                'missing_percent': col_data.isna().mean() * 100,
                'autocorrelation_lag1': col_data.dropna().astype(np.int64).autocorr(lag=1) if col_data.dropna().size > 1 else np.nan
            }

        profile[column] = col_profile

    profile_df = pd.DataFrame(profile).T
    return profile_df


def generate_target_profile(X: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Generate a profile for the target variable in the DataFrame X.

    Parameters:
        X (pd.DataFrame): The DataFrame containing the target variable.
        target_column (str): The name of the target column.

    Returns:
        pd.DataFrame: A DataFrame containing the profile of the target variable.
    """

    col = X[target_column].dropna()
    dtype = X[target_column].dtype

    profile = {}
    profile["dtype"] = str(dtype)
    profile["missing_percent"] = X[target_column].isna().mean() * 100
    profile["constant_value"] = col.nunique() == 1

    if pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
        col = col.astype(float)
        profile["mean"] = col.mean()
        profile["median"] = col.median()
        profile["std_dev"] = col.std()
        profile["q1"] = col.quantile(0.25)
        profile["q3"] = col.quantile(0.75)
        profile["iqr"] = profile["q3"] - profile["q1"]
        profile["skewness"] = skew(col) if len(col) > 1 else np.nan
        profile["kurtosis"] = kurtosis(col) if len(col) > 1 else np.nan
        profile["min"] = col.min()
        profile["max"] = col.max()
        profile["range"] = col.max() - col.min()

        lower = profile["q1"] - 1.5 * profile["iqr"]
        upper = profile["q3"] + 1.5 * profile["iqr"]
        profile["outlier_percent"] = (
            (col < lower) | (col > upper)).mean() * 100

    elif pd.api.types.is_bool_dtype(col):
        value_counts = col.value_counts(dropna=False)
        profile["value_counts"] = dict(value_counts)
        profile["num_classes"] = len(value_counts)
        if len(value_counts) > 0:
            profile["most_frequent_class"] = value_counts.index[0]
            profile["most_frequent_class_percent"] = (
                value_counts.iloc[0] / len(col)) * 100
            profile["class_distribution"] = {
                k: f"{(v/len(col))*100:.2f}%" for k, v in dict(value_counts).items()}
            profile["entropy"] = entropy(
                value_counts.values, base=2) if len(value_counts) > 1 else 0
            profile["class_imbalance_ratio"] = round(
                value_counts.iloc[0] / value_counts.iloc[1], 2) if len(value_counts) > 1 else np.nan

    else:
        value_counts = col.value_counts()
        class_counts = value_counts.to_dict()
        total = len(col)

        profile["num_classes"] = len(value_counts)
        profile["most_frequent_class"] = value_counts.index[0]
        profile["most_frequent_class_percent"] = (
            value_counts.iloc[0] / total) * 100
        profile["class_distribution"] = {
            k: f"{(v/total)*100:.2f}%" for k, v in class_counts.items()}
        profile["entropy"] = entropy(value_counts.values, base=2)
        profile["class_imbalance_ratio"] = round(
            value_counts.iloc[0] / value_counts.iloc[1], 2) if len(value_counts) > 1 else np.nan

    return pd.DataFrame(profile, index=[target_column])


def compute_pcc(X: pd.DataFrame, target_field: str = None, eps: float = 0.1, n_components: int = None, n_projections: int = 3) -> pd.DataFrame:
    """
    Originally called: `average_sketching_pcc`.

    Approximates the Pearson Correlation Coefficient matrix using multiple random projections.
    Averages results across multiple sketching runs for better accuracy.

    Null-handling policy (UPDATED):
    - Does NOT drop rows/columns.
    - Does NOT impute data values.
    - For math only, replaces NaNs with 0 *after* mean-centering, so projections and scaling work.
    - Any NaNs produced during normalization/correlation are set to 0 in the derived matrices.

    Parameters:
    - df: Pandas DataFrame
    - target: Optional target column name
    - eps: Approximation distortion level (0.1 = ~10% error)
    - n_components: Manual projection size (overrides eps if given)
    - n_projections: Number of random projections to average

    Returns:
    - pd.DataFrame: Averaged approximate correlation matrix
    """

    numeric_df = X.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        raise ValueError(
            "Need at least two numerical columns to compute correlation.")

    use_target = target_field in numeric_df.columns

    if use_target:
        feature_df = numeric_df.drop(columns=[target_field])
        target_vec = numeric_df[target_field]
    else:
        feature_df = numeric_df
        target_vec = None

    # --- Null-safe centering for features (keep original df as-is; only the math view is filled) ---
    centered_data = feature_df - feature_df.mean()
    centered_data = centered_data.fillna(0.0)      # ignore nulls in math
    transposed_data = centered_data.T               # shape: (num_features, num_samples)

    # --- Null-safe centering for target (prepare once, used in loop) ---
    if target_vec is not None:
        target_centered = (target_vec - target_vec.mean()).fillna(0.0).values.reshape(1, -1)
    else:
        target_centered = None

    # Estimate projection size if not given
    if n_components is None:
        n_features = int(transposed_data.shape[0])
        n_components = int(johnson_lindenstrauss_min_dim(
            n_features, eps=eps))

    # Initialize accumulator matrix
    n_features = int(transposed_data.shape[0])
    accum_matrix = np.zeros((n_features, n_features), dtype=float)
    accum_target_corr = np.zeros(
        n_features, dtype=float) if target_centered is not None else None

    for _ in range(n_projections):
        transformer = SparseRandomProjection(n_components=n_components)
        projected = transformer.fit_transform(transposed_data)  # (num_features, n_components)

        # Normalize per "row" (i.e., per original feature); guard against constant rows
        normalized = scale(projected, axis=1)                   # may produce NaNs if zero-variance
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

        approx_corr = np.corrcoef(normalized)                   # correlation across rows (features)
        approx_corr = np.nan_to_num(approx_corr, nan=0.0, posinf=0.0, neginf=0.0)
        accum_matrix += approx_corr

        if target_centered is not None:
            # Project the (1 x num_samples) target using the same transformer
            target_projected = transformer.transform(target_centered)  # (1, n_components)
            target_scaled = scale(target_projected, axis=1)            # (1, n_components)
            target_scaled = np.nan_to_num(target_scaled, nan=0.0, posinf=0.0, neginf=0.0).flatten()

            # Feature–target correlation approximation via normalized dot product
            denom = max(len(target_scaled), 1)
            feature_target_corr = np.dot(normalized, target_scaled) / denom
            accum_target_corr += feature_target_corr

    # Average across projections
    avg_corr_matrix = accum_matrix / n_projections
    corr_df = pd.DataFrame(
        avg_corr_matrix, index=feature_df.columns, columns=feature_df.columns)

    if target_centered is not None:
        avg_target_corr = accum_target_corr / n_projections
        # Add target correlations as a new column/row
        corr_df[target_field] = avg_target_corr
        # Build target row: [corr(feature_i, target), ..., 1.0]
        target_row = np.append(avg_target_corr, [1.0])
        corr_df.loc[target_field] = target_row

    # Final safety: ensure no NaNs left in the output correlation matrix
    corr_df = corr_df.astype(float).fillna(0.0).round(4)
    return corr_df


def compute_stable_mutual_info(X: pd.DataFrame, target_column, bin_counts=[10, 20, 30], near_zero_thresh=1e-4) -> pd.DataFrame:
    """
    Compute stable mutual information across multiple binning strategies.

    Parameters:
        X (pd.DataFrame): The DataFrame containing features and target.
        target_column (str): The name of the target column.
        bin_counts (list): List of bin counts to use for discretization.
        near_zero_thresh (float): Threshold to classify mutual information as near-zero.

    Returns:
        pd.DataFrame: DataFrame with features, average MI, standard deviation, stability, and MI flag.
    """

    def compute_histogram_mi(x, y):
        joint = pd.crosstab(x, y)
        joint_prob = joint / len(x)
        x_prob = joint_prob.sum(axis=1)
        y_prob = joint_prob.sum(axis=0)

        mi = 0.0
        for i in joint_prob.index:
            for j in joint_prob.columns:
                p_xy = joint_prob.loc[i, j]
                if p_xy > 0:
                    p_x = x_prob.loc[i]
                    p_y = y_prob.loc[j]
                    mi += p_xy * np.log(p_xy / (p_x * p_y))
        return mi / np.log(2)  # bits

    def compute_mutual_info(df, target_column, n_feature_bins=10, n_target_bins=10, max_cardinality=50, top_k=20):
        df = df.copy()
        target = df[target_column]
        df.drop(columns=[target_column], inplace=True)

        # Bin target if continuous
        if pd.api.types.is_numeric_dtype(target) and target.nunique() > 5:
            target = KBinsDiscretizer(n_bins=n_target_bins, encode='ordinal', strategy='quantile')\
                .fit_transform(target.values.reshape(-1, 1)).astype(int).ravel()

        mi_scores = {}
        for col in df.columns:
            x = df[col]

            # String / Object → Top-K One-hot Encoding
            if pd.api.types.is_string_dtype(x) or pd.api.types.is_object_dtype(x):
                x = x.fillna("Missing")
                if x.nunique() > max_cardinality:
                    top = x.value_counts().nlargest(top_k).index
                    x = x.apply(lambda val: val if val in top else "Other")
                dummies = pd.get_dummies(x, drop_first=False)
                total_mi = sum(compute_histogram_mi(
                    dummies[d], target) for d in dummies.columns)
                mi_scores[col] = total_mi

            # Categorical / Boolean
            elif pd.api.types.is_categorical_dtype(x) or pd.api.types.is_bool_dtype(x) or x.nunique() <= 5:
                if pd.api.types.is_categorical_dtype(x):
                    x = x.astype(str)
                x = x.fillna("Missing")
                mi_scores[col] = compute_histogram_mi(x, target)

            # Numeric
            elif pd.api.types.is_numeric_dtype(x):
                x = x.fillna(x.median())
                x_binned = KBinsDiscretizer(n_bins=n_feature_bins, encode='ordinal', strategy='quantile')\
                    .fit_transform(x.values.reshape(-1, 1)).astype(int).ravel()
                mi_scores[col] = compute_histogram_mi(x_binned, target)
        return mi_scores
    runs = []

    for bins in bin_counts:
        mi = compute_mutual_info(
            X,
            target_column=target_column,
            n_feature_bins=bins,
            n_target_bins=bins
        )
        runs.append(mi)

    all_features = runs[0].keys()
    records = []

    for feature in all_features:
        values = [run[feature] for run in runs]
        avg_mi = np.mean(values)
        std_mi = np.std(values)

        records.append({
            'Feature': feature,
            'Avg_MI': round(avg_mi, 6),
            'Std_MI': round(std_mi, 6),
            'Stability': 'Unstable' if std_mi > 0.01 else 'Stable',
            'MI_Flag': 'Near-Zero' if avg_mi < near_zero_thresh else 'Informative'
        })

    result_df = pd.DataFrame(records).sort_values(by='Avg_MI', ascending=False)
    return result_df


def generate_complete_profile(X: pd.DataFrame, target_column) -> str:
    """
        Generate a complete profile for the given DataFrame and target column.

    Parameters:
        X (pd.DataFrame): The DataFrame to profile.
        target_column (str): The name of the target column.

    Returns:
        str: A string representation of the sample data profile.
    """
    # Fix: Convert float16 to float32 (pandas doesn't support float16 indexes)
    X = X.copy()
    for col in X.columns:
        if X[col].dtype == np.float16:
            X[col] = X[col].astype(np.float32)

    profile_df = generate_feature_profile(X, target_column)
    target_profile_df = generate_target_profile(X, target_column)

    numeric_df = X.select_dtypes(include=[np.number])

    mi_report = compute_stable_mutual_info(
        X, target_column=target_column)

    data_profile_sample = """

    Basic Data Profile of the dataset:

    """ + str(profile_df) + """

    Target Profile of the dataset:

    """ + str(target_profile_df)

    # Pre-check: need two numerical columns for PCC, if not there, skip.
    if numeric_df.shape[1] >= 3:
        corr_avg = compute_pcc(X, target_field=target_column, n_projections=3)
        data_profile_sample += """

    PCC of the dataset (This is calculated using Average Sketching Method with 3 random projections):

    """ + str(corr_avg)

    data_profile_sample += """

    Mutual Information of the dataset (calculated using Stable Mutual Information Method):

    """ + str(mi_report)

    return data_profile_sample
