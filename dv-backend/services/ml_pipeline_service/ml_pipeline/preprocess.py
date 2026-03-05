# ml_pipeline/preprocess.py

# Library imports
import hashlib
import numpy as np
import pandas as pd

# Pipeline imports
from ml_pipeline.llm_agents import CodeExecutionAgent, PreprocessingAgent


def handle_null(X: pd.DataFrame, strategy: str = 'mean', constant_value=None, verify: bool = True) -> pd.DataFrame:
    """
    Handle null values in the DataFrame using the specified strategy.

    Parameters:
        X (pd.DataFrame): The input DataFrame with potential null values.
        strategy (str): The strategy to use for handling null values.
        constant_value (any, optional): The constant value to use if strategy is 'constant'.
        verify (bool): Whether to print verification steps.


    Returns:
        pd.DataFrame: The DataFrame with null values handled.
    """

    # Step 1: Initial Check
    initial_null_count = X.isnull().sum().sum()
    if verify:
        print("--- Initial Check ---")
        if initial_null_count == 0:
            print("No null values found.")
            return X
        else:
            print(f"Found {initial_null_count} total null values.")
            print("Null counts per column:")
            print(X.isnull().sum()[X.isnull().sum() > 0])
            print("-" * 25)

    # Apply the chosen strategy (in-place)
    print(f"Applying strategy: '{strategy}'...")
    if strategy == 'drop':
        X.dropna(inplace=True)
    elif strategy in ['mean', 'median']:
        numeric_cols = X.select_dtypes(include=np.number).columns
        if not numeric_cols.empty:
            fill_values = X[numeric_cols].mean(
            ) if strategy == 'mean' else X[numeric_cols].median()
            X.fillna(fill_values, inplace=True)
        else:
            print(
                "Warning: 'mean'/'median' strategy chosen but no numeric columns found.")
    elif strategy == 'mode':
        for col in X.columns:
            if X[col].isnull().any():
                mode_val = X[col].mode()[0]
                X[col].fillna(mode_val, inplace=True)
    elif strategy == 'ffill':
        X.fillna(method='ffill', inplace=True)
    elif strategy == 'bfill':
        X.fillna(method='bfill', inplace=True)
    elif strategy == 'constant':
        if constant_value is None:
            raise ValueError(
                "`constant_value` must be provided for 'constant' strategy.")
        X.fillna(constant_value, inplace=True)
    else:
        raise ValueError(f"Unknown strategy: '{strategy}'.")

    print("Strategy applied successfully.")

    # Step 2: Final Verification
    # Handle Later
    # if verify:
    #     final_null_count = X.isnull().sum().sum()
    #     print("\n--- Final Verification ---")
    #     if final_null_count == 0:
    #         print(
    #             f"Success! All {initial_null_count} null values have been handled.")
    #     else:
    #         print(f"Warning: {final_null_count} null values remain.")
    #         print("Remaining null counts per column:")
    #         print(X.isnull().sum()[X.isnull().sum() > 0])
    #     print("-" * 25)

    return X


def auto_cast_and_encode(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Automatically cast and encode DataFrame columns based on data types and cardinality.

    This function performs:
    - Boolean to uint8 conversion
    - Binary encoding for 2-value object columns
    - One-hot encoding for low cardinality (<= 50 unique values)
    - Hashing trick for high cardinality columns (> 50 values)
    - Safe downcasting for integers
    - PII detection and removal
    - Quasi-constant/constant feature removal

    Parameters:
        df (pd.DataFrame): Input DataFrame to process

    Returns:
        tuple[pd.DataFrame, dict]: Processed DataFrame and encoding report
    """
    report = {}
    df_out = df.copy()

    for col in df_out.columns:
        report[col] = {
            'dtype_before': str(df_out[col].dtype),
            'dtype_after': None,
            'action': None,
            'notes': None
        }

        # Handle boolean columns
        if df_out[col].dtype == 'bool':
            df_out[col] = df_out[col].astype('uint8')
            report[col]['dtype_after'] = 'uint8'
            report[col]['action'] = 'cast_to_uint8'
            continue

        # Handle object columns
        if df_out[col].dtype == 'object':
            unique_values = df_out[col].nunique()

            # Binary encoding for 2 unique values
            if unique_values == 2:
                values = sorted(df_out[col].dropna().unique())
                # Check for common binary patterns
                if set(values).issubset({'0', '1'}) or \
                   set(values).issubset({'yes', 'no'}) or \
                   set(values).issubset({'true', 'false'}) or \
                   set(values).issubset({'Y', 'N'}):
                    mapping = {values[0]: 0, values[1]: 1}
                    df_out[col] = df_out[col].map(mapping).astype('uint8')
                else:
                    df_out[col] = pd.Categorical(df_out[col]).codes.astype('uint8')
                report[col]['dtype_after'] = 'uint8'
                report[col]['action'] = 'binary_encode'

            # One-hot encoding for low cardinality
            elif unique_values <= 50:
                dummies = pd.get_dummies(df_out[col], prefix=col + '__', drop_first=True)
                df_out = pd.concat([df_out.drop(col, axis=1), dummies], axis=1)
                report[col]['dtype_after'] = 'one_hot_encoded'
                report[col]['action'] = 'one_hot_encode'

            # Hashing trick for high cardinality
            else:
                df_out[f'hash_{col}_32'] = df_out[col].apply(
                    lambda x: int(hashlib.md5(str(x).encode()).hexdigest(), 16) % 32
                ).astype('int16')
                df_out = df_out.drop(col, axis=1)
                report[col]['dtype_after'] = 'int16'
                report[col]['action'] = 'hashing_trick'
            continue

        # Safe downcasting for integers
        if df_out[col].dtype in ['int16', 'int32', 'int64']:
            if df_out[col].min() >= 0:
                df_out[col] = pd.to_numeric(df_out[col], downcast='unsigned')
            else:
                df_out[col] = pd.to_numeric(df_out[col], downcast='signed')
            report[col]['dtype_after'] = str(df_out[col].dtype)
            report[col]['action'] = 'safe_cast'
            continue

        # Remove quasi-constant or constant columns
        unique_ratio = df_out[col].nunique() / len(df_out)
        if unique_ratio >= 0.9 or df_out[col].nunique() == 1:
            df_out = df_out.drop(col, axis=1)
            report[col]['action'] = 'drop'
            report[col]['notes'] = 'quasi-constant or constant'
            continue

        # PII detection and removal
        if 'name' in col.lower() or 'email' in col.lower() or 'phone' in col.lower():
            df_out = df_out.drop(col, axis=1)
            report[col]['action'] = 'drop'
            report[col]['notes'] = 'PII'
            continue

        # No action needed
        report[col]['dtype_after'] = str(df_out[col].dtype)
        report[col]['action'] = 'no_action'

    return df_out, report


def generate_preprocessing_insights(data_profile_main: str) -> str:
    """
    Build and run the “expert insights” prompt, then ensemble the results.
    """
    # Block #1 prompt
    prompt = \
        """
        You are an expert in data preprocessing.

        I will provide you with a data profile generated from a dataset using a profiling function.

        Your task:
        - Analyze the data profile and suggest preprocessing steps for each column or issue
        - Focus on key data quality and modeling issues like:
        missing values, outliers, skewness, constant columns, high cardinality, incorrect data types, inconsistent formatting, etc.
        - Use all available profile metrics such as:
        mean, std_dev, min, max, median, q1, q3, skewness, kurtosis, outlier_percent, missing_percent, zero_percent, coefficient_of_variation, mode, entropy, autocorrelation_lag1, high_cardinality, is_constant, lowercase_consistent, Co-rreleation coefficients, Mutual Information

        Instructions:
        - For each column or problem, explain what needs to be addressed and why
        - Suggest the exact preprocessing action (e.g., impute Age with median, one-hot encode Gender, drop constant column)
        - Be clear and precise. Do not generate code, only preprocessing steps
        - You should not one-hot encode the target field. (Strict Rule - But you have to perform encoding of its target. You cannot leave the Object type of target as it is.)
        - You have to tell how to encode the target if its not numeric. No using one-hot encoding. (Mention this in your response)

        Input:
        Data Profile:
        """ + "\n" + str(data_profile_main)

    agent = PreprocessingAgent()
    # Block #2 ensemble of three models
    a = agent.llama(prompt)
    b = agent.quen(prompt)
    c = agent.deepseek(prompt)

    d = """
    I will give you outputs of 3 different LLMs, you have to decide the final output using the Voting Method.

    Do not reduce the content, just combine and give.

    Do not give preamble, Appendices, Footnotes or Supplemental information.
    """ + str(a) + "/n" + str(b) + "/n" + str(c)

    return agent.ensemble(d)


def generate_and_run_preprocessing_code(
    data_profile_main: str,
    insights: str,
    sampled_dataset: pd.DataFrame
) -> None:
    """
    Build and execute the final code-generation prompt.
    """

    prompt_code_pre = \
        """
        You are a Python Code Generator LLM specialized in data preprocessing.

        You will be given:
        1. A pandas DataFrame named `sampled_df` (already loaded)
        2. A detailed data profile generated using a profiling tool
        3. Preprocessing insights provided by a data preprocessing expert

        Your task:
        - Generate clean, executable Python code that performs all preprocessing actions described in the insights
        - Use the data profile as a reference to support or confirm column types, ranges, missingness, etc.
        - Apply preprocessing directly on the existing DataFrame `sampled_df`
        - Use appropriate libraries such as pandas, numpy, and scikit-learn
        - Handle missing values, encoding, scaling, outlier treatment, formatting, or any other task mentioned in the insights
        - Avoid placeholder functions unless absolutely required
        - Do not reload the dataset or redefine `sampled_df`
        - Do not include any output text or comments
        - Output only the Python code

        Inputs:
        -----------------------
        # Data Profile:
        """ + "/n" + str(data_profile_main) + "/n" + """
        -----------------------
        # Preprocessing Insights:
        """ + "/n" + str(insights) + "/n" + """
        -----------------------

        Assume the dataset is already loaded in a pandas DataFrame called `sampled_df`.

        Only output Python code that applies the preprocessing steps to `sampled_df` based on the given insights.
        """

    exec_agent = CodeExecutionAgent(
        custom_namespace={'sampled_df': sampled_dataset})
    exec_agent.ask_and_run(prompt_code_pre)
    sampled_df = exec_agent.namespace.get('sampled_df')

    # Fix: Convert float16 to float32 (pandas doesn't support float16 indexes)
    for col in sampled_df.columns:
        if sampled_df[col].dtype == np.float16:
            sampled_df[col] = sampled_df[col].astype(np.float32)

    return sampled_df


# Example usage within this module:
# X = convert_types_via_llm(X)
# insights = generate_preprocessing_insights(generate_simple_data_profile(X))
# generate_and_run_preprocessing_code(generate_simple_data_profile(X), insights)
# print(X)
