# ml_pipeline/llm_agents.py

"""
LLM agent wrappers for the ML pipeline.
"""

import os
import re
import sys
import traceback
from typing import Any, Callable, List, Literal, Optional

import pandas as pd
# Library imports
from groq import Groq
# Pipeline imports
from ml_pipeline.config import GROQ_API_KEY, OPENAI_API_KEY
from ml_pipeline.data_profiler import (generate_basic_data_profile,
                                       generate_complete_profile)
from openai import OpenAI
from pydantic import BaseModel, Field

# Groq Client Initialization
groq_client = Groq(api_key=GROQ_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)


class LLMLogger:
    """Simple logger for LLM agents that writes to stdout"""

    def info(self, message: str):
        print(message, flush=True)
        sys.stdout.flush()

    def error(self, message: str):
        print(f"ERROR: {message}", flush=True)
        sys.stderr.flush()


# Global LLM logger
_llm_logger = LLMLogger()


class RepresentativenessCheck(BaseModel):
    """Pydantic model for structured representativeness validation (from nov5 research)."""
    is_representative: bool = Field(
        description="True if the sample dataset is representative of the population; False otherwise."
    )
    reason: str = Field(
        description="A concise explanation of why the sample is or is not representative."
    )


# AutoGluon model codes (valid codes from AutoGluon)

AutoGluonCode = Literal[
    "GBM", "CAT", "XGB", "RF", "XT", "LR", "KNN",
    "REALMLP", "NN_TORCH", "FASTAI", "TABM", "MITRA",
    "FT_TRANSFORMER", "TABPFNV2", "AG_AUTOMM"
]


class ModelSelection(BaseModel):
    """Pydantic model for LLM-based model selection (nov5 research Cell 38)."""
    selected_models: List[AutoGluonCode] = Field(
        description="Reduced set of AutoGluon model codes recommended for this dataset."
    )


class PreprocessingAgent:
    """
    Encapsulates LLM calls to generate data preprocessing plans
    using various models via the Groq API.
    """

    def __init__(self, api_key: str = None):
        """
        Initialize the Groq client.

        Parameters:
            api_key (str, optional): Groq API key. If not provided,
                falls back to GROQ_API_KEY from config or the
                GROQ_API_KEY environment variable.
        """
        key = api_key or GROQ_API_KEY or os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=key)

    def _chat(self, user_prompt: str, system_prompt: str, model: str, **kwargs) -> str:
        """
        Internal helper to send a chat completion request and collect streamed output.

        Parameters:
            user_prompt (str): The content to send as the user message.
            system_prompt (str): The content to send as the system message.
            model (str): The identifier of the LLM model to use.
            **kwargs: Additional parameters forwarded to the Groq API
                (e.g., temperature, max_tokens, top_p, stream).

        Returns:
            str: The concatenated stream of content from the LLM response.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ]
        completion = self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs,
        )
        output = ""
        for chunk in completion:
            output += chunk.choices[0].delta.content or ""
        return output

    def llama(self, user_prompt: str, model: str = "llama-3.3-70b-versatile") -> str:
        """
        Generate a preprocessing plan using the LLaMA model.

        Parameters:
            user_prompt (str): The data profile or instructions to provide as input.
            model (str): The specific Groq LLaMA model to use.

        Returns:
            str: The LLM-generated preprocessing steps.
        """
        system_prompt = (
            "You are a Data Pre Processing Expert. "
            "Give detailed methods and actions to perform pre-processing "
            "of a dataset where its data profile is given."
        )
        return self._chat(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=0.9,
            max_tokens=30000,
            top_p=1,
            stream=True,
        )

    def quen(self, user_prompt: str, model: str = "qwen/qwen3-32b") -> str:
        """
        Generate a preprocessing plan using the Qwen model.

        Parameters:
            user_prompt (str): The data profile or instructions to provide as input.
            model (str): The specific Groq Qwen model to use.

        Returns:
            str: The LLM-generated preprocessing steps.
        """
        system_prompt = (
            "You are a Data Pre Processing Expert. "
            "Give detailed methods and actions to perform pre-processing "
            "of a dataset where its data profile is given."
        )
        return self._chat(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=0.9,
            max_tokens=30000,
            top_p=1,
            stream=True,
        )

    def deepseek(self, user_prompt: str, model: str = "llama-3.3-70b-versatile") -> str:
        """
        Generate a preprocessing plan using a large language model.

        Parameters:
            user_prompt (str): The data profile or instructions to provide as input.
            model (str): The specific Groq model to use.

        Returns:
            str: The LLM-generated preprocessing steps.
        """
        system_prompt = (
            "You are a Data Pre Processing Expert. "
            "Give detailed methods and actions to perform pre-processing "
            "of a dataset where its data profile is given."
        )
        return self._chat(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=0.9,
            max_tokens=30000,
            top_p=1,
            stream=True,
        )

    def ensemble(self, user_prompt: str, model: str = "qwen/qwen3-32b") -> str:
        """
        Generate a preprocessing plan using an ensemble voting strategy.

        Parameters:
            user_prompt (str): The data profile or instructions to provide as input.
            model (str): The specific Groq ensemble model to use.

        Returns:
            str: The LLM-generated preprocessing steps decided by ensemble voting.
        """
        system_prompt = (
            "You are an Agent for Deciding Output based on Voting method. "
            "We are using Ensemble Learning and you are a Voting Ensemble LLM."
        )
        return self._chat(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=0.3,
            max_tokens=30000,
            top_p=1,
            stream=True,
        )


class CodeExecutionAgent:
    """
    Agent to execute code snippets in the global namespace.
    """

    def __init__(self, model: str = "gpt-4.1", custom_namespace: dict = {}):
        """
        Initialize the CodeExecutionAgent with a custom namespace.

        Args:
            custom_namespace (dict): Custom namespace to execute code in.
        """
        self.namespace = globals() | locals() | custom_namespace
        self.model = model
        self.context = []

    def extract_code(self, response: str) -> str:
        """
        Extract code snippets from the LLM response.

        Args:
            response (str): The response from the LLM.

        Returns:
            str: The extracted code snippet.
        """
        match = re.search(r"```(?:python)?(.*?)```", response, re.DOTALL)
        return match.group(1).strip() if match else response.strip()

    def execute_code(self, code: str):
        """
        Execute the provided code in the global namespace.

        Args:
            code (str): The code to execute.
        """

        error_message = None
        try:
            exec(code, self.namespace)
        except Exception:
            error_message = traceback.format_exc()

        return error_message

    def build_prompt(self, user_input: str) -> list[dict]:
        """
        Build the chat prompt including system rules, history, and new user input.

        Parameters:
            user_input (str): The latest user instruction.

        Returns:
            list[dict]: Messages for a chat completion call.
        """
        system_msg = {
            "role": "system",
            "content": (
                "You are a helpful assistant. Only return Python code in markdown blocks "
                "that is continuous to previous code according to prompt. No explanations. "
                "If an error is given, fix it."
            )
        }
        return [system_msg] + self.context + [{"role": "user", "content": user_input}]

    def ask_and_run(self, user_input: str) -> Any:
        """
        Loop: send user input, extract and exec code, retry up to 4 times on error.

        Parameters:
            user_input (str): Instruction for the agent on what code to produce.
        """
        # Initial code generation
        messages = self.build_prompt(user_input)
        response = openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            top_p=1,
            stream=False,
            stop=None,
        ).choices[0].message.content

        code = self.extract_code(response)
        _llm_logger.info(f"🧠 Generated code for execution")
        error = self.execute_code(code)

        # Append history
        self.context.extend([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": response}
        ])

        # Retry on error
        retry_count = 0
        while error and retry_count < 4:
            _llm_logger.info(
                f"⚠️ Code execution error, retrying... (attempt {retry_count + 1}/4)")

            # Build more specific error guidance
            error_guidance = f"""
Fix the code error below. Remember:
1. Use `sampled_df` variable that's already in namespace
2. Target column must be accessed with exact string (not variable name)
3. Must create `metrics` dict with all required keys
4. Must create `trained_model` variable with fitted model
5. Convert numpy types to Python float() before storing in metrics

Error encountered:
{error}

Generate corrected Python code:
"""

            self.context.append({
                "role": "user",
                "content": error_guidance
            })
            response = openai_client.chat.completions.create(
                model=self.model,
                messages=self.context,
                temperature=0.7,
                top_p=1,
                stream=False,
                stop=None,
            ).choices[0].message.content

            code = self.extract_code(response)
            _llm_logger.info("🔁 Retrying with fixed code")
            error = self.execute_code(code)
            self.context.append({"role": "assistant", "content": response})
            retry_count += 1

        if error:
            _llm_logger.error(
                f"❌ Code execution failed after 4 retries: {error}")
        else:
            _llm_logger.info("✅ Code executed successfully")


class BaseLLMAgent:
    """
    Base class for LLM agents.
    """

    def __init__(self, model: str, system_prompt: str):
        self.model = model
        self.system_prompt = system_prompt


class TypeConversionAgent(BaseLLMAgent):
    """
    Agent to handle type conversion using LLM.
    """

    MODEL = 'llama-3.3-70b-versatile'

    SYSTEM_PROMPT = """
    I am working with a dataset and I need help identifying the correct data types for each column.

    I will give you the following for each column:
    - Field name
    - Current data type (as inferred by pandas)
    - Top 10 values
    - A boolean flag indicating whether the column has exactly two unique values (True/False)

    Your task is to:
    - Reason internally about whether the current data type is appropriate
    - Suggest if it needs to be converted to another data type
    - Think about memory efficiency, data semantics, and usability
    - Finally, give me only Python code for converting each column to its correct data type using pandas

    Example 1:
    Field name: Smoker
    Data type: int64
    Top 10 values: [1, 0, 1, 0, 1, 0, 0, 1, 1, 0]
    Boolean flag: True

    Example 2:
    Field name: JoinDate
    Data type: object
    Top 10 values: ["2022-05-01", "2022-05-02", "2022-06-10", "2022-07-14", "2022-08-20", "2022-08-21", "2022-09-01", "2022-09-05", "2022-10-11", "2022-10-20"]
    Boolean flag: False

    Based on the above, give Python code to convert both columns to their correct data types using pandas. Do not include any explanations or comments. Output strictly Python code only.

    Here are the details:
    - The dataset is stored in a Dataframe called: X
    - and use X.loc to do inplace changes
    - Remember you are doing this for Machine Learning Purpose.
    - If Each Field for Exactly Two Unique Values is True its Boolean
    - Convert accordingly such that there are atleast 2 classes if it's a classification problem.
    - If it's two class, convert to binary (0/1).
    - Regression targets should be float/int.
    \n
    """

    def __init__(self):
        super().__init__(model=self.MODEL, system_prompt=self.SYSTEM_PROMPT)

    def _generate_user_prompt(self, X: pd.DataFrame) -> str:
        """
        Generate the user prompt for type conversion based on the DataFrame.
        """

        # Here the user prompt is actually the basic data profile of the DataFrame
        user_prompt = generate_basic_data_profile(X)

        return user_prompt

    def convert_types(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Convert DataFrame column types using LLM.
        """

        # Generate the user prompt based on the DataFrame
        user_prompt = self._generate_user_prompt(X)

        # Generate the LLM response (which is basically the conversion code in markdown format)
        llm_response = groq_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=1,
            max_tokens=8192,
            top_p=1,
            stream=True,
            stop=None,
        )

        # Extract the code markdown from the LLM response
        code_output_md = ""
        for chunk in llm_response:
            part = chunk.choices[0].delta.content or ""
            code_output_md += part  # Append each streamed part

        # Extract the code snippet from the markdown
        code_execution_agent = CodeExecutionAgent(custom_namespace={'X': X})
        code_output = code_execution_agent.extract_code(code_output_md)

        # Execute the code in the global namespace
        code_execution_agent.execute_code(code_output)

        # Return the modified DataFrame
        return X


class DataSamplingAgent(BaseLLMAgent):
    """
    Agent to handle data sampling using LLM.
    """

    MODEL = 'gpt-4o'

    SYSTEM_PROMPT = """
    You are a highly skilled Data Scientist, Statistician and Python Programmer.
    You specialize in intelligent data sampling techniques using Python, statistics, and machine learning.

    You will be given a detailed data profile of a pandas DataFrame named `X`.

    Your job is to:
    1. Understand the relationships using Pearson Correlation and Mutual Information.
    2. Extract a 5 percent representative sample from `X` and store it in `sampled_df` for Machine Learning Training.
    3. Combine appropriate sampling techniques (e.g., clustering, stratification, distribution matching).
    4. Output Python code — with minimal explanations.
    5. Use commonly available Python libraries like pandas, numpy, sklearn.
    6. The code must be robust and should not raise errors as I will be working on computationally heavy tasks.
    7. You should also debug the code based on the error and the previous conversation history.
    8. You will be provided with sample data profile and the population data profile for you to compare and check representativeness.
       If its not representative you should give new python code that can get me a representative sample.
    """

    conversation_history = []

    def __init__(self):
        super().__init__(model=self.MODEL, system_prompt=self.SYSTEM_PROMPT)

        # initialize the conversation history with the system prompt
        self.conversation_history.append(
            {"role": "system", "content": self.system_prompt})

    def _ask_sampling_llm(self, prompt: str) -> str:
        """
        Ask the LLM for sampling code based on the prompt.
        """
        self.conversation_history.append({"role": "user", "content": prompt})

        response = openai_client.chat.completions.create(
            model=self.model,
            messages=self.conversation_history,
            temperature=0.8,
            max_tokens=16384
        )

        reply = response.choices[0].message.content.strip()
        self.conversation_history.append(
            {"role": "assistant", "content": reply})

        return reply

    def sample_data(self, X: pd.DataFrame, target_column: str, sample_percentage: float = None) -> pd.DataFrame:
        """
        Sample DataFrame using LLM with intelligent sampling techniques.

        Args:
            X: Input DataFrame
            target_column: Name of target column
            sample_percentage: Percentage to sample (0-100). If None, no sampling is performed.

        Returns:
            Sampled DataFrame or original if no sampling needed
        """

        # Check if sampling should be performed
        if sample_percentage is None:
            _llm_logger.info(
                f"✅ No sampling percentage specified, using full dataset ({X.shape[0]:,} rows)")
            return X

        # Validate percentage
        if not (0 < sample_percentage <= 100):
            raise ValueError(
                f"sample_percentage must be between 0 and 100, got {sample_percentage}")

        sampling_pct = sample_percentage
        _llm_logger.info(
            f"🎯 Sampling with {sampling_pct}% of data ({X.shape[0]:,} rows)")

        # Generate complete data profile
        complete_data_profile = generate_complete_profile(X, target_column)

        _llm_logger.info("- Generated complete data profile for sampling")

        # Ask the LLM for sampling code (nov5 research format)
        sampling_prompt = f"""
I am giving you a data profile of a dataset that contains various statistical properties and relationships between features.
Your task is to analyze this data profile and generate Python code that performs intelligent sampling techniques
to create a representative sample for machine learning training.

Sample Size: {sampling_pct} percent of the population dataset.

Important: Assign the sampled dataset to a variable named `sampled_df`.

Here is the data profile of the population dataset:
{str(complete_data_profile)}
"""

        _llm_logger.info("- Getting sampling suggestions from LLM")
        sampling_reply = self._ask_sampling_llm(sampling_prompt)

        _llm_logger.info("- Executing sampling code")

        # Extract the code snippet from the reply
        code_execution_agent = CodeExecutionAgent(custom_namespace={'X': X})
        code_output = code_execution_agent.extract_code(sampling_reply)

        # Execute the code in the global namespace
        code_execution_agent.execute_code(code_output)

       # Dependency function for the intelligent sampling workflow runner (nov5 research)
        def _compare_data_profiles(population_data_profile, sample_data_profile):
            """
            Compare population and sample profiles using structured outputs (nov5 approach).
            Returns ('Yes' or 'No', reason) tuple.
            """
            compare_prompt = f"""
You are a data scientist specializing in sampling validation.

Compare the following data profiles and determine if the sample dataset is statistically representative of the population.

Rules:
1) If representative, set is_representative=True.
2) If not representative, set is_representative=False.
3) Always include a concise 'reason' (1–2 sentences) summarizing your decision.
4) Do NOT include extra text, explanations, or code — fill only the structured fields.

--- Population Data Profile ---
{population_data_profile}

--- Sample Data Profile ---
{sample_data_profile}
""".strip()

            self.conversation_history.append(
                {"role": "user", "content": compare_prompt})

            # Use OpenAI structured outputs with Pydantic model (nov5 approach)
            try:
                response = openai_client.beta.chat.completions.parse(
                    model=self.model,
                    messages=self.conversation_history,
                    response_format=RepresentativenessCheck,
                    temperature=0.0,
                    max_tokens=300,
                )

                result = response.choices[0].message.parsed

                self.conversation_history.append({
                    "role": "assistant",
                    "content": f"is_representative={result.is_representative}, reason={result.reason}"
                })

                return ("Yes" if result.is_representative else "No"), result.reason

            except Exception as e:
                _llm_logger.error(
                    f"⚠️  Structured output failed: {e}, using fallback")
                # Fallback to text-based if structured outputs fail
                reply = self._ask_sampling_llm(compare_prompt)
                # Parse text response
                if reply.strip().lower().startswith('yes'):
                    return ("Yes", "Fallback mode - appears representative")
                else:
                    return ("No", reply)

        def _troubleshoot_error(error):
            error_prompt = """

            I am getting an error while running the code you provided. Please debug the code and provide a new version that works correctly.

            Go through the error and also the previous code that you provided and based on that provide me the new code. If some continuation is needed, provide the code accordingly.

            I want the code to be robust and should not raise errors as I will be working on computationally heavy tasks.

            Still the population dataset will be in the dataframe `X` and the sample should be stored in `sampled_df`. This is very important.
            \n
            """ + str(error)

            reply = self._ask_sampling_llm(error_prompt)

            _llm_logger.info("- Received troubleshooting code from LLM")

            return reply

        # Wrapper function to handle the workflow

        def workflow(sampling_reply: str) -> pd.DataFrame:
            extracted_code = code_execution_agent.extract_code(sampling_reply)
            execution_result = code_execution_agent.execute_code(
                extracted_code)

            if execution_result is not None:
                _llm_logger.error(
                    f"ERROR: Error executing sampling code: {execution_result}")
                troubleshoot_reply = _troubleshoot_error(execution_result)
                return workflow(troubleshoot_reply)
            else:
                sampled_df: pd.DataFrame = code_execution_agent.namespace.get(
                    'sampled_df')

                # Check if sampled_df was created by the LLM code
                if sampled_df is None:
                    error_msg = "The sampling code did not create a 'sampled_df' variable. Please ensure your code creates a DataFrame named 'sampled_df' containing the sampled data."
                    _llm_logger.error(f"ERROR: {error_msg}")
                    troubleshoot_reply = _troubleshoot_error(error_msg)
                    return workflow(troubleshoot_reply)

                sampled_df = sampled_df.dropna()
                sample_profile = generate_complete_profile(
                    sampled_df, target_column)

                # Get representativeness check (now returns tuple: decision, reason)
                representativeness, reason = _compare_data_profiles(
                    complete_data_profile, sample_profile)

                if representativeness == "Yes":
                    _llm_logger.info("✅ Sampling completed successfully")
                    _llm_logger.info(f"Reason: {reason}")
                    return sampled_df
                else:
                    _llm_logger.info(
                        f"Sample not representative, retrying sampling...")
                    _llm_logger.info(f"Reason: {reason}")
                    retry_prompt = f"Sample not representative. Please give new code. The reason: {reason}"
                    retry_reply = self._ask_sampling_llm(retry_prompt)
                    return workflow(retry_reply)

        return workflow(sampling_reply)


class ModelTrainingAgent:
    """
    Agent that recommends ML models & hyperparameters via LLM ensemble,
    and generates & executes the corresponding training code.
    """

    SYSTEM_PROMPT = (
        "You are an Industry Leading machine learning expert specialized in "
        "meta-learning and AutoML. You tell what ML Models to apply, what "
        "Hyperparameters to choose just based on Data Profile of a dataset."
    )

    VOTING_PROMPT = (
        "I will give you outputs of 3 different LLMs, you have to decide the final "
        "output using the Voting Method.\n"
        "Just combine and give.\n"
        "Do not give preamble, Appendices, Footnotes or Supplemental information."
    )

    def __init__(self, api_key: str = None, model: str = "llama-3.3-70b-versatile"):
        key = api_key or GROQ_API_KEY or os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=key)
        self.model = model

    def _stream_chat(self, user_prompt: str, system_prompt: str, model: str, **kwargs) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ]
        completion = self.client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )
        output = ""
        for chunk in completion:
            output += chunk.choices[0].delta.content or ""
        return output

    def llama(self, user_prompt: str, model: str = None) -> str:
        return self._stream_chat(
            user_prompt=user_prompt,
            system_prompt=self.SYSTEM_PROMPT,
            model=model or self.model,
            temperature=0.9,
            max_tokens=30000,
            top_p=1,
            stream=True,
            stop=None
        )

    def quen(self, user_prompt: str, model: str = "qwen/qwen3-32b") -> str:
        return self._stream_chat(
            user_prompt=user_prompt,
            system_prompt=self.SYSTEM_PROMPT,
            model=model,
            temperature=0.9,
            max_tokens=30000,
            top_p=1,
            stream=True,
            stop=None
        )

    def deepseek(self, user_prompt: str, model: str = "llama-3.3-70b-versatile") -> str:
        return self._stream_chat(
            user_prompt=user_prompt,
            system_prompt=self.SYSTEM_PROMPT,
            model=model,
            temperature=0.9,
            max_tokens=30000,
            top_p=1,
            stream=True,
            stop=None
        )

    def ensemble(self, user_prompt: str, model: str = "qwen/qwen3-32b") -> str:
        return self._stream_chat(
            user_prompt=user_prompt,
            system_prompt=(
                "You are an Agent for Deciding Output based on Voting method. "
                "We are using Ensemble Learning and you are a Voting Ensemble LLM."
            ),
            model=model,
            temperature=0.3,
            max_tokens=30000,
            top_p=1,
            stream=True,
            stop=None
        )

    def ensemble_call(self, user_prompt: str) -> str:
        """
        Invoke three LLMs, then vote to pick the final recommendation.
        """
        a = self.llama(user_prompt)
        b = self.quen(user_prompt)
        c = self.deepseek(user_prompt)

        vote_input = "\n".join([self.VOTING_PROMPT, a, b, c])
        return self.ensemble(vote_input)

    def recommend_model(
        self,
        original_profile: str,
        final_profile: str,
        target_field: str = "target"
    ) -> str:
        """
        Build and run the model-recommendation prompt.
        """
        prompt = \
            f"""
            You are given **two versions** of the data profile for a dataset:

            1. **Original Data Profile** (before preprocessing): This includes raw statistics like column data types, missing values, skewness,
               and other distributional insights.
            2. **Final Data Profile** (after preprocessing): This reflects the dataset as it will be used for machine learning
               — encoded, imputed, transformed, and ready for modeling.

            Your tasks:
            1. Based on both profiles, identify the most appropriate **target variable** (if not explicitly given).
            2. Determine the **type of ML problem**: Classification or Regression.
            3. Recommend the **Top 2 ML models** most likely to perform well **on the final (preprocessed) dataset**. (Strictly Top 2)

            For each recommended model, include:
            - A short **justification** referencing patterns from both the original and final profile.
            - Suggested **initial hyperparameters** to avoid brute-force tuning.
            - Required **preprocessing steps** already applied or still necessary.
            - Appropriate **evaluation metric** (e.g., Accuracy, F1, RMSE, R²).
            - (Optional) An **estimated performance range** for that metric.
            - Your judgment on whether the model is likely to **underfit**, **overfit**, or **generalize well**, based on indicators like feature correlation, dimensionality, imbalance, and variance.

            ---

            ### Original Data Profile:
            {original_profile}

            ---

            ### Final Data Profile (Post-Preprocessing):
            {final_profile}

            ---

            ### Target Field: `{target_field}` - in the given dataframe.

            ---

            ### Output Format

            - **Target Field**:
            - **Problem Type**: Classification / Regression
            - **Top ML Model 1**:
            - Justification:
            - Recommended Hyperparameters:
            - Preprocessing Suggestions:
            - Evaluation Metric:
            - Expected Score Range (optional):
            - Model Fit Insight: Underfitting / Overfitting / Generalizes Well (with reason)

            You have to respond with Just 1 model and its hyperparameters for which you have high confidence percentage. Analyze properly.
            """
        return self.ensemble_call(prompt)

    def generate_training_code(
        self,
        final_profile: str,
        preprocessing_steps: str,
        model_recommendation: str,
        dataset: pd.DataFrame,
        target_field: str
    ) -> None:
        """
        Build the training-code prompt, then execute the generated code.
        """
        prompt_code_pre = \
            f"""
            You are a Python Code Generator LLM specialized in machine learning model selection and training.

            CRITICAL INFORMATION:
            - Variable `sampled_df` is already defined in the namespace (a preprocessed pandas DataFrame)
            - Target column name: "{target_field}" (use this EXACT string to access the column)
            - Required output variables: `metrics` (dict) and `trained_model` (sklearn/xgboost/lightgbm model object)

            You will be given:
            1. A pandas DataFrame named `sampled_df`, which is already fully preprocessed (encoding, imputation, scaling completed).
            2. A detailed data profile of `sampled_df`, showing column-wise statistics and updated types.
            3. A list of preprocessing steps that have **already been applied** to `sampled_df`.
            4. Model recommendations including:
               - The ML model to use
               - Recommended hyperparameters
               - Evaluation metric
               - Problem type (Classification or Regression)

            YOUR TASK - Generate Python code that:
            
            STEP 1: Extract target variable
            ```python
            y = sampled_df["{target_field}"]
            X = sampled_df.drop(columns=["{target_field}"])
            ```
            
            STEP 2: Train-test split (70/30 ratio)
            ```python
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            ```
            
            STEP 3: Train the recommended model with provided hyperparameters
            
            STEP 4: Evaluate and create metrics dict
            For CLASSIFICATION:
            ```python
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, log_loss
            y_pred = trained_model.predict(X_test)
            y_pred_proba = trained_model.predict_proba(X_test) if hasattr(trained_model, 'predict_proba') else None
            
            metrics = {{
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                'f1_score': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
                'loss': float(log_loss(y_test, y_pred_proba)) if y_pred_proba is not None else None,
                'auc': float(roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')) if y_pred_proba is not None and len(set(y_test)) > 1 else None,
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'sample_size': len(X_test)
            }}
            ```
            
            For REGRESSION:
            ```python
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            y_pred = trained_model.predict(X_test)
            
            metrics = {{
                'mse': float(mean_squared_error(y_test, y_pred)),
                'rmse': float(mean_squared_error(y_test, y_pred, squared=False)),
                'mae': float(mean_absolute_error(y_test, y_pred)),
                'r2': float(r2_score(y_test, y_pred)),
                'loss': float(mean_squared_error(y_test, y_pred)),
                'sample_size': len(X_test)
            }}
            ```

            CONTEXT PROVIDED:
            -----------------------
            # Final Data Profile (Post-Preprocessing):
            {final_profile}

            # Preprocessing Steps Already Applied:
            {preprocessing_steps}

            # Model Recommendation from Meta-Learner:
            {model_recommendation}
            -----------------------

            STRICT REQUIREMENTS:
            1. Do NOT redo preprocessing - it is already done
            2. Do NOT reload or redefine the dataset
            3. Use EXACT column name: "{target_field}" to access target
            4. Output ONLY executable Python code (no explanations)
            5. MUST create `metrics` dict with all required fields
            6. MUST create `trained_model` variable with the fitted model
            7. Use try-except for metrics that might fail (like AUC for single-class)
            8. Convert all numpy types to Python float/int using float() or int()
            9. Print classification report or regression metrics at the end

            Generate the complete, error-free Python code now:
            """
        executor = CodeExecutionAgent(
            custom_namespace={'sampled_df': dataset})
        executor.ask_and_run(prompt_code_pre)
        metrics = executor.namespace.get('metrics')
        trained_model = executor.namespace.get('trained_model')

        # Log what we got
        _llm_logger.info(f"Retrieved metrics type: {type(metrics)}")
        _llm_logger.info(f"Retrieved metrics value: {metrics}")
        _llm_logger.info(f"Retrieved model type: {type(trained_model)}")

        # If metrics is None, try to construct from namespace
        if metrics is None:
            _llm_logger.info(
                "Metrics is None, attempting to extract from namespace")
            metrics = {}
            # Try to get accuracy, precision, recall from namespace
            for key in ['accuracy', 'precision', 'recall', 'f1_score', 'f1', 'test_accuracy']:
                if key in executor.namespace:
                    val = executor.namespace[key]
                    # Ensure we get the actual value, not a function
                    if callable(val):
                        _llm_logger.info(
                            f"Skipping {key} - it's a function, not a value")
                        continue
                    metrics[key] = val

        # Clean metrics dict - remove any function objects
        if isinstance(metrics, dict):
            cleaned_metrics = {}
            for key, val in metrics.items():
                if callable(val):
                    _llm_logger.info(f"Removing function {key} from metrics")
                    continue
                # Convert numpy types to Python native types for JSON serialization
                if hasattr(val, 'item'):
                    try:
                        # Only convert to scalar if array has single element
                        if hasattr(val, 'size') and val.size == 1:
                            cleaned_metrics[key] = val.item()
                        else:
                            # Convert array to list for multi-element arrays
                            cleaned_metrics[key] = val.tolist() if hasattr(
                                val, 'tolist') else val
                    except (ValueError, AttributeError):
                        # If conversion fails, use the value as-is
                        cleaned_metrics[key] = val.tolist() if hasattr(
                            val, 'tolist') else val
                else:
                    cleaned_metrics[key] = val
            metrics = cleaned_metrics

        return metrics, trained_model

    def select_models_with_llm(self, data_profile_str: str) -> List[str]:
        """
        Use LLM with structured outputs to select optimal AutoGluon models (nov5 Cell 38).

        Args:
            data_profile_str: String representation of the data profile

        Returns:
            List of selected AutoGluon model codes
        """
        MODEL_SELECTION_PROMPT = """
You are an expert AutoML system designed to recommend the optimal machine learning models based on the characteristics of a given tabular dataset.

### INPUT:
You will be provided with a **Data Profile** of the dataset, containing various statistical metrics.

### GOAL:
Based on the metrics, recommend **a reduced set of 3-4 machine learning models** that:
- Are likely to achieve high predictive accuracy
- Are resistant to underfitting and overfitting
- Match the characteristics of the dataset (size, sparsity, feature types, correlations, etc.)
- Reduce the search space for hyperparameter tuning

Return your answer using the AutoGluon model codes below (ONLY use these exact codes):

| Model Name | AutoGluon Code |
|------------|----------------|
| LightGBM | `GBM` |
| CatBoost | `CAT` |
| XGBoost | `XGB` |
| Random Forest | `RF` |
| Extra Trees | `XT` |
| Linear Model (Logistic/Ridge) | `LR` |
| k-Nearest Neighbors | `KNN` |
| Neural Network (PyTorch) | `NN_TORCH` |
| RealMLP | `REALMLP` |
| TabM | `TABM` |
| FastAI | `FASTAI` |
| FT_TRANSFORMER | `FT_TRANSFORMER` |
| TABPFNV2 | `TABPFNV2` |
| MITRA | `MITRA` |

### CONSTRAINTS:
- Respond **only** with a JSON object that validates against the provided schema (field: `selected_models`).
- Include ONLY the exact codes from the table above (e.g., GBM, CAT, XGB, RF, XT, LR, KNN).
- Do not include explanations, reasoning, or extra fields.
- Do NOT use codes like EBM, GREEDYTREE, FIGS, etc. that are not in the table.
"""

        user_prompt = f"""
You will be given a Data Profile. Use it to choose and return the reduced set of AutoGluon model codes as 'selected_models'.

---DATA PROFILE START---
{data_profile_str}
---DATA PROFILE END---
"""

        try:
            response = openai_client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": MODEL_SELECTION_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=ModelSelection,
                temperature=0.2,
                max_tokens=2048,
            )

            parsed: ModelSelection = response.choices[0].message.parsed
            selected_models = parsed.selected_models

            _llm_logger.info(f"🧠 LLM selected models: {selected_models}")
            return list(selected_models)

        except Exception as e:
            _llm_logger.error(
                f"⚠️  Model selection failed: {e}, using default models")
            # Fallback to default models
            return ['GBM', 'CAT', 'XGB']

    def train_with_autogluon(
        self,
        sampled_df: pd.DataFrame,
        target_field: str,
        status_callback: Optional[Callable[[str, str], None]] = None
    ) -> tuple[dict, Any]:
        """
        Train using AutoGluon with LLM-based model selection (nov5 research lines 1105-1180).
        Uses structured outputs to select optimal models, then extracts comprehensive metrics.
        """
        import time

        import numpy as np
        from autogluon.tabular import TabularPredictor
        from sklearn.metrics import (accuracy_score, f1_score,
                                     mean_absolute_error, mean_squared_error,
                                     r2_score, roc_auc_score)
        from sklearn.model_selection import train_test_split

        _llm_logger.info(
            "🤖 Training with AutoGluon (nov5 research with LLM model selection)...")

        try:
            # Verify target column exists
            if target_field not in sampled_df.columns:
                raise ValueError(
                    f"Target column '{target_field}' not found in dataframe")

            # Train-test split (80/20 like nov5)
            train_data, test_data = train_test_split(
                sampled_df, test_size=0.2, random_state=42
            )

            _llm_logger.info(
                f"📊 Train size: {len(train_data)}, Test size: {len(test_data)}")

            # Stage: Model Recommendation (LLM-based model selection)
            if status_callback:
                status_callback("Model Recommendation", "start")

            # Generate data profile for LLM-based model selection (nov5 Cell 38)
            _llm_logger.info(
                "📈 Generating data profile for model selection...")
            from ml_pipeline.data_profiler import generate_complete_profile
            data_profile = generate_complete_profile(sampled_df, target_field)
            data_profile_str = str(data_profile)

            # Use LLM to select optimal models (nov5 lines 1105-1180)
            _llm_logger.info(
                "🧠 Asking LLM to select optimal models based on data profile...")
            selected_models = self.select_models_with_llm(data_profile_str)
            hyperparameters = {m: {} for m in selected_models}

            if status_callback:
                status_callback("Model Recommendation", "complete")

            # Stage 8: Model Training
            if status_callback:
                status_callback("Model Training", "start")

            # Create unique model path in backend's models/ directory
            model_path = f'../../models/autogluon_models_{int(time.time())}'

            _llm_logger.info(f"🏋️ Training models: {selected_models}")

            # Configure predictor (let AutoGluon auto-detect problem type)
            predictor = TabularPredictor(label=target_field, path=model_path)

            # Train with bagging and stacking
            predictor.fit(
                train_data=train_data,
                hyperparameters=hyperparameters,
                num_bag_folds=5,
                num_bag_sets=1,
                num_stack_levels=1,
                verbosity=0
            )

            # Get predictor info
            info = predictor.info()
            best_model = info['best_model']
            problem_type = predictor.problem_type.lower()

            _llm_logger.info(f"✅ Training complete. Best model: {best_model}")
            _llm_logger.info(f"   Problem type: {problem_type}")
            _llm_logger.info(
                f"   DEBUG - Problem type value: '{problem_type}' (type: {type(problem_type)})")
            _llm_logger.info(
                f"   Eval metric: {info.get('eval_metric', 'N/A')}")

            # Evaluate and get comprehensive metrics (nov5 Cell 46 approach)
            y_true = test_data[target_field]
            y_pred = predictor.predict(test_data)

            metrics = {}

            # Check for classification (AutoGluon returns 'binary', 'multiclass', or 'regression')
            if "class" in problem_type or problem_type == "binary":  # Classification
                _llm_logger.info("   Detected Classification Problem")

                from sklearn.metrics import (log_loss, precision_score,
                                             recall_score)

                acc = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
                precision = precision_score(
                    y_true, y_pred, average='macro', zero_division=0)
                recall = recall_score(
                    y_true, y_pred, average='macro', zero_division=0)

                _llm_logger.info(f"   DEBUG - Computed accuracy: {acc}")
                _llm_logger.info(f"   DEBUG - Computed f1_macro: {f1}")
                _llm_logger.info(f"   DEBUG - Computed precision: {precision}")
                _llm_logger.info(f"   DEBUG - Computed recall: {recall}")

                metrics['accuracy'] = float(acc)
                metrics['f1_macro'] = float(f1)
                metrics['precision'] = float(precision)
                metrics['recall'] = float(recall)

                # Compute log loss (cross-entropy) if probabilities available
                if predictor.can_predict_proba:
                    try:
                        y_proba = predictor.predict_proba(test_data)
                        loss = log_loss(y_true, y_proba)
                        metrics['loss'] = float(loss)
                        _llm_logger.info(
                            f"   DEBUG - Computed log_loss: {loss}")
                    except Exception as e:
                        _llm_logger.error(f"Could not compute log loss: {e}")

                _llm_logger.info(
                    f"   DEBUG - Metrics dict after adding classification metrics: {list(metrics.keys())}")

                # Try ROC-AUC for binary classification
                if predictor.can_predict_proba:
                    y_proba = predictor.predict_proba(test_data)
                    # Only for binary classification
                    if len(np.unique(y_true)) == 2 and problem_type == "binary":
                        try:
                            roc = roc_auc_score(y_true, y_proba.iloc[:, 1])
                            metrics['roc_auc'] = float(roc)
                            _llm_logger.info(
                                f"   DEBUG - Computed roc_auc: {roc}")
                        except Exception as e:
                            _llm_logger.error(
                                f"Could not compute ROC-AUC: {e}")

            elif "regression" in problem_type:  # Regression
                _llm_logger.info("   Detected Regression Problem")

                r2 = r2_score(y_true, y_pred)
                mae = mean_absolute_error(y_true, y_pred)
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)

                metrics['r2'] = float(r2)
                metrics['mae'] = float(mae)
                metrics['loss'] = float(mse)  # MSE as loss for regression
                metrics['rmse'] = float(rmse)

                _llm_logger.info(f"   DEBUG - Computed r2: {r2}")
                _llm_logger.info(f"   DEBUG - Computed mae: {mae}")
                _llm_logger.info(f"   DEBUG - Computed mse (loss): {mse}")
                _llm_logger.info(f"   DEBUG - Computed rmse: {rmse}")

            else:
                _llm_logger.warning(
                    f"   WARNING - Unknown problem type: '{problem_type}' - no metrics computed")
                _llm_logger.warning(
                    f"   Expected: 'binary', 'multiclass', or 'regression'")

            # Add additional info
            metrics['sample_size'] = int(len(test_data))
            metrics['train_size'] = int(len(train_data))
            metrics['best_model'] = str(best_model)
            metrics['problem_type'] = str(problem_type)
            metrics['eval_metric'] = str(info.get('eval_metric', 'unknown'))

            # Get best model hyperparameters (nov5 Cell 43)
            try:
                model_info = info['model_info'][best_model]
                metrics['best_model_hyperparameters'] = str(
                    model_info.get('hyperparameters', {}))
            except:
                pass

            # Add leaderboard info (top 3 models)
            leaderboard = predictor.leaderboard(test_data, silent=True)
            if not leaderboard.empty:
                top_models = leaderboard.head(
                    3)[['model', 'score_test', 'score_val']].to_dict('records')
                metrics['top_models'] = [
                    {k: (float(v) if isinstance(v, (np.integer, np.floating)) else str(v))
                     for k, v in model.items()}
                    for model in top_models
                ]

            # Get validation score for best model
            try:
                best_score = leaderboard.loc[leaderboard['model']
                                             == best_model, 'score_val'].item()
                metrics['best_model_score_val'] = float(best_score)
            except:
                pass

            # Extract ONLY the best model (discard all other trained models)
            _llm_logger.info(f"📦 Extracting best model: {best_model}")
            best_model_obj = predictor._trainer.load_model(best_model)

            # Store model path info for reference (predictor saves all models here)
            metrics['all_models_path'] = str(model_path)

            # Debug: Log final metrics before returning
            _llm_logger.info(
                f"   DEBUG - Final metrics keys: {list(metrics.keys())}")
            _llm_logger.info(f"   DEBUG - Final metrics values: {metrics}")

            # Mark Model Training stage as complete
            if status_callback:
                status_callback("Model Training", "complete")

            # Return metrics and ONLY the best model object
            _llm_logger.info(f"✅ Returning best model: {best_model}")
            return metrics, best_model_obj

        except Exception as e:
            _llm_logger.error(f"❌ AutoGluon training failed: {e}")
            import traceback
            _llm_logger.error(traceback.format_exc())

            # Mark Model Training stage as failed
            if status_callback:
                status_callback("Model Training", "error")

            return {'error': str(e)}, None
