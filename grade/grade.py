import warnings
import json
import typing as tp
import logging

import fsspec
import numpy as np
import pandas as pd

from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, GPT2Config

from .grade_start import (
    GReaTStart,
    CategoricalStart,
    ContinuousStart,
    RandomStart,
)
from .grade_trainer import GraDeTrainer
from .grade_utils import (
    _array_to_dataframe,
    _get_column_distribution,
    _convert_tokens_to_text,
    _convert_text_to_tabular_data,
    bcolors,
)
from .grade_model import TabDynamicGraphGPT2
from .grade_dataset import GraDeDataset, GraDeDataCollator


class GraDe:
    """GraDe Class for tabular data generation with graph structure learning."""

    def __init__(
        self,
        llm: str,
        experiment_dir: str = "trainer_great",
        epochs: int = 100,
        batch_size: int = 64,
        sparsity_lambda: float = 0.001,  # Graph sparsity regularization
        use_dynamic_graph: bool = True,  # Enable dynamic graph learning
        num_head_groups: int = 4,        # Attention head groups
        fd_lambda: float = 0.1,          # FD regularization weight
        fd_alpha: float = 0.5,           # FD minimum edge weight
        fd_list: list = None,            # Functional dependency list
        only_update_graph: bool = False,  # Only update graph parameters
        fixed_col_order: bool = False,    # Use fixed column order for training and sampling
        **train_kwargs,
    ):
        """Initialize GraDe model.

        Args:
            llm: HuggingFace checkpoint for the base LLM
            experiment_dir: Training checkpoints directory
            epochs: Number of training epochs
            batch_size: Training batch size
            sparsity_lambda: Graph sparsity regularization coefficient
            use_dynamic_graph: Whether to use dynamic graph learning
            num_head_groups: Number of attention head groups 
            fd_lambda: Weight coefficient for FD loss
            fd_alpha: Expected minimum edge weight for FD
            fd_list: Functional dependency list [[[left_feats], [right_feats]], ...]
            only_update_graph: Only update graph module parameters
            train_kwargs: Additional training hyperparameters
        """
        # Load Model and Tokenizer
        self.llm = llm
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model configuration
        config = GPT2Config.from_pretrained(self.llm)
        # Graph learning config
        config.sparsity_lambda = sparsity_lambda
        config.use_dynamic_graph = use_dynamic_graph
        config.num_head_groups = num_head_groups
        
        # FD configuration
        config.fd_lambda = fd_lambda
        config.fd_alpha = fd_alpha
        config.fd_list = fd_list if fd_list is not None else []
        
        # Initialize model
        self.model = TabDynamicGraphGPT2(config)
        original_model = AutoModelForCausalLM.from_pretrained(self.llm)
        self.model.load_state_dict(original_model.state_dict(), strict=False)
        
        # Save parameters
        self.sparsity_lambda = sparsity_lambda
        self.use_dynamic_graph = use_dynamic_graph
        self.num_head_groups = num_head_groups
        self.fd_lambda = fd_lambda
        self.fd_alpha = fd_alpha
        self.fd_list = fd_list if fd_list is not None else []
        self.only_update_graph = only_update_graph
        self.fixed_col_order = fixed_col_order

        # Training parameters
        self.experiment_dir = experiment_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_hyperparameters = train_kwargs

        # Data attributes
        self.columns = None
        self.num_cols = None
        self.conditional_col = None
        self.conditional_col_dist = None
        self.column_names = None

    def fit(
        self,
        data: tp.Union[pd.DataFrame, np.ndarray],
        column_names: tp.Optional[tp.List[str]] = None,
        conditional_col: tp.Optional[str] = None,
        resume_from_checkpoint: tp.Union[bool, str] = False,
        fd_list: tp.Optional[list] = None,
    ) -> GraDeTrainer:
        """Fine-tune model with tabular data.

        Args:
            data: Pandas DataFrame or Numpy Array with tabular data
            column_names: Feature names for Numpy Array input
            conditional_col: Column for generation starting point
            resume_from_checkpoint: Resume from checkpoint if True or path
            fd_list: Functional dependency list

        Returns:
            GReaTTrainer: The trainer object
        """
        df = _array_to_dataframe(data, columns=column_names)
        self._update_column_information(df)
        self._update_conditional_information(df, conditional_col)
        
        # Save column names and update FD list
        self.column_names = df.columns.tolist()
        # For fixed-order generation, save the first column's distribution
        if self.fixed_col_order:
            self.first_col_dist = _get_column_distribution(df, self.columns[0])
        if fd_list is not None:
            self.fd_list = fd_list
            self.model.fd_list = fd_list
            
        # Freeze non-graph parameters if needed
        if self.only_update_graph:
            total_params = 0
            trainable_params = 0
            attn_patterns = ['graph_generator', 'attn']
            
            for name, param in self.model.named_parameters():
                total_params += param.numel()
                is_attn_param = any(pattern in name for pattern in attn_patterns)
                
                if is_attn_param:
                    param.requires_grad = True
                    trainable_params += param.numel()
                else:
                    param.requires_grad = False
            
            # Log parameter stats
            trainable_percentage = 100.0 * trainable_params / total_params
            print(f"Only updating attention parameters: {trainable_params}/{total_params} ({trainable_percentage:.2f}%)")
            print(f"Update ratio: 1:{total_params/trainable_params:.1f}")

        # Prepare dataset and trainer
        logging.info("Converting data to dataset...")
        great_ds = GraDeDataset.from_pandas(df)
        great_ds.set_tokenizer(self.tokenizer, shuffle_columns=not self.fixed_col_order)

        # Column-aware data collator
        class ColumnAwareDataCollator(GraDeDataCollator):
            def __init__(self, tokenizer, column_names):
                super().__init__(tokenizer)
                self.column_names = column_names
            
            def __call__(self, features):
                batch = super().__call__(features)
                batch['column_names'] = self.column_names
                return batch

        # Configure trainer
        logging.info("Creating trainer...")
        training_args = TrainingArguments(
            self.experiment_dir,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            **self.train_hyperparameters,
        )
        
        great_trainer = GraDeTrainer(
            self.model,
            training_args,
            train_dataset=great_ds,
            tokenizer=self.tokenizer,
            data_collator=ColumnAwareDataCollator(self.tokenizer, self.column_names),
        )

        # Train model
        logging.info("Starting training...")
        great_trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        return great_trainer

    def sample(
        self,
        n_samples: int,
        start_col: tp.Optional[str] = "",
        start_col_dist: tp.Optional[tp.Union[dict, list]] = None,
        temperature: float = 0.7,
        k: int = 100,
        top_p: float = 1.0,
        max_length: int = 100,
        drop_nan: bool = False,
        device: str = "cuda",
    ) -> pd.DataFrame:
        """Generate synthetic tabular data samples.

        Args:
            n_samples: Number of samples to generate
            start_col: Feature to use as generation starting point
            start_col_dist: Feature distribution of starting feature
            temperature: Controls sampling randomness
            k: Sampling batch size
            top_p: Nucleus sampling parameter
            max_length: Maximum token length to generate
            drop_nan: Whether to drop rows with NaNs
            device: Device for generation

        Returns:
            DataFrame with generated samples
        """
        # Fixed-order generation: always start from the first column so the
        # prompt matches the fixed-order training distribution.
        if self.fixed_col_order and self.columns is not None:
            first_col = self.columns[0]
            if start_col and start_col != first_col:
                logging.warning(
                    "fixed_col_order=True: overriding start_col '%s' → '%s' (first column)",
                    start_col, first_col,
                )
            start_col = first_col
            start_col_dist = getattr(self, "first_col_dist", self.conditional_col_dist)

        great_start = self._get_start_sampler(start_col, start_col_dist)
        self.model.to(device)
        dfs = []

        with tqdm(total=n_samples) as pbar:
            already_generated = 0
            _cnt = 0
            try:
                while n_samples > already_generated:
                    start_tokens = great_start.get_start_tokens(k)
                    start_tokens = torch.tensor(start_tokens).to(device)
                    
                    # Generate tokens
                    generate_kwargs = {
                        "input_ids": start_tokens,
                        "max_length": max_length,
                        "do_sample": True,
                        "temperature": temperature,
                        "top_p": top_p,
                        "pad_token_id": 50256,
                    }
                    
                    tokens = self.model.generate(**generate_kwargs)

                    # Process generated data
                    text_data = _convert_tokens_to_text(tokens, self.tokenizer)
                    df_gen = _convert_text_to_tabular_data(text_data, self.columns)
                    df_gen = df_gen[~(df_gen == "placeholder").any(axis=1)]
                    df_gen = df_gen.dropna(how="all")
                    
                    if drop_nan:
                        df_gen = df_gen.dropna()

                    # Clean numerical values
                    for i_num_cols in self.num_cols:
                        coerced_series = pd.to_numeric(df_gen[i_num_cols], errors="coerce")
                        df_gen = df_gen[coerced_series.notnull() | df_gen[i_num_cols].isna()]

                    # Convert numerical columns
                    df_gen[self.num_cols] = df_gen[self.num_cols].astype(float)

                    dfs.append(df_gen)
                    already_generated += len(dfs[-1])
                    pbar.update(len(dfs[-1]))

                    # Safety check
                    _cnt += 1
                    if _cnt > 13 and already_generated == 0:
                        raise Exception("Breaking the generation loop!")

            except Exception as e:
                print(f"{bcolors.FAIL}Error: {str(e)}{bcolors.ENDC}")
                print(f"{bcolors.WARNING}Try increasing epochs or max_length parameter.{bcolors.ENDC}")
                print(f"{bcolors.OKBLUE}If problems persist, please report an issue.{bcolors.ENDC}")

        df_gen = pd.concat(dfs) if dfs else pd.DataFrame(columns=self.columns)
        df_gen = df_gen.reset_index(drop=True)
        return df_gen.head(n_samples)

    def sample_from_prompts(
        self,
        starting_prompts: tp.Union[str, list[str]],
        n_samples: int = 1,
        temperature: float = 0.7,
        max_length: int = 100,
        top_p: float = 0.95,
        drop_nan: bool = False,
        device: str = "cuda",
    ) -> pd.DataFrame:
        """Generate synthetic data based on text prompts.

        Args:
            starting_prompts: Conditioning prompt(s) (e.g., "Sex is female, Age is 26")
            n_samples: Samples per prompt
            temperature: Controls sampling randomness
            max_length: Maximum token length
            top_p: Nucleus sampling parameter
            drop_nan: Whether to drop rows with NaNs
            device: Generation device

        Returns:
            DataFrame with generated samples
        """
        self.model.to(device)
        starting_prompts = [starting_prompts] if isinstance(starting_prompts, str) else starting_prompts
        
        dfs = []
        total_samples = len(starting_prompts) * n_samples
        
        with tqdm(total=total_samples) as pbar:
            already_generated = 0
            _cnt = 0
            try:
                for prompt in starting_prompts:
                    prompt_generated = 0
                    start_token = torch.tensor(self.tokenizer(prompt)["input_ids"]).to(device)
                    
                    while prompt_generated < n_samples:
                        batch_size = min(100, n_samples - prompt_generated)
                        batch_input = torch.unsqueeze(start_token, 0).repeat(batch_size, 1)
                        
                        # Generate tokens
                        gens = self.model.generate(
                            input_ids=batch_input,
                            max_length=max_length,
                            do_sample=True,
                            temperature=temperature,
                            top_p=top_p,
                            pad_token_id=50256,
                        )
                        
                        # Process generations
                        text_data = _convert_tokens_to_text(gens, self.tokenizer)
                        df_gen = _convert_text_to_tabular_data(text_data, self.columns)
                        df_gen = df_gen[~(df_gen == "placeholder").any(axis=1)]
                        df_gen = df_gen.dropna(how="all")
                        
                        if drop_nan:
                            df_gen = df_gen.dropna()
                        
                        # Clean numerical values
                        for i_num_cols in self.num_cols:
                            coerced_series = pd.to_numeric(df_gen[i_num_cols], errors="coerce")
                            df_gen = df_gen[coerced_series.notnull() | df_gen[i_num_cols].isna()]
                        
                        df_gen[self.num_cols] = df_gen[self.num_cols].astype(float)
                        
                        dfs.append(df_gen)
                        prompt_generated += len(df_gen)
                        already_generated += len(df_gen)
                        pbar.update(len(df_gen))
                        
                        # Safety check
                        _cnt += 1
                        if _cnt > 13 and already_generated == 0:
                            raise Exception("Breaking the generation loop!")
                        
            except Exception as e:
                print(f"{bcolors.FAIL}Error: {str(e)}{bcolors.ENDC}")
                print(f"{bcolors.WARNING}Try increasing epochs or max_length parameter.{bcolors.ENDC}")
                print(f"{bcolors.OKBLUE}If problems persist, please report an issue.{bcolors.ENDC}")
        
        if not dfs:
            return pd.DataFrame(columns=self.columns)
        
        df_gen = pd.concat(dfs)
        df_gen = df_gen.reset_index(drop=True)
        return df_gen.head(total_samples)

    def save(self, path: str):
        """Save model weights and configuration."""
        fs = fsspec.filesystem(fsspec.utils.get_protocol(path))
        if fs.exists(path):
            warnings.warn(f"Directory {path} already exists and will be overwritten.")
        else:
            fs.mkdir(path)

        # Save attributes
        with fs.open(path + "/config.json", "w") as f:
            attributes = self.__dict__.copy()
            attributes.pop("tokenizer")
            attributes.pop("model")

            # Convert ndarray to list for JSON
            if isinstance(attributes["conditional_col_dist"], np.ndarray):
                attributes["conditional_col_dist"] = list(attributes["conditional_col_dist"])

            json.dump(attributes, f)

        # Save model weights
        torch.save(self.model.state_dict(), fs.open(path + "/model.pt", "wb"))

    def load_finetuned_model(self, path: str):
        """Load fine-tuned model weights."""
        self.model.load_state_dict(torch.load(fsspec.open(path, "rb")))

    @classmethod
    def load_from_dir(cls, path: str):
        """Load trained model from directory."""
        fs = fsspec.filesystem(fsspec.utils.get_protocol(path))
        assert fs.exists(path), f"Directory {path} does not exist."

        # Load attributes
        with fs.open(path + "/config.json", "r") as f:
            attributes = json.load(f)

        # Create new model instance
        great = cls(
            attributes["llm"],
            sparsity_lambda=attributes.get("sparsity_lambda", 0.001),
            use_dynamic_graph=attributes.get("use_dynamic_graph", True),
            num_head_groups=attributes.get("num_head_groups", 4),
            fd_lambda=attributes.get("fd_lambda", 0.1),
            fd_alpha=attributes.get("fd_alpha", 0.5),
            fd_list=attributes.get("fd_list", None),
            only_update_graph=attributes.get("only_update_graph", False),
        )

        # Set all attributes
        for k, v in attributes.items():
            setattr(great, k, v)

        # Initialize model
        config = GPT2Config.from_pretrained(great.llm)
        config.sparsity_lambda = great.sparsity_lambda
        config.use_dynamic_graph = great.use_dynamic_graph
        config.num_head_groups = great.num_head_groups
        config.fd_lambda = great.fd_lambda
        config.fd_alpha = great.fd_alpha
        config.fd_list = great.fd_list
        great.model = TabDynamicGraphGPT2(config)

        # Load weights
        great.model.load_state_dict(torch.load(fs.open(path + "/model.pt", "rb"), map_location="cpu"))

        return great

    def _update_column_information(self, df: pd.DataFrame):
        """Update column metadata."""
        self.columns = df.columns.to_list()
        self.num_cols = df.select_dtypes(include=np.number).columns.to_list()

    def _update_conditional_information(
        self, df: pd.DataFrame, conditional_col: tp.Optional[str] = None
    ):
        """Update conditional feature information."""
        assert conditional_col is None or isinstance(
            conditional_col, str
        ), f"Column name must be a string, not {type(conditional_col)}"
        assert (
            conditional_col is None or conditional_col in df.columns
        ), f"Column {conditional_col} not in dataset"

        self.conditional_col = conditional_col if conditional_col else df.columns[-1]
        self.conditional_col_dist = _get_column_distribution(df, self.conditional_col)

    def _get_start_sampler(
        self,
        start_col: tp.Optional[str],
        start_col_dist: tp.Optional[tp.Union[tp.Dict, tp.List]],
    ) -> GReaTStart:
        """Get the appropriate sampler for starting generation."""
        if start_col and start_col_dist is None:
            raise ValueError(f"Start column {start_col} given without distribution")
        if start_col_dist is not None and not start_col:
            raise ValueError(f"Start column distribution given without column name")

        assert start_col is None or isinstance(
            start_col, str
        ), f"Column name must be a string, not {type(start_col)}"
        assert (
            start_col_dist is None
            or isinstance(start_col_dist, dict)
            or isinstance(start_col_dist, list)
        ), f"Distribution must be list or dict, not {type(start_col_dist)}"

        start_col = start_col if start_col else self.conditional_col
        start_col_dist = start_col_dist if start_col_dist else self.conditional_col_dist

        if isinstance(start_col_dist, dict):
            return CategoricalStart(self.tokenizer, start_col, start_col_dist)
        elif isinstance(start_col_dist, list):
            return ContinuousStart(self.tokenizer, start_col, start_col_dist)
        else:
            return RandomStart(self.tokenizer, self.columns)

    def set_fd_list(self, fd_list, fd_lambda=None, fd_alpha=None):
        """Update functional dependency configuration."""
        self.fd_list = fd_list
        self.model.fd_list = fd_list
        
        if fd_lambda is not None:
            self.fd_lambda = fd_lambda
            self.model.fd_lambda = fd_lambda
        
        if fd_alpha is not None:
            self.fd_alpha = fd_alpha
            self.model.fd_alpha = fd_alpha
        
        return self
    
    def get_column_names(self):
        """Get column names for FD mapping."""
        return self.column_names if self.column_names else []
    
