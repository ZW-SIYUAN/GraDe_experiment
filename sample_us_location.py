"""
仅采样脚本：从已训练的 checkpoint 加载权重，跳过训练直接生成合成数据。
"""

import os
import sys
import logging
import time
import random

import numpy as np
import pandas as pd
import torch
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grade import GraDe

# ── 复现性 ────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── 路径 ──────────────────────────────────────────────────────────────────────
TRAIN_PATH      = os.path.join("data", "us_location", "train.csv")
CHECKPOINT_PATH = "checkpoints_us_location/checkpoint-51000"   # 最后一个 checkpoint
OUTPUT_PATH     = "synthetic_us_location.csv"

# ── 与训练脚本完全一致的超参 ──────────────────────────────────────────────────
LLM             = "gpt2-medium"
SPARSITY_LAMBDA = 0.001
FD_LAMBDA       = 0.1
FD_ALPHA        = 0.5
NUM_HEAD_GROUPS = 4
FD_LIST = [
    [[0], [3]],
    [[0], [4]],
    [[1], [4]],
    [[3], [0]],
]

# 采样参数
TEMPERATURE = 0.8
MAX_LENGTH  = 50
SAMPLE_K    = 100
TOP_P       = 1.0
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
USE_BF16    = torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def main():
    logger.info("加载数据：%s", TRAIN_PATH)
    train_df = pd.read_csv(TRAIN_PATH)
    n_samples = len(train_df)
    logger.info("训练集行数（目标采样量）：%d", n_samples)

    # 1. 重建与训练时完全相同的 GraDe 模型骨架（不训练）
    logger.info("重建模型骨架（llm=%s）...", LLM)
    model = GraDe(
        llm=LLM,
        experiment_dir="checkpoints_us_location",
        sparsity_lambda=SPARSITY_LAMBDA,
        use_dynamic_graph=True,
        num_head_groups=NUM_HEAD_GROUPS,
        fd_lambda=FD_LAMBDA,
        fd_alpha=FD_ALPHA,
        fd_list=FD_LIST,
        bf16=USE_BF16,
        fp16=False,
    )

    # 2. 用 PeftModel 加载 checkpoint 里的 LoRA adapter 权重
    logger.info("加载 LoRA adapter：%s", CHECKPOINT_PATH)
    model.model = PeftModel.from_pretrained(
        model.model,
        CHECKPOINT_PATH,
        is_trainable=False,   # 推理模式，不需要梯度
    )
    model.model.eval()
    logger.info("LoRA adapter 加载完毕")

    # 3. 手动设置列信息（训练时 fit() 会自动设，现在跳过训练需手动补）
    model.columns     = train_df.columns.tolist()
    model.num_cols    = train_df.select_dtypes(include=np.number).columns.tolist()
    model.column_names = train_df.columns.tolist()
    # 用训练集的 state_code 分布作为采样起点
    model.conditional_col      = "state_code"
    model.conditional_col_dist = train_df["state_code"].value_counts(normalize=True).to_dict()

    # 4. 采样
    logger.info("开始采样 %d 行  |  T=%.2f  k=%d  max_len=%d  device=%s",
                n_samples, TEMPERATURE, SAMPLE_K, MAX_LENGTH, DEVICE)
    t0 = time.time()
    synthetic_df = model.sample(
        n_samples=n_samples,
        start_col="state_code",
        start_col_dist=model.conditional_col_dist,
        temperature=TEMPERATURE,
        k=SAMPLE_K,
        top_p=TOP_P,
        max_length=MAX_LENGTH,
        drop_nan=False,
        device=DEVICE,
    )
    elapsed = time.time() - t0
    logger.info("采样完成  |  %.1f s  |  生成 %d / %d 行", elapsed, len(synthetic_df), n_samples)

    # 5. 保存
    synthetic_df.to_csv(OUTPUT_PATH, index=False)
    logger.info("合成数据已保存：%s", OUTPUT_PATH)
    logger.info("预览：\n%s", synthetic_df.head(5).to_string())


if __name__ == "__main__":
    main()
