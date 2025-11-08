<h3 align="center">Not All Features Deserve Attention:</h3>
<p align="center">Graph-Guided Dependency Learning for Tabular Data Generation with Language Models</p>

[![arXiv](https://img.shields.io/badge/arXiv-2507.18504-b31b1b.svg)](https://arxiv.org/abs/2507.18504)

GraDe is a synthetic tabular data generation framework that uses graph-guided dependency learning with language models. It extends language model capabilities with dynamic graph learning to capture and respect functional dependencies in tabular data.

This code is build on top of the code from the **great** work: [Language Models are Realistic Tabular Data Generators](https://github.com/tabularis-ai/be_great).

## Setup

```bash
pip install -r requirements.txt
```

## Functional Dependencies

Functional dependencies are relationships between columns where values in one set of columns uniquely determine values in another set.

In GraDe, functional dependencies are specified as a list of pairs, where each pair contains:
- Left features: indices of columns that determine other columns
- Right features: indices of columns that are determined by the left features

GraDe uses dynamic graph learning to enforce these dependencies during generation. 

For functional dependency extraction, we utilize both [HyFD](https://github.com/codocedo/hyfd) and [TANE](https://github.com/codocedo/tane) algorithms.


## Quickstart

GraDe can be used to generate synthetic tabular data while preserving the functional dependencies between columns:

```python
import pandas as pd
from grade import GraDe

data = pd.read_csv("your_data.csv")

# Format: [[[left_columns], [right_columns]], ...]
# Where left_columns determine right_columns
dependencies = [
    [[0, 1], [2, 3]],  # Columns 0 and 1 determine columns 2 and 3
    [[2], [4, 5]]      # Column 2 determines columns 4 and 5
]

model = GraDe(llm="gpt2", epochs=100, batch_size=64, fd_list=dependencies)
model.fit(data)
synthetic_data = model.sample(100)
```

## Citation 

If you found the resources in this repository useful, please link or cite our work:

``` bibtex
@inproceedings{zhang-etal-2025-features,
    title = "Not All Features Deserve Attention: Graph-Guided Dependency Learning for Tabular Data Generation with Language Models",
    author = "Zhang, Zheyu  and
      Yang, Shuo  and
      Prenkaj, Bardh  and
      Kasneci, Gjergji",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2025",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-emnlp.330/",
    pages = "6217--6242",
    ISBN = "979-8-89176-335-7",
    abstract = "Large Language Models (LLMs) have shown strong potential for tabular data generation by modeling textualized feature-value pairs. However, tabular data inherently exhibits sparse feature-level dependencies, where many feature interactions are structurally insignificant. This creates a fundamental mismatch as LLMs' self-attention mechanism inevitably distributes focus across all pairs, diluting attention on critical relationships, particularly in datasets with complex dependencies or semantically ambiguous features. To address this limitation, we propose GraDe (Graph-Guided Dependency Learning), a novel method that explicitly integrates sparse dependency graphs into LLMs' attention mechanism. GraDe employs a lightweight dynamic graph learning module guided by externally extracted functional dependencies, prioritizing key feature interactions while suppressing irrelevant ones. Our experiments across diverse real-world datasets demonstrate that GraDe outperforms existing LLM-based approaches by up to 12{\%} on complex datasets while achieving competitive results with state-of-the-art approaches in synthetic data quality. Our method is minimally intrusive yet effective, offering a practical solution for structure-aware tabular data modeling with LLMs."
}
```
