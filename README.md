# Week 7 - Day 2: Advanced RAG Retrieval

This project implements the two uploaded Week 7 Day 2 assessment sheets:

1. BM25 and vector search side by side (12 questions).
2. Hybrid search using weighted Reciprocal Rank Fusion (RRF).
3. Query rewriting into three alternative phrasings.
4. HyDE (Hypothetical Document Embeddings).
5. Bonus: combine query rewriting and HyDE with RRF.
6. Optional Qdrant native dense + sparse hybrid retrieval bonus.

## 1. Project structure

```text
Week7-Day2/
├── data/
│   ├── evaluation_questions.json
│   ├── vague_questions.json
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── core.py
│   └── evaluation.py
├── tests/
│   ├── test_core.py
│   └── test_rrf.py
├── outputs/
├── .env.example
├── .gitignore
├── main.py
└── requirements.txt
```

## 2. VS Code setup (Windows PowerShell)

Open the `Week7-Day2` folder in VS Code. Ensure your existing project is beside it:

```text
D:\Training\Week_7\
├── Week7-Day1\
│   └── data\       # your existing source documents
└── Week7-Day2\
```

Open the VS Code terminal and run:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy the environment file:

```powershell
Copy-Item .env.example .env
```

The default `DATA_DIR=../Week7-Day1/data` reads your existing Day 1 documents.
If your documents are stored elsewhere, update `DATA_DIR` in `.env`.

## 3. Run the assessment

Run all tasks:

```powershell
python main.py
```

Run only BM25/vector and hybrid RRF:

```powershell
python main.py --task hybrid
```

Run only query rewriting:

```powershell
python main.py --task rewrite
```

Run only HyDE and the combined rewriting + HyDE bonus:

```powershell
python main.py --task hyde
```

Optional Qdrant native hybrid/sparse-vector bonus (install separately):

```powershell
pip install "qdrant-client[fastembed]"
python main.py --task qdrant
```

The first run downloads `sentence-transformers/all-MiniLM-L6-v2`.
A Hugging Face cache/symlink warning on Windows is generally non-fatal.

## 4. Optional LLM-generated rewriting and HyDE

The project works locally without an API. By default, `USE_LLM=false` uses
a deterministic local fallback so that the complete retrieval pipeline can
be tested without API charges.

To use an OpenAI-compatible provider, edit `.env`:

```env
USE_LLM=true
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=your-provider-model-id
```

Use a model ID that is currently available to your provider. Never commit `.env`.

Set `INPUT_COST_PER_1M` and `OUTPUT_COST_PER_1M` to your provider's published
prices if you want the estimated cost calculation. Keep them at 0 for free models.
The code reports provider token usage when returned. Cost is an estimate based
on the configured rates, not a provider invoice.

## 5. Evaluation methodology

### Task 1
For each of 12 test questions, BM25 and vector retrieval each return their
top 3 chunks. A hit is recorded when a returned chunk contains one of the
question's `expected_terms`.

The first six questions are exact-term questions; the remaining six are
meaning-based questions. The comparison is saved to
`outputs/task1_task2_results.csv`.

### Task 2
RRF combines ranked lists using:

`score(document) = sum(weight_i / (k + rank_i))`

The implementation uses 1-based ranks and `k=60`. BM25 weights of 0.3,
0.5 and 0.7 are compared; vector weight is `1 - BM25 weight`.
The default recommendation is selected from the best measured Hit@3 on
the supplied evaluation set. With a small dataset, also inspect individual
question results before choosing a production default.

### Task 3
The rewriting prompt asks for exactly three alternative phrasings and
abbreviation expansion. The original query and three rewrites are embedded
and searched independently, then fused with RRF. Ten short/vague questions
are evaluated. Raw and rewritten Hit@3 and latency are recorded.

### Task 4
The HyDE prompt generates a short hypothetical answer without consulting
the source documents. That answer is embedded and used as the retrieval
query. The same ten questions are used to compare raw retrieval, rewriting,
HyDE, and the combined rewriting + HyDE RRF strategy.

### Hit@3 definition
A question is a hit if at least one of the top three retrieved chunks contains
at least one expected phrase. This is a transparent, lightweight proxy for
retrieval relevance. It is not a human relevance judgment.

**Important:** Update `expected_terms` in the two JSON files so they match
the actual content of your Day 1 corpus. Otherwise, the reported hit rate
may be misleading. The sample terms reflect the leave/notice examples from
your earlier Day 1 work and may not exist in every source document.

## 6. Output files

- `outputs/task1_task2_results.csv`
- `outputs/task3_query_rewriting_results.csv`
- `outputs/task4_hyde_results.csv`

These CSVs contain per-question results for your assessment submission.

## 7. Four-line comparison for the report

- BM25 is strongest when a query contains exact names, policy codes, numbers
  or distinctive keywords found in the document.
- Vector search is useful when the question uses different wording but has
  a similar meaning to the document.
- Hybrid RRF combines lexical matching and semantic similarity, reducing
  dependence on only one retrieval method.
- The best BM25/vector weight should be selected using the evaluation set
  and validated on additional unseen questions.

## 8. When to use each strategy in production

- **Raw vector search:** use for clear natural-language questions and when
  low latency is important.
- **Query rewriting:** use for short, vague, ambiguous questions and
  abbreviations such as WFH.
- **HyDE:** use when the question is abstract or phrased very differently
  from the expected document language. It can hurt when the generated
  hypothetical answer introduces incorrect assumptions or misleading terms.
- **Hybrid RRF:** use when both exact identifiers and semantic questions
  are common, and retrieval quality matters more than the extra compute.

## 9. Tests

```powershell
pytest -q
```

## 10. GitHub submission

```powershell
git init
git add .
git commit -m "Complete Week 7 Day 2 advanced RAG retrieval"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

Do not upload `.env`, virtual environments, downloaded model files, or
private company documents unless you are authorized to share them.
