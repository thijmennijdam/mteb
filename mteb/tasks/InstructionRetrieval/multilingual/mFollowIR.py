from __future__ import annotations

from collections import defaultdict

import datasets

from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskInstructionRetrieval import AbsTaskInstructionRetrieval

_LANGUAGES = {
    "fas": ["fas-Arab"],
    "rus": ["rus-Cyrl"],
    "zho": ["zho-Hans"],
}

EVAL_SPLIT = "test"


def load_data(
    path: str,
    langs: list,
    eval_splits: list,
    cache_dir: str | None = None,
    revision: str | None = None,
):
    corpus = {lang: {EVAL_SPLIT: {}} for lang in langs}
    queries = {lang: {EVAL_SPLIT: {}} for lang in langs}
    og_relevant_docs = {lang: {EVAL_SPLIT: {}} for lang in langs}
    changed_relevant_docs = {lang: {EVAL_SPLIT: {}} for lang in langs}
    top_ranked = {lang: {EVAL_SPLIT: {}} for lang in langs}

    for lang in langs:
        # Load corpus data
        corpus_data = datasets.load_dataset(
            path,
            f"corpus-{lang}",
            cache_dir=cache_dir,
            revision=revision,
            trust_remote_code=True,
        )
        corpus[lang][EVAL_SPLIT] = {
            row["_id"]: {"title": row["title"], "text": row["text"]}
            for row in corpus_data["corpus"]
        }

        # Load queries data
        queries_data = datasets.load_dataset(
            path,
            f"queries-{lang}",
            cache_dir=cache_dir,
            revision=revision,
            trust_remote_code=True,
        )
        queries[lang][EVAL_SPLIT] = {
            row["_id"]: {
                "text": row["text"],
                "instruction_og": row["instruction_og"],
                "instruction_changed": row["instruction_changed"],
                "keywords": row["keywords"] if "keywords" in row else None,
                "short_query": row["short_query"] if "short_query" in row else None,
            }
            for row in queries_data["queries"]
        }

        # Load qrels_og data
        qrels_og_data = datasets.load_dataset(
            path,
            f"qrels_og-{lang}",
            cache_dir=cache_dir,
            revision=revision,
            trust_remote_code=True,
        )
        for row in qrels_og_data[EVAL_SPLIT]:
            if row["query-id"] not in og_relevant_docs[lang][EVAL_SPLIT]:
                og_relevant_docs[lang][EVAL_SPLIT][row["query-id"]] = {
                    row["corpus-id"]: int(row["score"])
                }
            else:
                og_relevant_docs[lang][EVAL_SPLIT][row["query-id"]][
                    row["corpus-id"]
                ] = int(row["score"])

        # Load qrels_changed data
        qrels_changed_data = datasets.load_dataset(
            path,
            f"qrels_changed-{lang}",
            cache_dir=cache_dir,
            revision=revision,
            trust_remote_code=True,
        )
        for row in qrels_changed_data[EVAL_SPLIT]:
            if row["query-id"] not in changed_relevant_docs[lang][EVAL_SPLIT]:
                changed_relevant_docs[lang][EVAL_SPLIT][row["query-id"]] = {
                    row["corpus-id"]: int(row["score"])
                }
            else:
                changed_relevant_docs[lang][EVAL_SPLIT][row["query-id"]][
                    row["corpus-id"]
                ] = int(row["score"])

        # Load top_ranked data
        top_ranked_data = datasets.load_dataset(
            path,
            f"top_ranked-{lang}",
            cache_dir=cache_dir,
            revision=revision,
            trust_remote_code=True,
        )
        for row in top_ranked_data["top_ranked"]:
            if row["qid"] not in top_ranked[lang][EVAL_SPLIT]:
                top_ranked[lang][EVAL_SPLIT][row["qid"]] = [row["pid"]]
            else:
                top_ranked[lang][EVAL_SPLIT][row["qid"]].append(row["pid"])

    # make og_instructions and changed_instructions from queries and then turn queries into just queries
    og_instructions = {lang: {EVAL_SPLIT: defaultdict(dict)} for lang in queries}
    changed_instructions = {lang: {EVAL_SPLIT: defaultdict(dict)} for lang in queries}
    queries_only = {lang: {EVAL_SPLIT: {}} for lang in queries}
    for lang in queries:
        for split in queries[lang]:
            for qid in queries[lang][split]:
                text = queries[lang][split][qid]["text"]
                og_instructions[lang][split][text] = queries[lang][split][qid][
                    "instruction_og"
                ]
                changed_instructions[lang][split][text] = queries[lang][split][qid][
                    "instruction_changed"
                ]
                queries_only[lang][split][qid] = text

    queries = queries_only

    return (
        corpus,
        queries,
        og_instructions,
        changed_instructions,
        og_relevant_docs,
        changed_relevant_docs,
        top_ranked,
    )


class mFollowIRCrossLingual(MultilingualTask, AbsTaskInstructionRetrieval):
    metadata = TaskMetadata(
        name="mFollowIRCrossLingualInstructionRetrieval",
        description="This tasks measures retrieval instruction following ability on NeuCLIR narratives for the mFollowIR benchmark on the Farsi, Russian, and Chinese languages.",
        reference="https://neuclir.github.io/",
        dataset={
            "path": "jhu-clsp/mFollowIR-cross-lingual",
            "revision": "main",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=[EVAL_SPLIT],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_20",
        date=("2021-08-01", "2022-06-30"),
        domains=["News", "Written"],
        task_subtypes=[],
        license="odc-by",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{weller2024mfollowir,
  title={{mFollowIR: a Multilingual Benchmark for Instruction Following in Information Retrieval}},
  author={Weller, Orion and Chang, Benjamin and Yang, Eugene and Yarmohammadi, Mahsa and Barham, Sam and MacAvaney, Sean and Cohan, Arman and Soldaini, Luca and Van Durme, Benjamin and Lawrie, Dawn},
  journal={arXiv preprint TODO},
  year={2024}
}""",
        descriptive_stats={
            "n_samples": {"fas": 43 * 2, "rus": 40 * 2, "zho": 43 * 2},
            "test": {
                "num_docs": 121635,
                "num_queries": 126,
                "average_document_length": 2331.0777818884367,
                "average_query_length": 81.87301587301587,
                "average_instruction_length": 389.14285714285717,
                "average_changed_instruction_length": 448.5238095238095,
                "average_relevant_docs_per_query": 10.30952380952381,
                "average_top_ranked_per_query": 1000,
                "hf_subset_descriptive_stats": {
                    "fas": {
                        "num_docs": 41189,
                        "num_queries": 43,
                        "average_document_length": 3145.4990895627475,
                        "average_query_length": 80.18604651162791,
                        "average_instruction_length": 394.0232558139535,
                        "average_changed_instruction_length": 456.3488372093023,
                        "average_relevant_docs_per_query": 10.465116279069768,
                        "average_top_ranked_per_query": 1000,
                    },
                    "rus": {
                        "num_docs": 39326,
                        "num_queries": 40,
                        "average_document_length": 2784.0813456746173,
                        "average_query_length": 81.875,
                        "average_instruction_length": 371.125,
                        "average_changed_instruction_length": 431.8,
                        "average_relevant_docs_per_query": 9.775,
                        "average_top_ranked_per_query": 1000,
                    },
                    "zho": {
                        "num_docs": 41120,
                        "num_queries": 43,
                        "average_document_length": 1082.0501215953307,
                        "average_query_length": 83.55813953488372,
                        "average_instruction_length": 401.0232558139535,
                        "average_changed_instruction_length": 456.25581395348837,
                        "average_relevant_docs_per_query": 10.651162790697674,
                        "average_top_ranked_per_query": 1000,
                    },
                },
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        (
            self.corpus,
            self.queries,
            self.og_instructions,
            self.changed_instructions,
            self.og_relevant_docs,
            self.changed_relevant_docs,
            self.top_ranked,
        ) = load_data(
            path=self.metadata_dict["dataset"]["path"],
            langs=self.metadata.eval_langs,
            eval_splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True
