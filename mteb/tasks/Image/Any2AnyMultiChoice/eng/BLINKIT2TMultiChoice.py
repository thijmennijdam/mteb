from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyMultiChoice import AbsTaskAny2AnyMultiChoice
from mteb.abstasks.TaskMetadata import TaskMetadata


class BLINKIT2TMultiChoice(AbsTaskAny2AnyMultiChoice):
    metadata = TaskMetadata(
        name="BLINKIT2TMultiChoice",
        description="Retrieve the correct text answer based on images and specific retrieval instructions.",
        reference="https://arxiv.org/abs/2404.12390",
        dataset={
            "path": "JamieSJS/blink-it2t-multi",
            "revision": "ae713b03ae68e343f16c3bcdbd1b1ee760975d55",
        },
        type="Any2AnyMultiChoice",
        category="it2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2018-01-01", "2018-12-31"),
        domains=["Encyclopaedic"],
        task_subtypes=["Image Text Retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation="""@article{fu2024blink,
  title={Blink: Multimodal large language models can see but not perceive},
  author={Fu, Xingyu and Hu, Yushi and Li, Bangzheng and Feng, Yu and Wang, Haoyu and Lin, Xudong and Roth, Dan and Smith, Noah A and Ma, Wei-Chiu and Krishna, Ranjay},
  journal={arXiv preprint arXiv:2404.12390},
  year={2024}
}
""",
        descriptive_stats={
            "n_samples": {"test": 923},
            "avg_character_length": {
                "test": {
                    "average_document_length": 0.0,
                    "average_query_length": 0.0,
                    "num_documents": 24,
                    "num_queries": 923,
                    "average_relevant_docs_per_query": 1,
                }
            },
        },
    )