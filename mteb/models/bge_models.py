from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta, sentence_transformers_loader

bge_small_en_v1_5 = ModelMeta(
    loader=partial(
        sentence_transformers_loader,
        model_name="BAAI/bge-small-en-v1.5",
        revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        prompts={"query": "Represent this sentence for searching relevant passages: "},
    ),
    name="BAAI/bge-small-en-v1.5",
    languages=["eng_Latn"],
    open_source=True,
    revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    release_date="2023-09-12",  # initial commit of hf model.
)

class BGEM3Wrapper:
    """following the hf model card documentation."""

    def __init__(self, model_name: str, func: str = "dense_vecs", **kwargs: Any):
        self.model_name = model_name

        from FlagEmbedding import BGEM3FlagModel

        # Setting use_fp16 to True speeds up computation with a slight performance degradation
        self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        self.func = func

    def encode(  # type: ignore
        self,
        sentences: list[str],
        *,
        batch_size: int = 32,
        **kwargs: Any,
    ):
        if "prompt_name" in kwargs:
            kwargs.pop("prompt_name")
        if "request_qid" in kwargs:
            kwargs.pop("request_qid")
        if "show_progress_bar" in kwargs:
            kwargs.pop("show_progress_bar")
        if "convert_to_tensor" in kwargs:
            kwargs.pop("convert_to_tensor")

        return self.model.encode(
            sentences,
            batch_size=batch_size,
            **kwargs
        )[self.func]


bge_base_en_v1_5 = ModelMeta(
    loader=partial(
        sentence_transformers_loader,
        model_name="BAAI/bge-base-en-v1.5",
        revision="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
        prompts={"query": "Represent this sentence for searching relevant passages: "},
    ),
    name="BAAI/bge-base-en-v1.5",
    languages=["eng_Latn"],
    open_source=True,
    revision="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
    release_date="2023-09-11",  # initial commit of hf model.
)

bge_large_en_v1_5 = ModelMeta(
    loader=partial(
        sentence_transformers_loader,
        model_name="BAAI/bge-large-en-v1.5",
        revision="d4aa6901d3a41ba39fb536a557fa166f842b0e09",
        prompts={"query": "Represent this sentence for searching relevant passages: "},
    ),
    name="BAAI/bge-large-en-v1.5",
    languages=["eng_Latn"],
    open_source=True,
    revision="d4aa6901d3a41ba39fb536a557fa166f842b0e09",
    release_date="2023-09-12",  # initial commit of hf model.
)


bge_m3 = ModelMeta(
    loader=partial(BGEM3Wrapper, model_name="BAAI/bge-m3", func="dense_vecs"),  # type: ignore
    name="BAAI/bge-m3",
    languages=["eng_Latn"],
    open_source=True,
    revision="5617a9f61b028005a4858fdac845db406aefb181",
    release_date="2024-02-05",  # initial commit of hf model.
)
