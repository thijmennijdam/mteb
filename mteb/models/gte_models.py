from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta

from .instruct_wrapper import instruct_wrapper

class GTESTWrapper:
    def __init__(self, model_name: str, **kwargs: Any):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

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

        return self.model.encode(sentences, batch_size=batch_size, normalize_embeddings=True, **kwargs)


gte_Qwen2_7B_instruct = ModelMeta(
    loader=partial(
        instruct_wrapper,
        model_name_or_path="Alibaba-NLP/gte-Qwen2-7B-instruct",
        instruction_template="Instruct: {instruction}\nQuery: ",
        attn="cccc",
        pooling_method="lasttoken",
        mode="embedding",
        torch_dtype="auto",
        # The ST script does not normalize while the HF one does so unclear what to do
        # https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct#sentence-transformers
        normalized=True,
    ),
    name="Alibaba-NLP/gte-Qwen2-7B-instruct",
    languages=None,
    open_weights=True,
    revision="e26182b2122f4435e8b3ebecbf363990f409b45b",
    release_date="2024-06-15",  # initial commit of hf model.
    n_parameters=7_613_000_000,
    memory_usage=None,
    embed_dim=3584,
    license="apache-2.0",
    reference="https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
)

gte_multilingual_base = ModelMeta(
    loader=partial(GTESTWrapper, model_name="Alibaba-NLP/gte-multilingual-base"),
    name="Alibaba-NLP/gte-multilingual-base",
    languages=None,
    open_source=True,
    revision="7fc06782350c1a83f88b15dd4b38ef853d3b8503",
    release_date="2024-07-29",
)


if __name__ == "__main__":
    # Verify it reproduces https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct#sentence-transformers
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        "Alibaba-NLP/gte-Qwen2-7B-instruct", trust_remote_code=True
    )
    # Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████| 7/7 [00:10<00:00,  1.52s/it]
    # Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
    # In case you want to reduce the maximum length:
    model.max_seq_length = 8192
    queries = ["how much protein should a female eat", "summit define"]
    documents = [
        "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
        "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
    ]
    query_embeddings = model.encode(queries, prompt_name="query")
    document_embeddings = model.encode(documents)
    scores = (query_embeddings @ document_embeddings.T) * 100
    print(scores.tolist())
    # [[70.39706420898438, 3.4318461418151855], [4.516170978546143, 81.91815948486328]]

    import mteb

    mdl = mteb.get_model(gte_Qwen2_7B_instruct.name, gte_Qwen2_7B_instruct.revision)
    emb = mdl.encode(["Hello, world!"])
    model_mteb = mteb.get_model(
        "Alibaba-NLP/gte-Qwen2-7B-instruct"
    )  # gte_Qwen2_7B_instruct.name, gte_Qwen2_7B_instruct.revision)
    # Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.71it/s]
    # Created GritLM: torch.float32 dtype, lasttoken pool, embedding mode, cccc attn
    # Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
    # ----------Using 8 data-parallel GPUs----------
    query_embeddings_mteb = model_mteb.encode(
        queries,
        instruction="Given a web search query, retrieve relevant passages that answer the query",
    )
    document_embeddings_mteb = model_mteb.encode_corpus(documents)
    scores_mteb = (query_embeddings_mteb @ document_embeddings_mteb.T) * 100
    print(scores_mteb.tolist())
    # [[70.39706420898438, 3.4318461418151855], [4.516170978546143, 81.91815948486328]]
