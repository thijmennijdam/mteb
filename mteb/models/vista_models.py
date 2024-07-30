from __future__ import annotations

from functools import partial

import torch

try:  # a temporal fix for the dependency issues of vista models.
    from FlagEmbedding.visual.modeling import Visualized_BGE
except ImportError:
    Visualized_BGE = None
from PIL import Image
from tqdm import tqdm

from mteb.model_meta import ModelMeta

if Visualized_BGE is not None:

    class VisualizedBGEWrapper(Visualized_BGE):
        def __init__(
            self,
            model_name_bge: str = None,
            model_weight=None,
            normlized: bool = True,
            sentence_pooling_method: str = "cls",
            negatives_cross_device: bool = False,
            temperature: float = 0.02,
            from_pretrained=None,
        ):
            super().__init__(
                model_name_bge=model_name_bge,
                model_weight=model_weight,
                normlized=normlized,
                sentence_pooling_method=sentence_pooling_method,
                negatives_cross_device=negatives_cross_device,
                temperature=temperature,
                from_pretrained=from_pretrained,
            )
            self.eval()

        def encode_text(self, texts):
            """Currently override Visualized_BGE's the original implementation
            to fix attention_mask & embedding_output dtype misalignment
            """
            input_ids = texts["input_ids"]
            attention_mask = texts["attention_mask"]

            input_shape = input_ids.size()
            device = input_ids.device

            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

            head_mask = [None] * self.depth
            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
                attention_mask, input_shape
            )

            embedding_output = self.bge_embeddings(
                input_ids=input_ids,
                position_ids=None,
                token_type_ids=token_type_ids,
                inputs_embeds=None,
                past_key_values_length=0,
            )

            # this line is missing in vista, currently override "encode_text" only to fix this.
            extended_attention_mask = extended_attention_mask.to(embedding_output.dtype)

            encoder_outputs = self.bge_encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_values=None,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            sequence_output = encoder_outputs[0]

            t_reps = self.sentence_embedding(
                sequence_output, texts["attention_mask"]
            )  # tensor: reps with pooling
            if self.normlized:
                t_reps = torch.nn.functional.normalize(t_reps, dim=-1)
            return t_reps.contiguous()

        def encode(self, images=None, texts=None):
            if images is not None:
                if isinstance(images, list):
                    images = [
                        self.preprocess_val(
                            img if isinstance(img, Image.Image) else Image.open(img)
                        )
                        for img in images
                    ]
                    images = torch.stack(images)
                if texts is not None:
                    texts = self.tokenizer(texts, return_tensors="pt", padding=True)
                    return self.encode_mm(images.to(self.device), texts.to(self.device))
                else:
                    return self.encode_image(images.to(self.device))
            else:
                if texts is not None:
                    texts = self.tokenizer(texts, return_tensors="pt", padding=True)
                    return self.encode_text(texts.to(self.device))
                else:
                    return None

        def get_text_embeddings(self, texts: list[str], batch_size: int = 32):
            all_text_embeddings = []
            for i in tqdm(range(0, len(texts), batch_size)):
                batch_texts = texts[i : i + batch_size]
                with torch.no_grad():
                    batch_embeddings = self.encode(texts=batch_texts)
                all_text_embeddings.append(batch_embeddings.cpu())
            return torch.cat(all_text_embeddings, dim=0)

        def get_image_embeddings(self, images: list[Image.Image], batch_size: int = 32):
            all_image_embeddings = []
            for i in tqdm(range(0, len(images), batch_size)):
                batch_images = images[i : i + batch_size]
                with torch.no_grad():
                    batch_embeddings = self.encode(images=batch_images)
                all_image_embeddings.append(batch_embeddings.cpu())
            return torch.cat(all_image_embeddings, dim=0)

        def get_fused_embeddings(
            self,
            texts: list[str] = None,
            images: list[Image.Image] = None,
            batch_size: int = 32,
        ):
            all_embeddings = []
            assert len(texts) == len(images)
            for i in tqdm(range(0, len(texts), batch_size)):
                batch_texts = texts[i : i + batch_size]
                batch_images = images[i : i + batch_size]
                with torch.no_grad():
                    batch_embeddings = self.encode(
                        images=batch_images, texts=batch_texts
                    )
                all_embeddings.append(batch_embeddings.cpu())
            return torch.cat(all_embeddings, dim=0)

        def calculate_probs(self, text_embeddings, image_embeddings):
            text_embeddings = text_embeddings / text_embeddings.norm(
                dim=-1, keepdim=True
            )
            image_embeddings = image_embeddings / image_embeddings.norm(
                dim=-1, keepdim=True
            )
            logits = torch.matmul(image_embeddings, text_embeddings.T)
            probs = (logits * 100).softmax(dim=-1)
            return probs

    Visualized_BGE_base = ModelMeta(
        loader=partial(
            VisualizedBGEWrapper,
            model_name_bge="BAAI/bge-base-en-v1.5",
            model_weight="visualized_base_en_V1.5.pth",
        ),
        name="BAAI/bge-visualized-base",
        languages=["eng_Latn"],
        open_source=True,
        revision="98db10b10d22620010d06f11733346e1c98c34aa",
        release_date="2024-06-06",
    )

    Visualized_BGE_base = ModelMeta(
        loader=partial(
            VisualizedBGEWrapper,
            model_name_bge="BAAI/bge-m3",
            model_weight="visualized_m3.pth",
        ),
        name="BAAI/bge-visualized-m3",
        languages=["eng_Latn"],
        open_source=True,
        revision="98db10b10d22620010d06f11733346e1c98c34aa",
        release_date="2024-06-06",
    )

if __name__ == "__main__":
    if (
        Visualized_BGE is None
    ):  # a temporal fix to the dependency issues of Vista models.
        print("Visualized_BGE module is not available.")
    else:
        import mteb

        mdl = mteb.get_model(
            Visualized_BGE_base.name, Visualized_BGE_base.name.revision
        )
        emb = mdl.get_text_embeddings(["Hello, world!"])
