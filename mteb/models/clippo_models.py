from __future__ import annotations

import subprocess
from functools import partial
from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from mteb.model_meta import ModelMeta


def clippo_loader(**kwargs):
    RES = 224
    checkpoint_paths = {
        "clippo_b16_yfcc100m": "gs://big_vision/clippo/clippo_b16_yfcc100m.npz",
        "clippo_b16_yfcc100m_i21k_init": "gs://big_vision/clippo/clippo_b16_yfcc100m_i21k_init.npz",
        "clippo_b16_yfcc100m_i21k_init_25c4": "gs://big_vision/clippo/clippo_b16_yfcc100m_i21k_init_25c4.npz",
        "clippo_b16_yfcc100m_i21k_init_50c4": "gs://big_vision/clippo/clippo_b16_yfcc100m_i21k_init_50c4.npz",
        "clippo_b16_yfcc100m_i21k_init_75c4": "gs://big_vision/clippo/clippo_b16_yfcc100m_i21k_init_75c4.npz",
        "clippo_b16_100c4": "gs://big_vision/clippo/clippo_b16_100c4.npz",
    }

    try:
        import importlib
        import os
        import sys
        import jax

        repo_url = "https://github.com/google-research/big_vision.git"
        repo_dir = "big_vision"
        if not os.path.exists(repo_dir):
            subprocess.run(["git", "clone", "--branch=main", "--depth=1", repo_url])
            subprocess.run(["mv", "big_vision/big_vision/*", "big_vision/"])

        print("Installing dependencies...")
        subprocess.run(
            ["pip", "install", "-qr", "big_vision/requirements.txt"]
        )

        unifont_path = "big_vision/pp/proj/clippo/unifont-9.0.06.hex"
        if not os.path.exists(unifont_path):
            subprocess.run(["wget", "https://unifoundry.com/pub/unifont/unifont-9.0.06/font-builds/unifont-9.0.06.hex.gz", "https://unifoundry.com/pub/unifont/unifont-9.0.06/font-builds/unifont_upper-9.0.06.hex.gz"])
            subprocess.run(["gunzip", "unifont-9.0.06.hex.gz", "unifont_upper-9.0.06.hex.gz"])
            subprocess.run(["mv", "unifont-9.0.06.hex", "unifont_upper-9.0.06.hex", "big_vision/pp/proj/clippo/"])

        sys.path.insert(0, os.path.join(os.getcwd(), repo_dir))
        print(sys.path)

        from big_vision import utils
        from big_vision.configs.proj.clippo import train_clippo
        from big_vision.pp import builder as pp_builder

        config = train_clippo.get_config()
        for pp_modules in config.pp_modules:
            importlib.import_module(f"big_vision.pp.{pp_modules}")

        def tokenizer(inkey="text", outkey="text"):
            return (
                f"render_unifont("
                f'inkey="{inkey}", '
                f'outkey="{outkey}", '
                f"image_size={RES}, "
                f"lower=True, "
                f"font_size=16, "
                f"text_brightness=0, "
                f"background_brightness=127)|"
                f'value_range(-1, 1, inkey="{outkey}", outkey="{outkey}")'
            )

        pp_image_str = f"resize({RES})|value_range(-1,1)"
        pp_text_str = tokenizer()

        pp_image_fn = pp_builder.get_preprocess_fn(pp_image_str)
        pp_text_fn = pp_builder.get_preprocess_fn(pp_text_str)

    except Exception as e:
        raise ("Try rerunning.", e)


    class CLIPPOWrapper:
        def __init__(
            self, model_name: str = "clippo_b16_yfcc100m_i21k_init_25c4", **kwargs: Any
        ):
            self.model_name = model_name
            checkpoint_path = checkpoint_paths[model_name]
            subprocess.run(["gsutil", "cp", checkpoint_path, "."])
            config = train_clippo.get_config()
            model_module = importlib.import_module(
                f"big_vision.models.{config.model_name}"
            )
            self.mdl = model_module.Model(**config.model)
            print("Model loaded successfully:", self.mdl)
            self.params = utils.load_checkpoint_np(checkpoint_path)["params"]

        def preprocess_images(images):
            return [np.array(pp_image_fn({"image": img})["image"]) for img in images]

        def preprocess_texts(texts):
            return [np.array(pp_text_fn({"text": text})["text"]) for text in texts]

        @jax.jit
        def encode_images(self, images) -> np.ndarray:
            zimg, _, _ = self.mdl.apply({"params": self.params}, image=images)
            return zimg

        @jax.jit
        def encode_texts(self, texts) -> np.ndarray:
            ztxt, _, _ = self.mdl.apply({"params": self.params}, text=texts)
            return ztxt

        def get_image_embeddings(
            self, images: list[Image.Image] | DataLoader, batch_size: int = 32
        ) -> np.ndarray:
            all_image_embeddings = []

            if isinstance(images, DataLoader):
                for batch in tqdm(images):
                    inputs = self.preprocess_images(batch)
                    image_outputs = self.encode_images(inputs)
                    all_image_embeddings.append(image_outputs)
            else:
                for i in tqdm(range(0, len(images), batch_size)):
                    batch_images = images[i : i + batch_size]
                    image_outputs = self.encode_images(batch_images)
                    all_image_embeddings.append(image_outputs)

            all_image_embeddings = np.concatenate(all_image_embeddings)
            return all_image_embeddings

        def get_text_embeddings(
            self, texts: list[str], batch_size: int = 32
        ) -> np.ndarray:
            all_text_embeddings = []

            for i in tqdm(range(0, len(texts), batch_size)):
                batch_texts = texts[i : i + batch_size]
                inputs = self.preprocess_texts(batch_texts)
                text_outputs = self.encode_texts(inputs)
                all_text_embeddings.append(text_outputs)

            all_text_embeddings = np.concatenate(all_text_embeddings)
            return all_text_embeddings

        def calculate_probs(self, text_embeddings, image_embeddings):
            logits = image_embeddings @ text_embeddings.T
            probs = np.array(jax.nn.softmax(100 * logits, axis=1))
            return probs

        def get_fused_embeddings(
            self,
            texts: list[str] = None,
            images: list[Image.Image] = None,
            fusion_mode="sum",
            batch_size: int = 32,
        ):
            if texts is None and images is None:
                raise ValueError("Either texts or images must be provided")

            text_embeddings = None
            image_embeddings = None

            if texts is not None:
                text_embeddings = self.get_text_embeddings(
                    texts,
                    batch_size=batch_size,
                )

            if images is not None:
                image_embeddings = self.get_image_embeddings(
                    images,
                    batch_size=batch_size,
                )

            if text_embeddings is not None and image_embeddings is not None:
                if len(text_embeddings) != len(image_embeddings):
                    raise ValueError(
                        "The number of texts and images must have the same length"
                    )
                if fusion_mode == "sum":
                    fused_embeddings = text_embeddings + image_embeddings
                else:
                    # to do: add other fusion mode
                    raise ValueError(
                        f"fusion mode {fusion_mode} hasn't been implemented"
                    )
                return fused_embeddings
            elif text_embeddings is not None:
                return text_embeddings
            elif image_embeddings is not None:
                return image_embeddings

    return CLIPPOWrapper(**kwargs)


clippo_b16_yfcc100m = ModelMeta(
    loader=partial(
        clippo_loader,
        model_name="clippo_b16_yfcc100m",
    ),
    name="clippo_b16_yfcc100m",
    languages=["eng_Latn"],
    open_source=True,
    revision="1",
    release_date="2023-04-01",
)

clippo_b16_yfcc100m_i21k_init = ModelMeta(
    loader=partial(
        clippo_loader,
        model_name="clippo_b16_yfcc100m_i21k_init",
    ),
    name="clippo_b16_yfcc100m_i21k_init",
    languages=["eng_Latn"],
    open_source=True,
    revision="1",
    release_date="2023-04-01",
)

clippo_b16_yfcc100m_i21k_init_25c4 = ModelMeta(
    loader=partial(
        clippo_loader,
        model_name="clippo_b16_yfcc100m_i21k_init_25c4",
    ),
    name="clippo_b16_yfcc100m_i21k_init_25c4",
    languages=["eng_Latn"],
    open_source=True,
    revision="1",
    release_date="2023-04-01",
)

clippo_b16_yfcc100m_i21k_init_50c4 = ModelMeta(
    loader=partial(
        clippo_loader,
        model_name="clippo_b16_yfcc100m_i21k_init_50c4",
    ),
    name="clippo_b16_yfcc100m_i21k_init_50c4",
    languages=["eng_Latn"],
    open_source=True,
    revision="1",
    release_date="2023-04-01",
)

clippo_b16_yfcc100m_i21k_init_75c4 = ModelMeta(
    loader=partial(
        clippo_loader,
        model_name="clippo_b16_yfcc100m_i21k_init_75c4",
    ),
    name="clippo_b16_yfcc100m_i21k_init_75c4",
    languages=["eng_Latn"],
    open_source=True,
    revision="1",
    release_date="2023-04-01",
)

clippo_b16_100c4 = ModelMeta(
    loader=partial(
        clippo_loader,
        model_name="clippo_b16_100c4",
    ),
    name="clippo_b16_100c4",
    languages=["eng_Latn"],
    open_source=True,
    revision="1",
    release_date="2023-04-01",
)

