import json
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

from utils.project_paths import BASE_DIR

load_dotenv()


class EmbeddingManager:
    def __init__(self, base_dir=None, threshold=0.70, model="text-embedding-3-small"):
        self.client = None
        self.threshold = float(threshold)
        self.model = model
        self.base_dir = Path(base_dir) if base_dir else BASE_DIR

        self.cache_path = self.base_dir / "data" / "emb_cache.json"
        self.emb_csv_path = self.base_dir / "modelos" / "embeddings_frases_meta.csv"
        self.emb_npy_path = self.base_dir / "modelos" / "embeddings_frases.npy"

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = self._load_cache()
        self.emb_df, self.emb_matrix = self._load_precomputed_embeddings()

    def _load_cache(self):
        if not self.cache_path.exists():
            return {}

        try:
            with self.cache_path.open("r", encoding="utf-8") as file:
                return json.load(file)
        except Exception as exc:
            print(f"[EMB] Cache corrompido ({exc}). Criando novo cache limpo.")
            return {}

    def _load_precomputed_embeddings(self):
        if not self.emb_csv_path.exists() or not self.emb_npy_path.exists():
            return None, None

        return pd.read_csv(self.emb_csv_path), np.load(self.emb_npy_path)

    def _get_client(self):
        if self.client is None:
            self.client = OpenAI()
        return self.client

    def _save_cache(self):
        with self.cache_path.open("w", encoding="utf-8") as file:
            json.dump(self.cache, file, ensure_ascii=False, indent=2)

    def _normalize(self, frase: str) -> str:
        return frase.strip().lower()

    def _vector(self, emb):
        arr = np.asarray(emb, dtype=float)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        return arr

    def _find_phrase_match(self, frase_norm: str):
        if self.emb_df is None:
            return None

        phrase_column = "frase" if "frase" in self.emb_df.columns else "frase_exemplo"
        match = self.emb_df[self.emb_df[phrase_column].astype(str).str.lower() == frase_norm]
        if match.empty:
            return None

        if "indice" in match.columns:
            return int(match.iloc[0]["indice"])
        return int(match.index[0])

    def embed(self, frase: str):
        frase_norm = self._normalize(frase)

        if frase_norm in self.cache:
            return self._vector(self.cache[frase_norm])

        idx = self._find_phrase_match(frase_norm)
        if idx is not None and self.emb_matrix is not None:
            emb = self.emb_matrix[idx]
            self.cache[frase_norm] = emb.tolist()
            self._save_cache()
            return self._vector(emb)

        if self.emb_matrix is not None:
            temp = self._get_client().embeddings.create(
                model=self.model,
                input=frase_norm,
            ).data[0].embedding
            temp = np.asarray(temp).reshape(1, -1)

            sims = cosine_similarity(temp, self.emb_matrix)[0]
            idx_best = int(np.argmax(sims))
            sim_best = float(sims[idx_best])

            if sim_best >= self.threshold:
                emb = self.emb_matrix[idx_best]
                self.cache[frase_norm] = emb.tolist()
                self._save_cache()
                print(f"[EMB] Match semantico reutilizado sim={sim_best:.3f}")
                return self._vector(emb)

            print(f"[EMB] Frase nova - sim={sim_best:.3f}")
            self.cache[frase_norm] = temp.flatten().tolist()
            self._save_cache()
            return temp

        print(f"[EMB] API fallback para frase nova: {frase_norm}")
        emb = self._get_client().embeddings.create(
            model=self.model,
            input=frase_norm,
        ).data[0].embedding

        arr = np.asarray(emb).reshape(1, -1)
        self.cache[frase_norm] = arr.flatten().tolist()
        self._save_cache()
        return arr

    def embed_lote(self, frases):
        return np.vstack([self.embed(frase) for frase in frases])

    def similaridade(self, frase1, frase2):
        e1 = self.embed(frase1)
        e2 = self.embed(frase2)
        return float(cosine_similarity(e1, e2)[0][0])

    def frase_mais_semelhante(self, frase):
        if self.emb_matrix is None or self.emb_df is None:
            return None, None

        emb = self.embed(frase)
        sims = cosine_similarity(emb, self.emb_matrix)[0]
        idx = int(np.argmax(sims))
        phrase_column = "frase" if "frase" in self.emb_df.columns else "frase_exemplo"
        return self.emb_df.iloc[idx][phrase_column], float(sims[idx])

    def listar_cache(self):
        return list(self.cache.keys())
