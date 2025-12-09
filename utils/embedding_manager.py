import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class EmbeddingManager:

    def __init__(
        self,
        base_dir=None,
        threshold=0.70,
        model="text-embedding-3-small"
    ):
        """
        threshold = similaridade mínima para reutilizar embedding existente
        model = modelo OpenAI
        """

        self.client = OpenAI()
        self.threshold = float(threshold)
        self.model = model

        # -------------------------------
        # Resolver diretório base
        # -------------------------------
        if base_dir is None:
            self.BASE_DIR = os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                )
            )
        else:
            self.BASE_DIR = base_dir

        # -------------------------------
        # Caminhos
        # -------------------------------
        self.CACHE_PATH = os.path.join(self.BASE_DIR, "data", "emb_cache.json")
        self.EMB_CSV_PATH = os.path.join(self.BASE_DIR, "data", "frases_embedded.csv")
        self.EMB_NPY_PATH = os.path.join(self.BASE_DIR, "models", "embeddings_frases.npy")

        # -------------------------------
        # 1. Carregar Cache JSON
        # -------------------------------
        if os.path.exists(self.CACHE_PATH):
            try:
                with open(self.CACHE_PATH, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception as e:
                print(f"[EMB] Cache corrompido ({e}). Criando novo cache limpo.")
                self.cache = {}
        else:
            self.cache = {}

        # -------------------------------
        # 2. Carregar base pré-embedada
        # -------------------------------
        if os.path.exists(self.EMB_CSV_PATH) and os.path.exists(self.EMB_NPY_PATH):
            self.emb_df = pd.read_csv(self.EMB_CSV_PATH)
            self.emb_matrix = np.load(self.EMB_NPY_PATH)
        else:
            self.emb_df = None
            self.emb_matrix = None

    # =====================================================================
    # UTILS
    # =====================================================================
    def _save_cache(self):
        with open(self.CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def _normalize(self, frase: str) -> str:
        return frase.strip().lower()

    def _vector(self, emb):
        """Garante shape (1,1536) independente da origem."""
        arr = np.array(emb, dtype=float)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        return arr

    # =====================================================================
    # EMBEDDING PRINCIPAL
    # =====================================================================
    def embed(self, frase: str):
        frase_norm = self._normalize(frase)

        # --------------------------------------
        # A) CACHE DIRETO
        # --------------------------------------
        if frase_norm in self.cache:
            return self._vector(self.cache[frase_norm])

        # --------------------------------------
        # B) BUSCA EXATA EM BASE PRE-EMBEDDADA
        # --------------------------------------
        if self.emb_df is not None:
            match = self.emb_df[self.emb_df["frase"].str.lower() == frase_norm]
            if len(match) > 0:
                idx = int(match.iloc[0]["indice"])
                emb = self.emb_matrix[idx]

                self.cache[frase_norm] = emb.tolist()
                self._save_cache()
                return self._vector(emb)

        # --------------------------------------
        # C) MATCH SEMÂNTICO NA BASE PRE-EMBEDDADA
        # --------------------------------------
        if self.emb_matrix is not None:

            temp = self.client.embeddings.create(
                model=self.model,
                input=frase_norm
            ).data[0].embedding
            temp = np.array(temp).reshape(1, -1)

            sims = cosine_similarity(temp, self.emb_matrix)[0]
            idx_best = int(np.argmax(sims))
            sim_best = float(sims[idx_best])

            if sim_best >= self.threshold:
                emb = self.emb_matrix[idx_best]

                self.cache[frase_norm] = emb.tolist()
                self._save_cache()

                print(f"[EMB] Match semântico reutilizado sim={sim_best:.3f}")
                return self._vector(emb)

            print(f"[EMB] Frase nova — sim={sim_best:.3f}")

            # salvar embedding TEMP
            self.cache[frase_norm] = temp.flatten().tolist()
            self._save_cache()
            return temp

        # --------------------------------------
        # D) API — Fallback FINAL
        # --------------------------------------
        print(f"[EMB] API fallback para frase nova: {frase_norm}")

        emb = self.client.embeddings.create(
            model=self.model,
            input=frase_norm
        ).data[0].embedding

        arr = np.array(emb).reshape(1, -1)

        self.cache[frase_norm] = arr.flatten().tolist()
        self._save_cache()

        return arr

    # =====================================================================
    # EMBEDDING EM LOTE
    # =====================================================================
    def embed_lote(self, frases):
        return np.vstack([self.embed(f) for f in frases])

    # =====================================================================
    # SIMILARIDADE
    # =====================================================================
    def similaridade(self, frase1, frase2):
        e1 = self.embed(frase1)
        e2 = self.embed(frase2)
        return float(cosine_similarity(e1, e2)[0][0])

    # =====================================================================
    # BUSCA FRASE MAIS SEMELHANTE
    # =====================================================================
    def frase_mais_semelhante(self, frase):
        if self.emb_matrix is None:
            return None, None

        emb = self.embed(frase)
        sims = cosine_similarity(emb, self.emb_matrix)[0]
        idx = int(np.argmax(sims))

        return self.emb_df.iloc[idx]["frase"], float(sims[idx])

    # =====================================================================
    # DEBUG
    # =====================================================================
    def listar_cache(self):
        return list(self.cache.keys())
