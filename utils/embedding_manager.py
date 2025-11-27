"""
# 📘 EmbeddingManager — Guia de Uso (Resumo Oficial)

O `EmbeddingManager` é o utilitário responsável por transformar frases em vetores semânticos (embeddings) de forma **rápida, consistente e econômica**, reutilizando informações sempre que possível.
Ele combina três fontes de embeddings:

1. **Cache local** (`emb_cache.json`)
2. **Base pré-embeddada** (`frases_embedded.csv` + `embeddings_frases.npy`)
3. **Fallback via API OpenAI**

O objetivo é evitar cálculos repetidos, reduzir custo de API e garantir que frases semanticamente semelhantes recebam embeddings consistentes.

---

## 🚀 Importação

```python
from utils.embedding_manager import EmbeddingManager

emb = EmbeddingManager()
```

---

## 🔹 Gerar embedding de uma frase

```python
vetor = emb.embed("queda nos estoques de petróleo")
```

Pipeline interno:

1. tenta recuperar do cache
2. tenta buscar na base pré-embeddada
3. tenta correspondência semântica (cosine similarity)
4. se tudo falhar → gera via API e salva no cache

O retorno é um vetor `1 × 1536`.

---

## 🔹 Gerar embeddings em lote

```python
frases = [
    "OPEP reduz produção",
    "Demanda global aumenta",
    "Estoques caem nos EUA"
]

matriz = emb.embed_lote(frases)
```

Retorna matriz Nx1536.

---

## 🔹 Similaridade entre duas frases

```python
sim = emb.similaridade("queda nos estoques", "estoques caem nos EUA")
print(sim)
```

Retorna similaridade do cosseno entre –1 e 1.

---

## 🔹 Buscar frase mais semelhante da base

```python
frase_base, sim = emb.frase_mais_semelhante("produção da OPEP sobe")
print(frase_base, sim)
```

Retorna a frase mais parecida da base pré-embeddada.

---

## 🔹 Listar frases do cache

```python
emb.listar_cache()
```

Mostra todas as frases que já possuem embedding armazenado localmente.

"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI


class EmbeddingManager:

    def __init__(
        self,
        base_dir=None,
        threshold=0.70,
        model="text-embedding-3-small"
    ):
        """
        base_dir = raiz do projeto (TimesSeriesAgent)
        threshold = similaridade mínima para reutilizar embedding existente
        model = modelo de embedding da OpenAI
        """

        self.client = OpenAI()
        self.threshold = float(threshold)
        self.model = model

        # Resolver diretório raiz automaticamente
        if base_dir is None:
            self.BASE_DIR = os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                )
            )
        else:
            self.BASE_DIR = base_dir

        # Caminhos do sistema
        self.CACHE_PATH = os.path.join(self.BASE_DIR, "data", "emb_cache.json")
        self.EMB_CSV_PATH = os.path.join(self.BASE_DIR, "data", "frases_embedded.csv")
        self.EMB_NPY_PATH = os.path.join(self.BASE_DIR, "models", "embeddings_frases.npy")

        # -----------------------
        # 1. Carregar Cache JSON
        # -----------------------
        if os.path.exists(self.CACHE_PATH):
            with open(self.CACHE_PATH, "r", encoding="utf-8") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}

        # -----------------------
        # 2. Carregar base pré-embedada
        # -----------------------
        if os.path.exists(self.EMB_CSV_PATH) and os.path.exists(self.EMB_NPY_PATH):
            self.emb_df = pd.read_csv(self.EMB_CSV_PATH)
            self.emb_matrix = np.load(self.EMB_NPY_PATH)
        else:
            self.emb_df = None
            self.emb_matrix = None

    # =====================================================================
    # UTILIDADES INTERNAS
    # =====================================================================

    def _save_cache(self):
        """Salva o cache no disco."""
        with open(self.CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def _normalize(self, frase: str) -> str:
        """Normaliza frase para lookup no cache."""
        return frase.strip().lower()

    # =====================================================================
    # EMBEDDING PRINCIPAL (um por vez)
    # =====================================================================

    def embed(self, frase: str):
        """
        Retorna embedding da frase com pipeline:
        1. Cache
        2. Base CSV+NPY
        3. Similaridade semântica
        4. API OpenAI
        """

        frase_norm = self._normalize(frase)

        # ------------------------------------------
        # A) CACHE DIRETO
        # ------------------------------------------
        if frase_norm in self.cache:
            return np.array(self.cache[frase_norm]).reshape(1, -1)

        # ------------------------------------------
        # B) BUSCA EXATA NA BASE EMBEDDADA
        # ------------------------------------------
        if self.emb_df is not None:
            match = self.emb_df[self.emb_df["frase"].str.lower() == frase_norm]
            if len(match) > 0:
                idx = int(match.iloc[0]["indice"])
                emb = self.emb_matrix[idx]

                self.cache[frase_norm] = emb.tolist()
                self._save_cache()
                return emb.reshape(1, -1)

        # ------------------------------------------
        # C) BUSCA SEMÂNTICA — cosine similarity
        # ------------------------------------------
        if self.emb_matrix is not None:

            # gerar embedding temporário
            emb_temp = self.client.embeddings.create(
                model=self.model,
                input=frase_norm
            ).data[0].embedding
            emb_temp = np.array(emb_temp).reshape(1, -1)

            sims = cosine_similarity(emb_temp, self.emb_matrix)[0]
            idx_best = int(np.argmax(sims))
            sim_best = float(sims[idx_best])

            if sim_best >= self.threshold:
                emb = self.emb_matrix[idx_best]

                self.cache[frase_norm] = emb.tolist()
                self._save_cache()

                print(f"[EMB] Match semântico: sim={sim_best:.3f} — usando embedding existente.")
                return emb.reshape(1, -1)

            print(f"[EMB] Frase nova — sim={sim_best:.3f}")

            # salvar embedding TEMP
            self.cache[frase_norm] = emb_temp.tolist()
            self._save_cache()
            return emb_temp

        # ------------------------------------------
        # D) API — fallback final
        # ------------------------------------------
        print(f"[EMB] API fallback para frase nova: {frase_norm}")

        emb = self.client.embeddings.create(
            model=self.model,
            input=frase_norm
        ).data[0].embedding

        self.cache[frase_norm] = emb
        self._save_cache()

        return np.array(emb).reshape(1, -1)

    # =====================================================================
    # EMBEDDING EM LOTE
    # =====================================================================

    def embed_lote(self, frases: list):
        """
        Retorna embeddings para lista de frases.
        Usa cache + base + similares + API minimamente.
        """
        return np.vstack([self.embed(f) for f in frases])

    # =====================================================================
    # SIMILARIDADE ENTRE DUAS FRASES
    # =====================================================================

    def similaridade(self, frase1: str, frase2: str):
        """
        Similaridade semântica entre duas frases usando cosine similarity.
        """
        e1 = self.embed(frase1)
        e2 = self.embed(frase2)
        return float(cosine_similarity(e1, e2)[0][0])

    # =====================================================================
    # BUSCAR FRASE MAIS SEMELHANTE NA BASE EMBEDDADA
    # =====================================================================

    def frase_mais_semelhante(self, frase: str):
        """
        Retorna (frase_base, similaridade) mais próxima da base embeddada.
        """
        if self.emb_matrix is None:
            return None, None

        emb = self.embed(frase)
        sims = cosine_similarity(emb, self.emb_matrix)[0]
        idx = int(np.argmax(sims))
        return self.emb_df.iloc[idx]["frase"], float(sims[idx])

    # =====================================================================
    # DEBUG / INSPEÇÃO
    # =====================================================================

    def listar_cache(self):
        """Lista frases no cache."""
        return list(self.cache.keys())
