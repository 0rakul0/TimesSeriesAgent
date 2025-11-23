"""
modelo_eval_baseline_hibrido_v3_1.py
----------------------------------------------------------------
Versão multi-modelos:
 - Compara vários modelos simultaneamente
 - Gera 1 único HTML com:
      • Real
      • Técnico de cada modelo
      • Híbrido (somente LSTMs e Transformer)
      • RMSE técnico e híbrido na legenda
----------------------------------------------------------------
Uso:

python modelo_eval_baseline_hibrido_v3_1.py \
    --comparar models/LSTM_d0.1.pt models/LSTM_d0.3.pt models/Transformer.pt \
    --modo full
"""
import argparse
import os
import random
import glob              # <-- adicione isto
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import plotly.graph_objects as go
from tqdm import tqdm
import json

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
openai_client = OpenAI()

# ======================= CONFIG ============================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CAMINHO_DADOS = "../data/dados_combinados.csv"
PASTA_JSON = "../output_noticias"
OUTPUT_DIR = "../outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_FRASES = "../data/frases_impacto_clusters.csv"
EMB_PATH    = "../models/embeddings_frases.npy"

SEQ_LEN_DEFAULT = 30

# ======================= MODELOS SUPORTADOS ============================

import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden=128, layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(1)

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden=128, layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, layers,
                          batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze(1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-np.log(10000.) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden=128,
                 heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.LayerNorm(hidden)
        )
        self.pos = PositionalEncoding(hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        z = self.embed(x)
        z = self.pos(z)
        out = self.encoder(z)
        return self.fc(out[:, -1, :]).squeeze(1)

class MLPLag(nn.Module):
    def __init__(self, input_size, seq_len=30):
        super().__init__()
        self.seq_len = seq_len
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size * seq_len, 256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )
    def forward(self, x):
        return self.fc(x).squeeze(1)

# ======================= EMBEDDINGS ============================

def carregar_base_frases():
    if not os.path.exists(CSV_FRASES):
        raise FileNotFoundError(CSV_FRASES)

    meta_df = pd.read_csv(CSV_FRASES, parse_dates=["data"])
    meta_df["motivo"] = meta_df["motivo"].astype(str)

    if os.path.exists(EMB_PATH):
        print("[INFO] Carregando embeddings com tqdm...")
        emb_mmap = np.load(EMB_PATH, mmap_mode="r")
        emb_matrix = np.empty_like(emb_mmap)
        for i in tqdm(range(len(emb_mmap)), desc="Embeddings", ncols=80):
            emb_matrix[i] = emb_mmap[i]
        return meta_df, emb_matrix

    # Gerar se não existir
    emb_list = []
    for motivo in tqdm(meta_df["motivo"].tolist(),
                       desc="Gerando embeddings", ncols=80):
        try:
            emb = openai_client.embeddings.create(
                model="text-embedding-3-large",
                input=motivo
            ).data[0].embedding
        except:
            emb = [0]*1536
        emb_list.append(emb)

    emb_matrix = np.array(emb_list, dtype=np.float32)
    np.save(EMB_PATH, emb_matrix)
    return meta_df, emb_matrix

# ======================= NOTÍCIAS ============================

def extrair_motivos(pasta_json, janela_dias, ref_date):
    arquivos = glob.glob(os.path.join(pasta_json, "evento_*.json"))
    ref = pd.to_datetime(ref_date)
    inicio = ref - pd.Timedelta(days=janela_dias)

    noticias = {}
    for arq in arquivos:
        try:
            with open(arq, "r", encoding="utf-8") as f:
                j = json.load(f)
            dt = pd.to_datetime(j.get("data"))
            if not (inicio <= dt <= ref):
                continue
            motivos = j.get("motivos_identificados", [])
            motivos = [m.strip() for m in motivos if str(m).strip()]
            if motivos:
                noticias.setdefault(dt.normalize(), []).extend(motivos)
        except:
            continue
    return noticias

from sklearn.metrics.pairwise import cosine_similarity

def buscar_similares(motivo, meta_df, emb_matrix, top_k=8, threshold=0.55):
    motivos = meta_df["motivo"].tolist()

    try:
        emb = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=motivo
        ).data[0].embedding
        emb = np.array(emb).reshape(1,-1)
        sims = cosine_similarity(emb, emb_matrix)[0]
        idxs = np.argsort(sims)[::-1][:top_k]

        out = []
        for i in idxs:
            if sims[i] >= threshold:
                out.append((motivos[i],
                            float(meta_df.iloc[i]["peso_medio"]),
                            float(sims[i])))
        return out
    except:
        return []

def calcular_pesos(noticias, meta_df, emb_matrix):
    pesos = {}
    for dia, motivos in noticias.items():
        lista = []
        for m in motivos:
            sims = buscar_similares(m, meta_df, emb_matrix)
            if sims:
                num = sum(p*s for (_,p,s) in sims)
                den = sum(s for (_,_,s) in sims)
                if den > 0:
                    lista.append(num/den)
        if lista:
            pesos[dia] = float(np.mean(lista))
    return pesos

# ======================= CORREÇÃO HÍBRIDA ============================

def aplicar_correcao(pred_df, pesos, modo="30d"):
    df = pred_df.copy()
    df["Pred_Ajustado"] = df["Pred"]

    dias = df.index
    if modo == "30d":
        target = dias[-30:]
    else:
        target = dias

    eventos = sorted(pesos.items(),
                     key=lambda x: pd.to_datetime(x[0]))

    mapa = {}

    for dt_raw, peso_raw in eventos:
        dt = pd.to_datetime(dt_raw)
        pos = dias.searchsorted(dt)
        if pos >= len(dias):
            continue

        d0 = dias[pos]
        peso = float(peso_raw)

        for k in range(4):
            pos_k = pos + k
            if pos_k >= len(dias): break
            dk = dias[pos_k]

            if dk not in target:
                continue

            mapa[dk] = (d0, peso, k)

    for dia, (d0,peso,k) in mapa.items():
        alpha = np.exp(-k/4.0)
        suave = np.tanh((peso/100)*alpha)
        df.loc[dia, "Pred_Ajustado"] = df.loc[dia,"Pred"]*(1+suave)

    return df

# ======================= PREVISÃO ============================

def reconstruir_modelo(nome, input_size, seq_len):
    nome = nome.lower()
    if "gru" in nome:
        return GRUModel(input_size)
    elif "transformer" in nome:
        return TransformerModel(input_size)
    elif "mlp" in nome:
        return MLPLag(input_size, seq_len)
    else:
        return LSTMModel(input_size)

def carregar_modelo(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    cols = ckpt["train_columns"]
    target_col = ckpt["target_col"]
    seq_len = ckpt["seq_len"]
    scaler = ckpt["scaler"]

    model = reconstruir_modelo(path, len(cols), seq_len)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(DEVICE)
    model.eval()

    return model, scaler, target_col, seq_len, cols

def prever(model, scaler, df, seq_len, target_col, train_cols):
    df2 = df.copy()

    for c in train_cols:
        if c not in df2.columns:
            df2[c] = 0.0

    df2 = df2[train_cols]
    arr = scaler.transform(df2)

    tgt_idx = train_cols.index(target_col)

    preds_s = []
    reals_s = []
    dates = []

    for i in range(len(arr)-seq_len):
        seq = arr[i:i+seq_len]
        seq_t = torch.tensor(seq[np.newaxis,:,:], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            p = model(seq_t).cpu().numpy()[0]

        preds_s.append(p)
        reals_s.append(arr[i+seq_len, tgt_idx])
        dates.append(df2.index[i+seq_len])

    preds_s = np.array(preds_s)
    reals_s = np.array(reals_s)

    def inv(v):
        z = np.zeros((len(v), len(train_cols)))
        z[:, tgt_idx] = v
        return scaler.inverse_transform(z)[:, tgt_idx]

    return pd.DataFrame({
        "Date": dates,
        "Pred": inv(preds_s),
        "Real": inv(reals_s)
    }).set_index("Date")

# ======================= PLOT MULTI-MODELOS ============================

def plot_comparacao(resultados, modo, html_out):
    fig = go.Figure()

    # Plot Real (usar o primeiro modelo como referência)
    first_key = next(iter(resultados.keys()))
    fig.add_trace(go.Scatter(
        x=resultados[first_key]["df"].index,
        y=resultados[first_key]["df"]["Real"],
        name="Real",
        mode="lines",
        line=dict(color="black", width=3)
    ))

    # Plot técnico & híbrido
    for nome, info in resultados.items():
        df = info["df"]

        fig.add_trace(go.Scatter(
            x=df.index, y=df["Pred"],
            name=f"{nome} (Técnico | RMSE={info['rmse_tecnico']:.3f})",
            mode="lines"
        ))

        if info.get("df_hibrido") is not None:
            dfh = info["df_hibrido"]
            fig.add_trace(go.Scatter(
                x=dfh.index, y=dfh["Pred_Ajustado"],
                name=f"{nome} (Híbrido | RMSE={info['rmse_hibrido']:.3f})",
                mode="lines",
                line=dict(width=3, dash="dash")
            ))

    fig.update_layout(
        title=f"Comparação de Modelos — Ajuste Híbrido ({modo})",
        template="plotly_white",
        width=1400,
        height=700
    )

    fig.write_html(html_out)
    print("📊 HTML salvo em:", html_out)

# ======================= MAIN ============================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparar", nargs="+", required=True,
                        help="Lista de modelos .pt")
    parser.add_argument("--modo", choices=["30d","full"], default="30d",
                        help="30d=últimos 30 dias | full=série inteira")
    args = parser.parse_args()

    print("\n🔍 Avaliando modelos:", args.comparar)
    print("Modo:", args.modo)

    # Dados
    df = pd.read_csv(CAMINHO_DADOS, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).normalize()
    df = df.ffill().bfill().fillna(0.0)

    # Embeddings
    meta_df, emb_matrix = carregar_base_frases()

    resultados = {}

    for path in args.comparar:
        nome = os.path.basename(path).replace(".pt","")
        print(f"\n=== Carregando {nome} ===")

        model, scaler, target_col, seq_len, cols = carregar_modelo(path)

        # previsão técnica
        df_pred = prever(model, scaler, df, seq_len, target_col, cols)

        # preparar híbrido apenas para LSTM e Transformer
        nome_lower = nome.lower()
        precisa_hibrido = ("lstm" in nome_lower) or ("transformer" in nome_lower)

        df_hibrido = None
        rmse_hibrido = None

        if precisa_hibrido:
            print(f"[HÍBRIDO] Aplicando notícias para {nome}...")

            # extrair motivos 30d ou full
            if args.modo == "30d":
                noticias = extrair_motivos(PASTA_JSON, 30, df_pred.index[-1])
            else:
                lista_jsons = glob.glob(os.path.join(PASTA_JSON, "evento_*.json"))
                datas_jsons = [pd.to_datetime(os.path.basename(f).split("_")[-1].replace(".json", ""))
                               for f in lista_jsons]

                dias_total = (max(datas_jsons) - min(datas_jsons)).days

                noticias = extrair_motivos(PASTA_JSON, dias_total, df_pred.index[-1])

            pesos = calcular_pesos(noticias, meta_df, emb_matrix)

            df_hibrido = aplicar_correcao(df_pred, pesos, modo=args.modo)

            rmse_hibrido = np.sqrt(mean_squared_error(
                df_hibrido["Real"],
                df_hibrido["Pred_Ajustado"]
            ))

        rmse_tecnico = np.sqrt(mean_squared_error(
            df_pred["Real"], df_pred["Pred"]
        ))

        resultados[nome] = {
            "df": df_pred,
            "df_hibrido": df_hibrido,
            "rmse_tecnico": rmse_tecnico,
            "rmse_hibrido": rmse_hibrido
        }

    # Gerar HTML final
    out_file = os.path.join(
        OUTPUT_DIR,
        f"comparacao_multi_modelos_{args.modo}.html"
    )

    plot_comparacao(resultados, args.modo, out_file)

    print("\n🎯 Avaliação concluída.")

if __name__ == "__main__":
    main()
