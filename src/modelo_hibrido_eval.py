"""
modelo_eval_baseline_hibrido_v2.py
------------------------------------------------------------
Versão aprimorada do modelo híbrido:
✅ Usa embeddings OpenAI para calcular similaridade semântica entre motivos.
✅ Aplica correção bidirecional (positiva e negativa) apenas nos últimos 30 dias.
✅ Mostra comparação clara entre previsão técnica e ajustada.
------------------------------------------------------------
Autor: Jefferson Anjos
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
import torch.serialization
from datetime import timedelta
from openai import OpenAI
from dotenv import load_dotenv

# ============== CONFIGURAÇÃO ==============
load_dotenv()
openai_client = OpenAI()

CAMINHO_DADOS = "../data/dados_combinados.csv"
PASTA_JSON = "../output_noticias"
MODELO_BASELINE = "../models/lstm_baseline_price.pt"
HTML_SAIDA = "../img/previsao_baseline_hibrido_v2.html"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN_DEFAULT = 30
CSV_FRASES = "../data/frases_impacto_clusters.csv"
EMB_PATH = "../models/embeddings_frases.npy"
EMB_META = "../data/frases_embedded.csv"


# ============== CARREGAR BASE E EMBEDDINGS ==============
def carregar_base_frases():
    """
    Carrega a base de frases (motivos) e seus embeddings.
    Se EMB_PATH existir, carrega com mmap e tqdm (barra de progresso).
    Se NÃO existir, gera embeddings uma única vez usando API e salva.

    Retorna:
        meta_df (DataFrame) - frases/motivos
        emb_matrix (np.ndarray) - matriz de embeddings (1536 dims)
    """
    import os
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    # ---------------------------------------------
    # 1. Carregar CSV com frases
    # ---------------------------------------------
    if not os.path.exists(CSV_FRASES):
        raise FileNotFoundError(f"Arquivo de frases não encontrado: {CSV_FRASES}")

    meta_df = pd.read_csv(CSV_FRASES, parse_dates=["data"])
    meta_df["motivo"] = meta_df["motivo"].astype(str)

    # ---------------------------------------------
    # 2. Se embeddings já existem → carregar com tqdm
    # ---------------------------------------------
    if os.path.exists(EMB_PATH):
        print("[INFO] Arquivo de embeddings encontrado no disco.")
        print("[INFO] Carregando embeddings com mmap + tqdm...")

        # mmap evita carregar tudo na RAM de uma vez (mais leve)
        emb_mmap = np.load(EMB_PATH, mmap_mode="r")

        # pré-aloca matriz final
        emb_matrix = np.empty(
            (len(emb_mmap), emb_mmap.shape[1]),
            dtype=emb_mmap.dtype
        )

        # barra de progresso ao carregar
        for i in tqdm(range(len(emb_mmap)),
                      desc="Carregando embeddings",
                      ncols=80):
            emb_matrix[i] = emb_mmap[i]

        print(f"[INFO] Embeddings carregados: {emb_matrix.shape}")
        return meta_df, emb_matrix

    # ---------------------------------------------
    # 3. Caso NÃO exista → gerar embeddings uma vez só
    # ---------------------------------------------
    print("[INFO] Nenhum arquivo de embeddings encontrado.")
    print("[INFO] Gerando embeddings usando API (apenas UMA vez)...")

    from openai import OpenAI
    client = OpenAI()

    embeddings = []
    motivos = meta_df["motivo"].tolist()

    for frase in tqdm(motivos, desc="Gerando embeddings", ncols=80):
        try:
            emb = client.embeddings.create(
                model="text-embedding-3-large",
                input=frase
            ).data[0].embedding
            embeddings.append(emb)

        except Exception as e:
            print(f"[ERRO] Falha ao gerar embedding para: {frase}")
            print("Motivo:", e)
            embeddings.append([0] * 1536)  # fallback seguro

    emb_matrix = np.array(embeddings, dtype=np.float32)

    # salvar para uso futuro (zero novo custo de API)
    try:
        np.save(EMB_PATH, emb_matrix)
        print(f"[INFO] Embeddings salvos em: {EMB_PATH}")
    except Exception as e:
        print(f"[ERRO] Falha ao salvar embeddings: {e}")

    return meta_df, emb_matrix

# ============== BUSCAR SIMILARES ==============
def buscar_similares(motivo, meta_df, emb_matrix=None, top_k=8, sim_threshold=0.55):
    resultados = []
    motivos_base = meta_df['motivo'].astype(str).tolist()

    if emb_matrix is not None:
        try:
            emb_m = openai_client.embeddings.create(model="text-embedding-3-small", input=motivo).data[0].embedding
            emb_m = np.array(emb_m).reshape(1, -1)
            sims = cosine_similarity(emb_m, emb_matrix)[0]
            idxs = np.argsort(sims)[::-1][:top_k]
            for i in idxs:
                s = float(sims[i])
                if s >= sim_threshold:
                    resultados.append((motivos_base[i], float(meta_df.iloc[i]['peso_medio']), s))
            return resultados
        except Exception:
            pass

    matches = get_close_matches(motivo, motivos_base, n=top_k, cutoff=0.45)
    for m in matches:
        idx = motivos_base.index(m)
        from difflib import SequenceMatcher
        s = SequenceMatcher(None, motivo, m).ratio()
        resultados.append((m, float(meta_df.iloc[idx]['peso_medio']), s))
    return resultados


# ============== EXTRAIR MOTIVOS DAS NOTÍCIAS RECENTES ==============
def extrair_motivos_ultimos_dias(pasta_json, janela_dias=30, ref_date=None):
    arquivos = glob.glob(os.path.join(pasta_json, "evento_*.json"))
    noticias = {}
    hoje = pd.to_datetime(ref_date).normalize() if ref_date else pd.Timestamp.today().normalize()
    inicio = hoje - pd.Timedelta(days=janela_dias)

    for path in arquivos:
        try:
            with open(path, "r", encoding="utf-8") as f:
                j = json.load(f)
            data_str = j.get("data") or os.path.basename(path).split("_")[-1].replace(".json", "")
            data = pd.to_datetime(data_str).normalize()
            if not (inicio < data <= hoje):
                continue
            motivos = j.get("motivos_identificados") or []
            motivos = [str(m).strip() for m in motivos if m.strip()]
            if motivos:
                noticias.setdefault(data, []).extend(motivos)
        except Exception:
            continue
    return noticias


# ============== CALCULAR PESOS POR NOTÍCIA ==============
def calcular_pesos_por_noticia(noticias_dict, meta_df, emb_matrix=None):
    pesos_por_data = {}
    for data, motivos in noticias_dict.items():
        pesos = []
        for motivo in motivos:
            sims = buscar_similares(motivo, meta_df, emb_matrix=emb_matrix)
            if not sims:
                continue
            numer = sum(p * s for (_, p, s) in sims)
            denom = sum(s for (_, _, s) in sims)
            if denom > 0:
                peso_medio = numer / denom
                pesos.append(peso_medio)
        if pesos:
            pesos_por_data[data] = float(np.mean(pesos))
    return pesos_por_data


# ============== APLICAR CORREÇÃO D+n ==============
def aplicar_correcao_d_plus_1(
    pred_df,
    pesos_por_data,
    motivos_por_data=None,     # <-- NOVO: motivos por data
    janela_test: int = 30,
    meia_vida: float = 4.0,
    dias_propagacao: int = 3
):
    """
    Aplica correção por notícia com:
    - reset por notícia
    - limite de dias úteis de propagação
    - motivo impresso somente no dia D0
    """

    df = pred_df.copy()
    df["Pred_Ajustado"] = df["Pred"].copy()

    dias_uteis = df.index
    test_index = dias_uteis[-janela_test:]

    # notícias ordenadas
    noticias_ordenadas = sorted(pesos_por_data.items(), key=lambda x: pd.to_datetime(x[0]))

    # converte para dias úteis
    noticias_processadas = []
    for data_raw, peso_raw in noticias_ordenadas:
        dt = pd.to_datetime(data_raw)

        # ajusta final de semana → próximo pregão
        pos = dias_uteis.searchsorted(dt)
        if pos >= len(dias_uteis):
            continue
        data_evento = dias_uteis[pos]

        peso = np.clip(float(peso_raw), -1.0, 1.0)
        noticias_processadas.append((data_evento, peso))

    # mapeamento final dia -> evento
    mapa_dia_para_evento = {}

    for data_evento, peso in noticias_processadas:
        idx_evento = dias_uteis.get_loc(data_evento)

        for k in range(dias_propagacao + 1):
            idx_k = idx_evento + k
            if idx_k >= len(dias_uteis):
                break
            dia_k = dias_uteis[idx_k]

            if dia_k not in test_index:
                continue

            # sobrescreve (reset) se outra notícia posterior cobrir o mesmo dia
            mapa_dia_para_evento[dia_k] = (data_evento, peso, k)

    # aplica ajustes
    for dia, (data_evento, peso, k) in mapa_dia_para_evento.items():
        alpha_k = float(np.exp(-k / meia_vida))
        peso_suave = float(np.tanh((peso / 100.0) * alpha_k))

        # ajuste não cumulativo
        df.loc[dia, "Pred_Ajustado"] = df.loc[dia, "Pred"] * (1.0 + peso_suave)

        # motivo no D0
        if k == 0 and motivos_por_data is not None:
            motivos = motivos_por_data.get(pd.to_datetime(data_evento), [])
        else:
            motivos = []

        print(
            f"[AJUSTE] Evento {data_evento.date()} → dia={dia.date()} | "
            f"k={k} | peso={peso:.3f} | meia_vida={meia_vida} | "
            f"alpha_k={alpha_k:.4f} | suave={peso_suave:.6f} | "
            f"motivo={motivos if k == 0 else 'propagação'}"
        )

    return df




# ============== CARREGAR MODELO ==============
def carregar_modelo(model_path):
    torch.serialization.add_safe_globals([])
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    from model_baseline_lstm import LSTMPrice
    model = LSTMPrice(len(ckpt["train_columns"]), 128, 2).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["scaler"], ckpt["target_col"], ckpt["seq_len"], ckpt["train_columns"]


# ============== PREVER ==============
def prever(model, scaler, df, seq_len, target_col, train_cols):
    df_al = df.copy()
    faltantes = [c for c in train_cols if c not in df_al.columns]
    for c in faltantes:
        df_al[c] = 0.0
    df_al = df_al[train_cols]

    arr = scaler.transform(df_al)
    target_idx = train_cols.index(target_col)

    preds_scaled, reals_scaled, dates = [], [], []
    for i in range(len(arr) - seq_len):
        seq = arr[i:i + seq_len, :]
        seq_tensor = torch.tensor(seq[np.newaxis, :, :], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            pred_scaled = model(seq_tensor).cpu().numpy().item()
        preds_scaled.append(pred_scaled)
        reals_scaled.append(arr[i + seq_len, target_idx])
        dates.append(df_al.index[i + seq_len])

    def inverse(values):
        zeros = np.zeros((len(values), len(train_cols)))
        zeros[:, target_idx] = values
        return scaler.inverse_transform(zeros)[:, target_idx]

    df_out = pd.DataFrame({
        "Date": dates,
        "Pred": inverse(np.array(preds_scaled)),
        "Real": inverse(np.array(reals_scaled))
    }).set_index("Date")

    print(f"[INFO] Previsões geradas: {len(df_out)} amostras.")
    return df_out


# ============== PLOTAR ==============
def plotar(pred_df, pesos_por_data, html_path):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df["Real"],
                             mode="lines", name="Preço Real", line=dict(color="black", width=2)))
    fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df["Pred"],
                             mode="lines", name="Previsto (LSTM técnico)", line=dict(color="blue", width=2)))

    ultimos = pred_df.tail(30)
    fig.add_trace(go.Scatter(x=ultimos.index, y=ultimos["Pred_Ajustado"],
                             mode="lines+markers", name="Previsto Ajustado (com notícias)",
                             line=dict(color="orange", width=3),
                             marker=dict(size=6, color="orange")))

    dias_com_peso = [d + timedelta(days=1) for d in pesos_por_data.keys() if d + timedelta(days=1) in pred_df.index[-30:]]
    if dias_com_peso:
        fig.add_trace(go.Scatter(x=dias_com_peso, y=pred_df.loc[dias_com_peso, "Pred_Ajustado"],
                                 mode="markers", name="Dias com impacto D+1",
                                 marker=dict(color="red", size=8, symbol="star")))

    rmse = np.sqrt(mean_squared_error(pred_df["Real"], pred_df["Pred_Ajustado"]))
    r2 = r2_score(pred_df["Real"], pred_df["Pred_Ajustado"])
    titulo = f"Previsão Híbrida com Ajuste Semântico (D+1) — RMSE={rmse:.3f} | R²={r2:.3f}"
    fig.update_layout(title=titulo, xaxis_title="Data", yaxis_title="Preço (R$)",
                      template="plotly_white", width=1200, height=600)
    fig.write_html(html_path, auto_open=False)
    print(f"✅ Gráfico salvo em {html_path}")


# ============== MAIN ==============
def main():
    df = pd.read_csv(CAMINHO_DADOS, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).normalize()
    df = df.ffill().bfill().fillna(0.0)
    model, scaler, target_col, seq_len, cols = carregar_modelo(MODELO_BASELINE)
    pred_df = prever(model, scaler, df, seq_len, target_col, cols)

    meta_df, emb_matrix = carregar_base_frases()
    noticias_30 = extrair_motivos_ultimos_dias(PASTA_JSON, janela_dias=30, ref_date=pred_df.index[-1])
    pesos_30 = calcular_pesos_por_noticia(noticias_30, meta_df, emb_matrix)
    pred_corr = aplicar_correcao_d_plus_1(pred_df, pesos_30, janela_test=30)
    plotar(pred_corr, pesos_30, HTML_SAIDA)

    rmse = np.sqrt(mean_squared_error(pred_corr["Real"], pred_corr["Pred_Ajustado"]))
    r2 = r2_score(pred_corr["Real"], pred_corr["Pred_Ajustado"])
    print(f"[RESULT] Avaliação Final — RMSE={rmse:.4f} | R²={r2:.4f}")


if __name__ == "__main__":
    main()
