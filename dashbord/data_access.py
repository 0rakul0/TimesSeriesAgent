from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.mvp import prever_proximos_3
from utils.project_paths import DATA_DIR, MODELOS_DIR, OUTPUT_NOTICIAS_DIR


@dataclass(frozen=True)
class AssetConfig:
    code: str
    ticker: str
    csv_path: Path
    cluster_path: Path


ASSET_CONFIGS: dict[str, AssetConfig] = {
    "PETR4": AssetConfig(
        code="PETR4",
        ticker="PETR4.SA",
        csv_path=DATA_DIR / "dados_petr4_brent.csv",
        cluster_path=DATA_DIR / "cluster_petr4.csv",
    ),
    "PRIO3": AssetConfig(
        code="PRIO3",
        ticker="PRIO3.SA",
        csv_path=DATA_DIR / "dados_prio3_brent.csv",
        cluster_path=DATA_DIR / "cluster_prio3.csv",
    ),
    "EXXO34": AssetConfig(
        code="EXXO34",
        ticker="EXXO34.SA",
        csv_path=DATA_DIR / "dados_exxo34_brent.csv",
        cluster_path=DATA_DIR / "cluster_exxo34.csv",
    ),
}

BRENT_CONFIG = AssetConfig(
    code="BRENT",
    ticker="BZ=F",
    csv_path=DATA_DIR / "dados_acao_BZ=F_5y.csv",
    cluster_path=DATA_DIR / "cluster_brent.csv",
)


def list_assets() -> list[str]:
    return list(ASSET_CONFIGS)


def get_asset_config(asset_code: str) -> AssetConfig:
    asset_code = asset_code.upper()
    if asset_code == "BRENT":
        return BRENT_CONFIG
    return ASSET_CONFIGS[asset_code]


def repair_text(value):
    if isinstance(value, dict):
        return {key: repair_text(val) for key, val in value.items()}
    if isinstance(value, list):
        return [repair_text(item) for item in value]
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return text

    if any(token in text for token in ("Ã", "â", "�")):
        for encoding in ("latin1", "cp1252"):
            try:
                fixed = text.encode(encoding).decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            if fixed.count("�") <= text.count("�"):
                text = fixed
                break

    return text


def _parse_jsonish_list(value) -> list[str]:
    if isinstance(value, list):
        return [repair_text(str(item)) for item in value]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    text = str(value).strip()
    if not text:
        return []

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, list):
                return [repair_text(str(item)) for item in parsed]
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue

    return [repair_text(text)]


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            "_".join(str(part) for part in col if str(part) and str(part) != "None").strip("_")
            for col in df.columns.to_flat_index()
        ]
    return df


def _find_first_column(df: pd.DataFrame, candidates: list[str]) -> str:
    lower_map = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        key = candidate.lower()
        if key in lower_map:
            return lower_map[key]
    raise KeyError(f"Nenhuma coluna encontrada para {candidates}. Colunas: {list(df.columns)}")


@lru_cache(maxsize=8)
def load_price_history(asset_code: str) -> pd.DataFrame:
    cfg = get_asset_config(asset_code)
    df = pd.read_csv(cfg.csv_path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    df = _flatten_columns(df)

    ticker = cfg.ticker
    ticker_short = ticker.replace(".SA", "")
    columns = {
        "Open": _find_first_column(df, [f"Open_{ticker}", f"Open_{ticker_short}", "Open"]),
        "High": _find_first_column(df, [f"High_{ticker}", f"High_{ticker_short}", "High"]),
        "Low": _find_first_column(df, [f"Low_{ticker}", f"Low_{ticker_short}", "Low"]),
        "Close": _find_first_column(df, [f"Close_{ticker}", f"Close_{ticker_short}", "Close"]),
        "Volume": _find_first_column(df, [f"Volume_{ticker}", f"Volume_{ticker_short}", "Volume"]),
    }

    renamed = df.rename(columns={value: key for key, value in columns.items()})
    out = renamed[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()
    out = out.dropna(subset=["Date", "Close"]).reset_index(drop=True)
    out["RetornoPct"] = out["Close"].pct_change().mul(100.0)
    return out


@lru_cache(maxsize=4)
def load_brent_history() -> pd.DataFrame:
    df = pd.read_csv(BRENT_CONFIG.csv_path)
    df = _flatten_columns(df)
    date_col = _find_first_column(df, ["Date", "Datetime"])
    columns = {
        "Open": _find_first_column(df, ["Open"]),
        "High": _find_first_column(df, ["High"]),
        "Low": _find_first_column(df, ["Low"]),
        "Close": _find_first_column(df, ["Close"]),
        "Volume": _find_first_column(df, ["Volume"]),
    }

    renamed = df.rename(columns={date_col: "Date", **{value: key for key, value in columns.items()}})
    out = renamed[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    out["Date"] = pd.to_datetime(out["Date"], utc=True).dt.tz_convert(None).dt.normalize()
    out = out.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    out["RetornoPct"] = out["Close"].pct_change().mul(100.0)
    return out


def _event_path(asset_code: str, session_date: str | pd.Timestamp) -> Path:
    date_str = pd.to_datetime(session_date).strftime("%Y-%m-%d")
    return OUTPUT_NOTICIAS_DIR / f"evento_{asset_code.upper()}_{date_str}.json"


@lru_cache(maxsize=32)
def load_event_catalog(asset_code: str) -> list[dict]:
    asset_code = asset_code.upper()
    events = []
    for path in sorted(OUTPUT_NOTICIAS_DIR.glob(f"evento_{asset_code}_*.json"), reverse=True):
        try:
            with path.open("r", encoding="utf-8") as handle:
                raw = repair_text(json.load(handle))
        except (OSError, json.JSONDecodeError):
            continue

        session_date = pd.to_datetime(raw.get("data"), errors="coerce")
        if pd.isna(session_date):
            continue

        events.append(
            {
                "date": session_date.normalize(),
                "path": path,
                "sentimento": raw.get("sentimento_do_mercado", "neutro"),
                "motivos": raw.get("motivos_identificados", []) or [],
                "resumo": raw.get("o_que_houve", ""),
                "fontes": raw.get("fontes", []) or [],
                "retorno_no_dia": raw.get("retorno_no_dia"),
            }
        )
    return events


def list_event_dates(asset_code: str) -> list[pd.Timestamp]:
    return [event["date"] for event in load_event_catalog(asset_code)]


def list_brent_event_dates() -> list[pd.Timestamp]:
    return list_event_dates("BRENT")


@lru_cache(maxsize=64)
def load_event_detail(asset_code: str, session_date: str) -> dict | None:
    path = _event_path(asset_code, session_date)
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as handle:
        payload = repair_text(json.load(handle))

    payload["data"] = pd.to_datetime(payload["data"]).normalize()
    payload["motivos_identificados"] = payload.get("motivos_identificados", []) or []
    payload["fontes"] = payload.get("fontes", []) or []
    payload["seq"] = payload.get("seq", {}) or {}
    return payload


def get_row_for_date(price_df: pd.DataFrame, session_date: str | pd.Timestamp) -> pd.Series:
    session_date = pd.to_datetime(session_date).normalize()
    filtered = price_df[price_df["Date"] <= session_date]
    if filtered.empty:
        return price_df.iloc[0]
    return filtered.iloc[-1]


def build_fifty_day_mask(asset_code: str, session_date: str | pd.Timestamp) -> pd.DataFrame:
    price_df = load_price_history(asset_code)
    session_date = pd.to_datetime(session_date).normalize()
    window = price_df[price_df["Date"] <= session_date].tail(50).copy()
    if window.empty:
        return window

    base = float(window["Close"].iloc[0])
    window["CloseNorm"] = window["Close"] / base * 100.0 if base else 100.0
    window["MediaMovel5"] = window["Close"].rolling(5, min_periods=1).mean()
    return window


def build_daily_focus_window(asset_code: str, session_date: str | pd.Timestamp, days: int = 10) -> pd.DataFrame:
    price_df = load_price_history(asset_code)
    session_date = pd.to_datetime(session_date).normalize()
    return price_df[price_df["Date"] <= session_date].tail(days).copy()


def fetch_intraday_series(asset_code: str, session_date: str | pd.Timestamp) -> tuple[pd.DataFrame, str | None]:
    cfg = get_asset_config(asset_code)
    session_date = pd.to_datetime(session_date).normalize()
    end_date = session_date + pd.Timedelta(days=1)

    try:
        candles = yf.download(
            cfg.ticker,
            start=session_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval="15m",
            auto_adjust=False,
            progress=False,
        )
    except Exception as exc:  # pragma: no cover
        return pd.DataFrame(), f"Falha ao consultar Yahoo Finance: {exc}"

    if candles.empty:
        return pd.DataFrame(), "Yahoo Finance nao retornou candles intradiarios para esta data."

    candles = _flatten_columns(candles.reset_index())
    time_col = _find_first_column(candles, ["Datetime", "Date"])
    close_col = _find_first_column(candles, [f"Close_{cfg.ticker}", "Close"])
    high_col = _find_first_column(candles, [f"High_{cfg.ticker}", "High"])
    low_col = _find_first_column(candles, [f"Low_{cfg.ticker}", "Low"])
    open_col = _find_first_column(candles, [f"Open_{cfg.ticker}", "Open"])
    volume_col = _find_first_column(candles, [f"Volume_{cfg.ticker}", "Volume"])

    out = candles.rename(
        columns={
            time_col: "Datetime",
            open_col: "Open",
            high_col: "High",
            low_col: "Low",
            close_col: "Close",
            volume_col: "Volume",
        }
    )[["Datetime", "Open", "High", "Low", "Close", "Volume"]].copy()
    out["Datetime"] = pd.to_datetime(out["Datetime"]).dt.tz_localize(None)
    out = out[out["Datetime"].dt.normalize() == session_date].reset_index(drop=True)

    if out.empty:
        return pd.DataFrame(), "O provedor nao disponibilizou a serie intradiaria filtrada para esta sessao."

    return out, None


def fetch_brent_intraday(session_date: str | pd.Timestamp) -> tuple[pd.DataFrame, str | None]:
    return fetch_intraday_series("BRENT", session_date)


@lru_cache(maxsize=8)
def load_cluster_table(asset_code: str) -> pd.DataFrame:
    cfg = get_asset_config(asset_code)
    if not cfg.cluster_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(cfg.cluster_path)
    for column in ("frase_exemplo", "ativo_cluster"):
        if column in df.columns:
            df[column] = df[column].map(repair_text)

    df["frases_originais_list"] = (
        df["frases_originais"].map(_parse_jsonish_list)
        if "frases_originais" in df.columns
        else [[] for _ in range(len(df))]
    )
    df["event_dates_list"] = (
        df["event_dates"].map(_parse_jsonish_list)
        if "event_dates" in df.columns
        else [[] for _ in range(len(df))]
    )
    df["search_text"] = df.apply(
        lambda row: " ".join(
            [repair_text(str(row.get("frase_exemplo", "")))] + [repair_text(text) for text in row["frases_originais_list"]]
        ).strip(),
        axis=1,
    )
    return df


def match_cluster(asset_code: str, motivos: list[str]) -> dict | None:
    clusters = load_cluster_table(asset_code)
    motivos = [repair_text(motivo) for motivo in motivos if str(motivo).strip()]
    if clusters.empty or not motivos:
        return None

    corpus = clusters["search_text"].tolist() + motivos
    vectorizer = TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(corpus)
    cluster_matrix = matrix[: len(clusters)]

    best_row = None
    best_motivo = None
    best_similarity = -1.0
    for idx, motivo in enumerate(motivos):
        motivo_vector = matrix[len(clusters) + idx]
        sims = cosine_similarity(motivo_vector, cluster_matrix).flatten()
        cluster_idx = int(np.argmax(sims))
        similarity = float(sims[cluster_idx])
        if similarity > best_similarity:
            best_similarity = similarity
            best_row = clusters.iloc[cluster_idx]
            best_motivo = motivo

    if best_row is None:
        return None

    seq_values = []
    for idx in range(6):
        key = f"seq_d{idx}"
        value = best_row.get(key)
        seq_values.append(float(value) if pd.notna(value) else np.nan)

    return {
        "cluster_id": int(best_row["cluster"]),
        "motivo_referencia": best_motivo,
        "similaridade": best_similarity,
        "frase_exemplo": repair_text(str(best_row.get("frase_exemplo", ""))),
        "n_eventos": int(best_row.get("n_eventos", 0)),
        "n_motivos_unicos": int(best_row.get("n_motivos_unicos", 0)),
        "seq_values": seq_values,
        "seq_map": {f"D+{idx}": seq_values[idx] for idx in range(len(seq_values))},
        "frases_originais": best_row.get("frases_originais_list", []),
        "event_dates": best_row.get("event_dates_list", []),
    }


def build_similar_news_table(cluster_match: dict | None, motivos: list[str]) -> pd.DataFrame:
    if not cluster_match:
        return pd.DataFrame(columns=["Data", "Noticia Semelhante", "Similaridade"])

    phrases = cluster_match.get("frases_originais", [])
    if not phrases:
        return pd.DataFrame(columns=["Data", "Noticia Semelhante", "Similaridade"])

    dates = cluster_match.get("event_dates", [])
    reference = " ".join(repair_text(motivo) for motivo in motivos if str(motivo).strip())
    if not reference:
        reference = repair_text(cluster_match.get("frase_exemplo", ""))

    corpus = phrases + [reference]
    vectorizer = TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(corpus)
    similarities = cosine_similarity(matrix[-1], matrix[:-1]).flatten()

    rows = []
    for idx, phrase in enumerate(phrases):
        rows.append(
            {
                "Data": dates[idx] if idx < len(dates) else "",
                "Noticia Semelhante": repair_text(phrase),
                "Similaridade": round(float(similarities[idx]), 3),
            }
        )

    return pd.DataFrame(rows).sort_values(["Similaridade", "Data"], ascending=[False, False]).reset_index(drop=True)


def _best_model_row(asset_code: str) -> pd.Series:
    resultados = pd.read_csv(DATA_DIR / "resultado_comparacao_modelos.csv")
    subset = resultados[resultados["Ativo"].astype(str).str.contains(asset_code, na=False)].copy()
    if subset.empty:
        raise ValueError(f"Nenhum modelo encontrado para {asset_code}.")
    return subset.sort_values("RMSE_Hibrido").iloc[0]


@lru_cache(maxsize=64)
def load_base_projection(asset_code: str, session_date: str) -> dict:
    best_row = _best_model_row(asset_code)
    model_type = str(best_row["Modelo"]).strip().lower()
    model_path = MODELOS_DIR / f"{model_type}_{asset_code.lower()}.pt"

    cfg = get_asset_config(asset_code)
    full_df = pd.read_csv(cfg.csv_path, parse_dates=["Date"]).sort_values("Date")
    cutoff = pd.to_datetime(session_date).normalize()
    filtered = full_df[full_df["Date"].dt.normalize() <= cutoff].copy()
    if filtered.empty:
        filtered = full_df.copy()

    latest_available = full_df["Date"].max().normalize()
    if cutoff >= latest_available:
        preds, df_pred_full, last_hist_pred = prever_proximos_3(str(model_path), str(cfg.csv_path), model_type)
    else:
        temp_path = None
        with NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8", newline="") as handle:
            filtered.to_csv(handle.name, index=False)
            temp_path = handle.name
        try:
            preds, df_pred_full, last_hist_pred = prever_proximos_3(str(model_path), temp_path, model_type)
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

    return {
        "model_type": model_type,
        "scale": float(best_row.get("Scale_Selecionado", 0.4)),
        "preds_base": [float(value) for value in preds],
        "df_pred_full": df_pred_full.copy(),
        "last_hist_pred": float(last_hist_pred),
    }


def compute_projection(asset_code: str, cluster_match: dict | None, session_date: str) -> dict:
    base = load_base_projection(asset_code, session_date)
    price_df = load_price_history(asset_code)
    reference_row = get_row_for_date(price_df, session_date)
    last_close = float(reference_row["Close"])

    preds_base = [float(value) for value in base["preds_base"]]
    preds_adjusted = preds_base[:]
    impacts = [0.0, 0.0, 0.0]
    similarity = 0.0

    if cluster_match:
        similarity = float(cluster_match.get("similaridade", 0.0))
        seq_values = cluster_match.get("seq_values", [])
        impacts = []
        for idx in range(1, 4):
            impact = seq_values[idx] if idx < len(seq_values) else np.nan
            impacts.append(float(impact) if pd.notna(impact) else 0.0)

        preds_adjusted = [
            pred * (1 + base["scale"] * similarity * (impact / 100.0))
            for pred, impact in zip(preds_base, impacts)
        ]

    forecast_table = pd.DataFrame(
        {
            "Horizonte": ["D+1", "D+2", "D+3"],
            "Previsao Base": preds_base,
            "Previsao Ajustada": preds_adjusted,
            "Impacto Medio (%)": impacts,
        }
    )

    return {
        "modelo": base["model_type"].upper(),
        "scale": base["scale"],
        "similaridade": similarity,
        "last_close": last_close,
        "preds_base": preds_base,
        "preds_ajustadas": preds_adjusted,
        "forecast_table": forecast_table,
        "last_hist_pred": base["last_hist_pred"],
        "projecao_ajustada_d1": preds_adjusted[0],
        "projecao_pct_d1": ((preds_adjusted[0] / last_close) - 1.0) * 100.0 if last_close else 0.0,
        "maximo_projetado": max([last_close] + preds_adjusted),
    }


def load_relevant_brent_event(session_date: str | pd.Timestamp, max_gap_days: int = 7) -> dict | None:
    target = pd.to_datetime(session_date).normalize()
    events = load_event_catalog("BRENT")
    if not events:
        return None

    exact = [event for event in events if event["date"] == target]
    if exact:
        return load_event_detail("BRENT", exact[0]["date"].strftime("%Y-%m-%d"))

    prior = [event for event in events if event["date"] <= target]
    if not prior:
        return None

    chosen = prior[0]
    gap = (target - chosen["date"]).days
    if gap > max_gap_days:
        return None
    return load_event_detail("BRENT", chosen["date"].strftime("%Y-%m-%d"))


def build_asset_brent_window(asset_code: str, session_date: str | pd.Timestamp, days: int = 50) -> pd.DataFrame:
    asset_df = load_price_history(asset_code)[["Date", "Close"]].rename(columns={"Close": "Close_Asset"})
    brent_df = load_brent_history()[["Date", "Close"]].rename(columns={"Close": "Close_Brent"})

    target = pd.to_datetime(session_date).normalize()
    merged = asset_df.merge(brent_df, on="Date", how="inner")
    merged = merged[merged["Date"] <= target].tail(days).copy()
    if merged.empty:
        return merged

    asset_base = float(merged["Close_Asset"].iloc[0])
    brent_base = float(merged["Close_Brent"].iloc[0])
    merged["AssetNorm"] = merged["Close_Asset"] / asset_base * 100.0 if asset_base else 100.0
    merged["BrentNorm"] = merged["Close_Brent"] / brent_base * 100.0 if brent_base else 100.0
    return merged


def list_brent_events_in_window(session_date: str | pd.Timestamp, lookback_days: int = 45) -> list[dict]:
    target = pd.to_datetime(session_date).normalize()
    start = target - pd.Timedelta(days=lookback_days)
    events = []
    for event in load_event_catalog("BRENT"):
        if start <= event["date"] <= target:
            detail = load_event_detail("BRENT", event["date"].strftime("%Y-%m-%d"))
            if detail:
                events.append(detail)
    return sorted(events, key=lambda item: item["data"])


def search_event_matches(asset_code: str, query: str, limit: int = 12) -> list[dict]:
    query = repair_text(query).strip().lower()
    if not query:
        return []

    tokens = [token for token in query.split() if token]
    rows = []
    for source_asset in (asset_code.upper(), "BRENT"):
        for event in load_event_catalog(source_asset):
            motivos = [repair_text(motivo) for motivo in event.get("motivos", [])]
            resumo = repair_text(event.get("resumo", ""))
            fontes = [repair_text(fonte) for fonte in event.get("fontes", [])]
            searchable = " ".join([source_asset, resumo, *motivos, *fontes]).lower()

            rows.append(
                {
                    "asset": source_asset,
                    "date": event["date"].strftime("%Y-%m-%d"),
                    "retorno_no_dia": event.get("retorno_no_dia"),
                    "resumo": resumo,
                    "motivos": motivos,
                    "fontes": fontes,
                    "search_text": searchable,
                }
            )

    if not rows:
        return []

    lexical_corpus = [row["search_text"] for row in rows] + [query]
    vectorizer = TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(lexical_corpus)
    query_vector = matrix[-1]
    lexical_scores = cosine_similarity(query_vector, matrix[:-1]).flatten()

    for idx, row in enumerate(rows):
        overlap = sum(1 for token in tokens if token in row["search_text"])
        phrase_bonus = 0.12 if query in row["search_text"] else 0.0
        asset_bonus = 0.06 if row["asset"] == asset_code.upper() else 0.03
        rows[idx]["lexical_score"] = float(lexical_scores[idx])
        rows[idx]["keyword_score"] = overlap / max(len(tokens), 1)
        rows[idx]["stage1_score"] = rows[idx]["lexical_score"] + 0.18 * rows[idx]["keyword_score"] + phrase_bonus + asset_bonus

    stage1 = sorted(rows, key=lambda item: (item["stage1_score"], item["date"]), reverse=True)[: max(limit * 3, 20)]

    docs_for_rerank = []
    for item in stage1:
        doc_text = " ".join([item["resumo"], *item["motivos"], *item["fontes"]]).strip()
        docs_for_rerank.append(doc_text or item["search_text"])

    rerank_vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="char_wb",
        ngram_range=(3, 5),
    )
    rerank_matrix = rerank_vectorizer.fit_transform(docs_for_rerank + [query])
    rerank_scores = cosine_similarity(rerank_matrix[-1], rerank_matrix[:-1]).flatten().tolist()

    ranked = []
    max_date = max(pd.to_datetime(item["date"]) for item in stage1)
    min_date = min(pd.to_datetime(item["date"]) for item in stage1)
    date_span = max((max_date - min_date).days, 1)

    for item, semantic_score in zip(stage1, rerank_scores):
        recency = (pd.to_datetime(item["date"]) - min_date).days / date_span
        combined = 0.50 * item["stage1_score"] + 0.40 * float(semantic_score) + 0.10 * float(recency)
        ranked.append(
            {
                **item,
                "semantic_score": float(semantic_score),
                "score": combined,
            }
        )

    ranked.sort(key=lambda item: (item["score"], item["date"]), reverse=True)
    return ranked[:limit]
