from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dashbord.data_access import (  # noqa: E402
    build_asset_brent_window,
    build_daily_focus_window,
    build_fifty_day_mask,
    build_similar_news_table,
    compute_projection,
    fetch_brent_intraday,
    fetch_intraday_series,
    get_row_for_date,
    list_assets,
    list_brent_events_in_window,
    list_event_dates,
    load_brent_history,
    load_event_detail,
    load_price_history,
    load_relevant_brent_event,
    match_cluster,
    search_event_matches,
)
from dashbord.knowledge_base import answer_question  # noqa: E402


def configure_page() -> None:
    st.set_page_config(
        page_title="TimesSeriesAgent Dashboard",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(
        """
        <style>
            :root {
                --paper: #f7f3eb;
                --panel: rgba(255, 251, 245, 0.84);
                --ink: #1f2a37;
                --muted: #5f6b7a;
                --accent: #0f766e;
                --accent-soft: rgba(15, 118, 110, 0.10);
                --warn: #b45309;
                --border: rgba(31, 42, 55, 0.10);
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(15, 118, 110, 0.16), transparent 32%),
                    radial-gradient(circle at top right, rgba(180, 83, 9, 0.12), transparent 28%),
                    linear-gradient(180deg, #f9f5ee 0%, #f2ece2 100%);
                color: var(--ink);
            }

            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }

            .hero {
                padding: 1.4rem 1.6rem;
                border: 1px solid var(--border);
                border-radius: 24px;
                background: linear-gradient(135deg, rgba(255,255,255,0.90), rgba(247,243,235,0.88));
                box-shadow: 0 18px 45px rgba(64, 49, 24, 0.08);
                margin-bottom: 1rem;
            }

            .hero h1 {
                margin: 0;
                font-size: 2.2rem;
                letter-spacing: -0.03em;
            }

            .hero p {
                margin: 0.55rem 0 0 0;
                color: var(--muted);
                max-width: 70rem;
            }

            .panel {
                padding: 1rem 1.1rem;
                border-radius: 22px;
                border: 1px solid var(--border);
                background: var(--panel);
                box-shadow: 0 14px 32px rgba(64, 49, 24, 0.06);
                margin-bottom: 1rem;
            }

            .chip-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.45rem;
                margin-top: 0.6rem;
            }

            .chip {
                display: inline-block;
                padding: 0.28rem 0.65rem;
                border-radius: 999px;
                background: var(--accent-soft);
                color: var(--accent);
                font-size: 0.86rem;
                border: 1px solid rgba(15, 118, 110, 0.14);
            }

            .subtle {
                color: var(--muted);
                font-size: 0.92rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def plot_intraday_chart(
    asset_code: str,
    intraday_df: pd.DataFrame,
    fallback_df: pd.DataFrame,
    brent_intraday_df: pd.DataFrame,
    brent_fallback_df: pd.DataFrame,
) -> go.Figure:
    fig = go.Figure()

    if not intraday_df.empty:
        fig.add_trace(
            go.Scatter(
                x=intraday_df["Datetime"],
                y=intraday_df["Close"],
                mode="lines+markers",
                line=dict(color="#0f766e", width=3),
                marker=dict(size=5, color="#0f766e"),
                fill="tozeroy",
                fillcolor="rgba(15, 118, 110, 0.10)",
                name=f"{asset_code} 15m",
            )
        )
        if not brent_intraday_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=brent_intraday_df["Datetime"],
                    y=brent_intraday_df["Close"],
                    mode="lines",
                    line=dict(color="#b45309", width=2, dash="dot"),
                    name="Brent 15m",
                    yaxis="y2",
                )
            )
        fig.update_layout(title="Serie do ativo no dia", xaxis_title="Horario", yaxis_title="Preco")
    else:
        fig.add_trace(
            go.Candlestick(
                x=fallback_df["Date"],
                open=fallback_df["Open"],
                high=fallback_df["High"],
                low=fallback_df["Low"],
                close=fallback_df["Close"],
                increasing_line_color="#0f766e",
                decreasing_line_color="#b45309",
                name=asset_code,
            )
        )
        if not brent_fallback_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=brent_fallback_df["Date"],
                    y=brent_fallback_df["Close"],
                    mode="lines+markers",
                    line=dict(color="#b45309", width=2, dash="dot"),
                    marker=dict(size=5),
                    name="Brent",
                    yaxis="y2",
                )
            )
        fig.update_layout(title="Serie diaria de apoio", xaxis_title="Data", yaxis_title="Preco")

    fig.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.55)",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        yaxis2=dict(title="Brent", overlaying="y", side="right", showgrid=False),
    )
    return fig


def plot_mask_chart(mask_df: pd.DataFrame, session_date: pd.Timestamp) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=mask_df["Date"],
            y=mask_df["CloseNorm"],
            mode="lines",
            line=dict(color="#1f4e79", width=3),
            name="Close normalizado",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=mask_df["Date"],
            y=mask_df["MediaMovel5"],
            mode="lines",
            line=dict(color="#b45309", width=2, dash="dot"),
            name="Media movel 5",
            yaxis="y2",
        )
    )
    fig.add_vline(x=session_date, line_width=2, line_dash="dash", line_color="#b45309")
    fig.update_layout(
        title="Mascara de comportamento em 50 pregoes",
        template="plotly_white",
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.55)",
        yaxis=dict(title="Base 100"),
        yaxis2=dict(title="Preco", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    return fig


def plot_cluster_behavior(cluster_match: dict | None) -> go.Figure:
    seq_values = cluster_match.get("seq_values", []) if cluster_match else []
    labels = [f"D+{idx}" for idx in range(len(seq_values))]
    cleaned = [float(value) if pd.notna(value) else 0.0 for value in seq_values]
    colors = ["#0f766e" if value >= 0 else "#b45309" for value in cleaned]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=cleaned,
            marker_color=colors,
            text=[f"{value:.2f}%" for value in cleaned],
            textposition="outside",
            name="Impacto medio",
        )
    )
    fig.update_layout(
        title="Cluster de comportamento associado",
        template="plotly_white",
        height=280,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.55)",
        yaxis_title="Impacto medio (%)",
        xaxis_title="Horizonte",
    )
    return fig


def plot_asset_brent_context(compare_df: pd.DataFrame, brent_events: list[dict]) -> go.Figure:
    fig = go.Figure()
    if compare_df.empty:
        fig.update_layout(
            title="Ativo x Brent no contexto recente",
            template="plotly_white",
            height=300,
            margin=dict(l=20, r=20, t=60, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.55)",
        )
        return fig

    fig.add_trace(
        go.Scatter(
            x=compare_df["Date"],
            y=compare_df["AssetNorm"],
            mode="lines+markers",
            line=dict(color="#0f766e", width=3),
            marker=dict(size=5),
            name="Ativo base 100",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=compare_df["Date"],
            y=compare_df["BrentNorm"],
            mode="lines+markers",
            line=dict(color="#b45309", width=3, dash="dot"),
            marker=dict(size=5),
            name="Brent base 100",
        )
    )

    for event in brent_events[-4:]:
        event_date = pd.to_datetime(event["data"])
        fig.add_vline(x=event_date, line_color="rgba(180,83,9,0.35)", line_dash="dash", line_width=1)

    fig.update_layout(
        title="Ativo x Brent no contexto recente",
        template="plotly_white",
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.55)",
        yaxis_title="Base 100",
        xaxis_title="Data",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    return fig


def render_metric_cards(intraday_df: pd.DataFrame, price_row: pd.Series, projection: dict) -> None:
    if not intraday_df.empty:
        min_day = float(intraday_df["Low"].min())
        max_day = float(intraday_df["High"].max())
    else:
        min_day = float(price_row["Low"])
        max_day = float(price_row["High"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Minima do dia", f"R$ {min_day:,.2f}", delta=f"Fechamento ref. R$ {float(price_row['Close']):,.2f}")
    col2.metric("Maxima do dia", f"R$ {max_day:,.2f}", delta=f"Max. projetado R$ {projection['maximo_projetado']:,.2f}")
    col3.metric(
        "Projecao ajustada",
        f"R$ {projection['projecao_ajustada_d1']:,.2f}",
        delta=f"{projection['projecao_pct_d1']:+.2f}% em D+1",
    )


def render_event_panel(event_detail: dict, cluster_match: dict | None) -> None:
    sentimento = str(event_detail.get("sentimento_do_mercado", "neutro")).capitalize()
    fontes = ", ".join(event_detail.get("fontes", [])) or "Sem fontes registradas"
    retorno = event_detail.get("retorno_no_dia")

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Evento do dia")
    st.caption(
        f"{pd.to_datetime(event_detail['data']).strftime('%d/%m/%Y')} · "
        f"Sentimento {sentimento}"
        + (f" · Retorno {float(retorno):+.2f}%" if retorno is not None else "")
    )
    st.write(event_detail.get("o_que_houve", "Sem resumo disponivel."))
    st.markdown(f"**Fontes capturadas:** {fontes}")

    motivos = event_detail.get("motivos_identificados", [])
    if motivos:
        chips = "".join(f'<span class="chip">{motivo}</span>' for motivo in motivos)
        st.markdown(f'<div class="chip-row">{chips}</div>', unsafe_allow_html=True)

    if cluster_match:
        st.markdown("---")
        st.markdown(f"**Cluster associado:** `{cluster_match['cluster_id']}`")
        st.markdown(cluster_match["frase_exemplo"])
        st.caption(f"Similaridade {cluster_match['similaridade']:.3f} · {cluster_match['n_eventos']} eventos historicos no cluster")
    st.markdown("</div>", unsafe_allow_html=True)


def render_brent_panel(brent_event: dict | None, brent_row: pd.Series, brent_events: list[dict]) -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Contexto do Brent")
    st.caption(
        f"Fechamento de referencia: US$ {float(brent_row['Close']):,.2f} · "
        f"Min: US$ {float(brent_row['Low']):,.2f} · Max: US$ {float(brent_row['High']):,.2f}"
    )

    if brent_event:
        st.markdown(
            f"**Evento relevante em {pd.to_datetime(brent_event['data']).strftime('%d/%m/%Y')}:** "
            f"{brent_event.get('o_que_houve', '')}"
        )
        motivos = brent_event.get("motivos_identificados", [])
        if motivos:
            chips = "".join(f'<span class="chip">{motivo}</span>' for motivo in motivos)
            st.markdown(f'<div class="chip-row">{chips}</div>', unsafe_allow_html=True)
    else:
        st.info("Nao encontrei um evento de Brent suficientemente proximo da data selecionada.")

    if brent_events:
        rows = []
        for event in brent_events[-4:]:
            rows.append(
                {
                    "Data": pd.to_datetime(event["data"]).strftime("%Y-%m-%d"),
                    "Motivos": ", ".join(event.get("motivos_identificados", [])[:2]),
                    "Retorno (%)": round(float(event.get("retorno_no_dia", 0.0)), 2),
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_search_results(search_results: list[dict]) -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Resultados da busca de eventos")
    rows = []
    for item in search_results:
        rows.append(
            {
                "Origem": item["asset"],
                "Data": item["date"],
                "Retorno (%)": round(float(item["retorno_no_dia"]), 2) if item["retorno_no_dia"] is not None else None,
                "Motivos": ", ".join(item["motivos"][:2]),
                "Resumo": item["resumo"][:140] + ("..." if len(item["resumo"]) > 140 else ""),
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=260)
    st.markdown("</div>", unsafe_allow_html=True)


def render_chat(asset_code: str, event_detail: dict, cluster_match: dict | None, projection: dict) -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Chat com a base de conhecimento")
    st.caption("Pergunte sobre o projeto, o ativo, o cluster atual ou a previsao ajustada.")

    if "dashboard_chat_history" not in st.session_state:
        st.session_state.dashboard_chat_history = []

    for message in st.session_state.dashboard_chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ex.: por que este cluster costuma reagir mal em D+2?")
    if prompt:
        st.session_state.dashboard_chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        answer = answer_question(prompt, asset_code, event_detail, cluster_match, projection)
        answer_text = answer["answer"]
        if answer["sources"]:
            answer_text += "\n\nFontes consultadas: " + ", ".join(answer["sources"])

        st.session_state.dashboard_chat_history.append({"role": "assistant", "content": answer_text})
        with st.chat_message("assistant"):
            st.markdown(answer_text)

    st.markdown("</div>", unsafe_allow_html=True)


def build_fallback_event_detail(asset_code: str, session_date: str) -> dict:
    return {
        "data": pd.to_datetime(session_date).normalize(),
        "ativo": asset_code,
        "sentimento_do_mercado": "neutro",
        "fontes": [],
        "motivos_identificados": [],
        "o_que_houve": (
            f"Nao existe um evento estruturado do ativo {asset_code} nessa data na base local. "
            "Use os graficos e o contexto do Brent para inspecionar o comportamento do mercado."
        ),
        "retorno_no_dia": None,
        "seq": {asset_code: []},
    }


def main() -> None:
    configure_page()

    st.markdown(
        """
        <div class="hero">
            <h1>TimesSeriesAgent Dashboard</h1>
            <p>
                Painel Streamlit para acompanhar o evento do dia, a serie do ativo,
                a mascara de 50 pregoes, a projecao ajustada do modelo e o cluster de noticias
                historicamente semelhante.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    controls = st.columns([1, 1, 1.6])
    asset_code = controls[0].selectbox("Ativo", list_assets(), index=1)
    event_dates = list_event_dates(asset_code)
    if not event_dates:
        st.error("Nao encontrei eventos estruturados para este ativo em `output_noticias/`.")
        return

    date_labels = [date.strftime("%Y-%m-%d") for date in event_dates]
    search_query = controls[2].text_input(
        "Pesquisar evento",
        value=st.session_state.get("dashboard_search_query", ""),
        placeholder="Ex.: OPEP, oferta global, guerra, estoque, Ormuz",
    )
    st.session_state.dashboard_search_query = search_query

    search_results = search_event_matches(asset_code, search_query) if search_query.strip() else []
    result_options = {}
    if search_results:
        for item in search_results:
            label = (
                f"{item['asset']} · {item['date']} · "
                f"{', '.join(item['motivos'][:2]) or item['resumo'][:80]}"
            )
            result_options[label] = item

        selected_result_label = controls[1].selectbox(
            "Resultado da busca",
            list(result_options.keys()),
            key=f"search_result_{asset_code}",
        )
        selected_label = result_options[selected_result_label]["date"]
    else:
        selected_label = controls[1].selectbox("Data do evento", date_labels, index=0)
        controls[2].markdown(
            '<div class="panel subtle">A serie intradiaria usa Yahoo Finance em 15 minutos quando a sessao estiver disponivel. '
            "Para datas antigas sem candles intradiarios, o painel faz fallback para uma visao diaria.</div>",
            unsafe_allow_html=True,
        )

    event_detail = load_event_detail(asset_code, selected_label)
    if not event_detail:
        event_detail = build_fallback_event_detail(asset_code, selected_label)

    chat_context = f"{asset_code}:{selected_label}"
    if st.session_state.get("dashboard_chat_context") != chat_context:
        st.session_state.dashboard_chat_context = chat_context
        st.session_state.dashboard_chat_history = []

    price_df = load_price_history(asset_code)
    price_row = get_row_for_date(price_df, selected_label)
    intraday_df, intraday_notice = fetch_intraday_series(asset_code, selected_label)
    fallback_df = build_daily_focus_window(asset_code, selected_label)
    mask_df = build_fifty_day_mask(asset_code, selected_label)
    brent_history = load_brent_history()
    brent_row = get_row_for_date(brent_history, selected_label)
    brent_intraday_df, brent_intraday_notice = fetch_brent_intraday(selected_label)
    brent_fallback_df = brent_history[brent_history["Date"] <= pd.to_datetime(selected_label)].tail(10).copy()
    brent_event = load_relevant_brent_event(selected_label)
    brent_events = list_brent_events_in_window(selected_label)
    compare_df = build_asset_brent_window(asset_code, selected_label)
    cluster_match = match_cluster(asset_code, event_detail.get("motivos_identificados", []))
    similar_news_df = build_similar_news_table(cluster_match, event_detail.get("motivos_identificados", []))
    projection = compute_projection(asset_code, cluster_match, selected_label)

    left_col, right_col = st.columns([1.35, 1], gap="large")

    with left_col:
        if search_results:
            render_search_results(search_results)

        render_metric_cards(intraday_df, price_row, projection)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            plot_intraday_chart(asset_code, intraday_df, fallback_df, brent_intraday_df, brent_fallback_df),
            width="stretch",
            config={"displayModeBar": False},
        )
        if intraday_notice:
            st.caption(intraday_notice)
        if brent_intraday_notice:
            st.caption(f"Brent: {brent_intraday_notice}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(plot_mask_chart(mask_df, pd.to_datetime(selected_label)), width="stretch", config={"displayModeBar": False})
        st.dataframe(
            projection["forecast_table"].style.format(
                {
                    "Previsao Base": "R$ {:.2f}",
                    "Previsao Ajustada": "R$ {:.2f}",
                    "Impacto Medio (%)": "{:.2f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            f"Modelo selecionado: {projection['modelo']} · Scale operacional: {projection['scale']:.2f} · "
            f"Similaridade do cluster: {projection['similaridade']:.3f}"
        )
        st.markdown("</div>", unsafe_allow_html=True)

        render_chat(asset_code, event_detail, cluster_match, projection)

    with right_col:
        render_event_panel(event_detail, cluster_match)

        render_brent_panel(brent_event, brent_row, brent_events)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(plot_asset_brent_context(compare_df, brent_events), width="stretch", config={"displayModeBar": False})
        st.caption("As linhas pontilhadas marcam os eventos recentes de Brent capturados pela base.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(plot_cluster_behavior(cluster_match), width="stretch", config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Noticias semelhantes")
        if similar_news_df.empty:
            st.info("Nao encontrei noticias semelhantes para o cluster atual.")
        else:
            st.dataframe(
                similar_news_df.style.format({"Similaridade": "{:.3f}"}),
                width="stretch",
                hide_index=True,
                height=420,
            )
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
