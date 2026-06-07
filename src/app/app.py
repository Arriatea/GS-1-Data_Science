from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Global Solution - Focos de Calor",
    layout="wide",
)

RAW_PATH = Path("data/raw/viirs-snpp_2024_Brazil.csv")
PROCESSED_DIR = Path("data/processed")

NUMERIC_COLUMNS = ["frp", "bright_ti4", "bright_ti5", "scan", "track"]
FILTER_COLUMNS = ["mes", "daynight", "confidence", "type"]


def _raw_dtypes() -> dict[str, str]:
    return {
        "latitude": "float32",
        "longitude": "float32",
        "bright_ti4": "float32",
        "scan": "float32",
        "track": "float32",
        "acq_time": "int16",
        "satellite": "category",
        "instrument": "category",
        "confidence": "category",
        "version": "category",
        "bright_ti5": "float32",
        "frp": "float32",
        "daynight": "category",
        "type": "int8",
    }


def _find_processed_file() -> Path | None:
    if not PROCESSED_DIR.exists():
        return None

    candidates = (
        sorted(PROCESSED_DIR.glob("*.parquet"))
        + sorted(PROCESSED_DIR.glob("*.csv"))
    )
    return candidates[0] if candidates else None


def _ensure_date_columns(data: pd.DataFrame) -> pd.DataFrame:
    if "acq_date" in data.columns:
        data["acq_date"] = pd.to_datetime(data["acq_date"], errors="coerce")
        if "mes" not in data.columns:
            data["mes"] = data["acq_date"].dt.month
        if "dia" not in data.columns:
            data["dia"] = data["acq_date"].dt.day
    return data


def _remove_outliers_iqr(data: pd.DataFrame) -> pd.DataFrame:
    columns = [col for col in ["frp", "bright_ti4", "bright_ti5"] if col in data.columns]
    if not columns:
        return data.copy()

    keep_rows = pd.Series(True, index=data.index)
    for column in columns:
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = (data[column] < lower) | (data[column] > upper)
        keep_rows = keep_rows & ~outliers

    return data.loc[keep_rows].copy()


@st.cache_data(show_spinner="Carregando e preparando os dados...")
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    processed_file = _find_processed_file()

    if processed_file:
        if processed_file.suffix == ".parquet":
            original = pd.read_parquet(processed_file)
        else:
            original = pd.read_csv(processed_file)
        original = _ensure_date_columns(original)
        return original.copy(), original.copy(), f"Base tratada: {processed_file}"

    if not RAW_PATH.exists():
        return pd.DataFrame(), pd.DataFrame(), "Base não encontrada"

    original = pd.read_csv(
        RAW_PATH,
        parse_dates=["acq_date"],
        dtype=_raw_dtypes(),
    )
    original = _ensure_date_columns(original)
    treated = _remove_outliers_iqr(original)
    return original, treated, f"Base bruta com tratamento no app: {RAW_PATH}"


def apply_filters(data: pd.DataFrame, selected_filters: dict[str, list]) -> pd.DataFrame:
    filtered = data.copy()

    for column, values in selected_filters.items():
        if column in filtered.columns and values:
            filtered = filtered[filtered[column].isin(values)]

    return filtered


def frequency_table(data: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in data.columns or data.empty:
        return pd.DataFrame(columns=["categoria", "quantidade", "percentual"])

    counts = data[column].value_counts(dropna=False)
    percentages = data[column].value_counts(normalize=True, dropna=False) * 100

    return pd.DataFrame(
        {
            "categoria": counts.index.astype(str),
            "quantidade": counts.values,
            "percentual": percentages.values,
        }
    )


def plot_monthly(data: pd.DataFrame):
    if "mes" not in data.columns or data.empty:
        return None

    monthly = (
        data["mes"]
        .value_counts()
        .sort_index()
        .rename_axis("mes")
        .reset_index(name="quantidade")
    )

    fig = px.bar(
        monthly,
        x="mes",
        y="quantidade",
        text="quantidade",
        labels={"mes": "Mês", "quantidade": "Quantidade"},
        title="Quantidade de focos por mês",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), height=430)
    return fig


def plot_daynight(data: pd.DataFrame):
    table = frequency_table(data, "daynight")
    if table.empty:
        return None

    fig = px.bar(
        table,
        x="categoria",
        y="quantidade",
        text="percentual",
        labels={"categoria": "Período", "quantidade": "Quantidade"},
        title="Distribuição por dia/noite",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), height=390)
    return fig


def plot_frp_histogram(data: pd.DataFrame):
    if "frp" not in data.columns or data.empty:
        return None

    counts, edges = np.histogram(data["frp"].dropna(), bins=55)
    histogram = pd.DataFrame(
        {
            "frp": (edges[:-1] + edges[1:]) / 2,
            "frequencia": counts,
        }
    )

    fig = px.bar(
        histogram,
        x="frp",
        y="frequencia",
        labels={"frp": "FRP"},
        title="Distribuição do FRP",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), height=390)
    return fig


def plot_frp_boxplot(data: pd.DataFrame):
    if "frp" not in data.columns or data.empty:
        return None

    sample_size = min(len(data), 80_000)
    plot_data = data.sample(sample_size, random_state=42) if len(data) > sample_size else data

    fig = px.box(
        plot_data,
        x="frp",
        labels={"frp": "FRP"},
        title=f"Boxplot do FRP ({len(plot_data):,} pontos exibidos)",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), height=320)
    return fig


def plot_geo_scatter(data: pd.DataFrame):
    required = {"latitude", "longitude"}
    if not required.issubset(data.columns) or data.empty:
        return None

    sample_size = min(len(data), 120_000)
    plot_data = data.sample(sample_size, random_state=42) if len(data) > sample_size else data

    fig = px.scatter(
        plot_data,
        x="longitude",
        y="latitude",
        opacity=0.18,
        labels={"longitude": "Longitude", "latitude": "Latitude"},
        title=f"Distribuição geográfica dos focos ({len(plot_data):,} pontos exibidos)",
    )
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), height=560)
    return fig


def plot_correlation(data: pd.DataFrame):
    columns = [col for col in NUMERIC_COLUMNS if col in data.columns]
    if len(columns) < 2 or data.empty:
        return None, pd.DataFrame()

    corr = data[columns].corr().round(2)

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            text=corr.values,
            texttemplate="%{text:.2f}",
            colorbar=dict(title="Correlação"),
        )
    )
    fig.update_layout(
        title="Matriz de correlação",
        margin=dict(l=10, r=10, t=55, b=10),
        height=430,
    )
    return fig, corr


def variation_table(data: pd.DataFrame) -> pd.DataFrame:
    columns = [col for col in NUMERIC_COLUMNS if col in data.columns]
    if not columns or data.empty:
        return pd.DataFrame()

    stats = pd.DataFrame(
        {
            "media": data[columns].mean(),
            "desvio_padrao": data[columns].std(),
            "coef_variacao_%": (data[columns].std() / data[columns].mean()) * 100,
        }
    )
    return stats.sort_values("coef_variacao_%", ascending=False).round(2)


def month_table(data: pd.DataFrame) -> pd.DataFrame:
    if "mes" not in data.columns or data.empty:
        return pd.DataFrame()

    counts = data["mes"].value_counts().sort_index()
    percentages = data["mes"].value_counts(normalize=True).sort_index() * 100
    return pd.DataFrame({"quantidade": counts, "percentual": percentages}).round(2)


def frp_percentiles(data: pd.DataFrame) -> pd.Series:
    if "frp" not in data.columns or data.empty:
        return pd.Series(dtype="float64")

    return (
        data["frp"]
        .quantile([0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
        .rename(index={0.25: "q1", 0.50: "q2", 0.75: "q3", 0.90: "p90", 0.95: "p95", 0.99: "p99"})
        .round(2)
    )


def format_number(value) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:,.0f}".replace(",", ".")


def format_decimal(value, digits=2) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:.{digits}f}"


def selected_filter_values(data: pd.DataFrame) -> dict[str, list]:
    st.sidebar.header("Filtros")
    filters = {}

    for column in FILTER_COLUMNS:
        if column not in data.columns:
            continue

        values = sorted(data[column].dropna().unique().tolist())
        label = "mês" if column == "mes" else column
        selected = st.sidebar.multiselect(label, values, default=values)
        filters[column] = selected

    st.sidebar.caption("Um abraço para o Fernando Nemec! 🔥🛰️")
    return filters


def render_metrics(data: pd.DataFrame):
    total = len(data)
    frp_mean = data["frp"].mean() if "frp" in data.columns and not data.empty else np.nan
    frp_median = data["frp"].median() if "frp" in data.columns and not data.empty else np.nan
    frp_max = data["frp"].max() if "frp" in data.columns and not data.empty else np.nan

    if "mes" in data.columns and not data.empty:
        top_month = data["mes"].value_counts().idxmax()
    else:
        top_month = "-"

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Registros", format_number(total))
    col2.metric("Média FRP", format_decimal(frp_mean))
    col3.metric("Mediana FRP", format_decimal(frp_median))
    col4.metric("Mês com mais focos", top_month)
    col5.metric("Maior FRP", format_decimal(frp_max))


def render_plot(fig):
    if fig is None:
        st.info("Não foi possível gerar este gráfico com as colunas disponíveis.")
        return
    st.plotly_chart(fig, use_container_width=True)


def business_answers(data: pd.DataFrame, original: pd.DataFrame):
    stats = variation_table(data)
    months = month_table(data)
    _, corr = plot_correlation(data)
    frp_q = frp_percentiles(data)

    with st.expander("1. Quais variáveis apresentam maior variabilidade?", expanded=True):
        st.dataframe(stats, use_container_width=True)
        if not stats.empty:
            top = stats.index[0]
            coef = stats.iloc[0]["coef_variacao_%"]
            st.write(
                f"Analisando esses dados, eu percebi que `{top}` foi a variável que mais variou "
                f"proporcionalmente, com {coef:.2f}% no coeficiente de variação. Esse cálculo ajuda "
                "porque compara o desvio padrão com a média da própria coluna."
            )

    with st.expander("2. Existem padrões temporais relevantes?", expanded=True):
        st.dataframe(months, use_container_width=True)
        if not months.empty:
            top_months = months.sort_values("quantidade", ascending=False).head(2)
            listed = " e ".join(str(index) for index in top_months.index.tolist())
            st.write(
                f"Olhando melhor, os meses que mais apareceram foram {listed}. Isso mostra um padrão "
                "dentro de 2024, mas não dá para afirmar que é histórico porque a base tem só esse ano."
            )

    with st.expander("3. Há concentração geográfica de ocorrências?", expanded=True):
        st.write(
            "Pelo gráfico de latitude e longitude, dá para observar se os pontos ficam bem espalhados "
            "ou se existem áreas com maior densidade. Como a base não traz estado ou bioma diretamente, "
            "eu não vou afirmar uma região específica sem outra base de apoio."
        )

    with st.expander("4. Existem valores extremos relevantes?", expanded=True):
        st.dataframe(frp_q.to_frame("frp"), use_container_width=True)
        if not frp_q.empty:
            st.write(
                f"No `frp`, o Q3 ficou em {frp_q.get('q3'):.2f}, mas os percentis maiores sobem para "
                f"{frp_q.get('p90'):.2f} no P90, {frp_q.get('p95'):.2f} no P95 e {frp_q.get('p99'):.2f} "
                "no P99. Isso me mostra que, mesmo com a base tratada, ainda existe uma parte menor "
                "dos focos com intensidade mais alta."
            )

    with st.expander("5. Há relação entre variáveis numéricas?", expanded=True):
        st.dataframe(corr, use_container_width=True)
        if {"bright_ti4", "frp"}.issubset(corr.columns):
            value = corr.loc["bright_ti4", "frp"]
            if value > 0:
                st.write(
                    f"A correlação entre `bright_ti4` e `frp` foi positiva, com {value:.2f}. "
                    "Isso indica que focos com maior brilho ou temperatura tendem a ter maior FRP."
                )
            else:
                st.write(
                    f"A correlação entre `bright_ti4` e `frp` foi {value:.2f}, então não apareceu "
                    "uma relação positiva forte nessa base."
                )

    with st.expander("6. Quais categorias concentram maior frequência?", expanded=True):
        for column in ["daynight", "confidence", "type"]:
            table = frequency_table(data, column)
            if table.empty:
                continue
            st.caption(column)
            st.dataframe(table, use_container_width=True, hide_index=True)
            top = table.iloc[0]
            st.write(
                f"Na coluna `{column}`, a categoria {top['categoria']} foi a que mais apareceu, "
                f"com {format_number(top['quantidade'])} registros."
            )

    with st.expander("7. Existem distribuições assimétricas?", expanded=True):
        st.write(
            "O histograma e o boxplot do `frp` ajudam a ver que a maior parte dos valores fica em "
            "intensidades menores, mas ainda existe uma cauda de valores mais altos. Isso combina "
            "com a ideia de que poucos focos são bem mais intensos do que a maioria."
        )


def main():
    original_df, df, source_message = load_data()

    st.title("Global Solution Data Science — Análise de Focos de Calor no Brasil em 2024")
   

    if df.empty:
        st.error(
            "Não encontrei a base de dados. Coloque o arquivo em "
            "`data/raw/viirs-snpp_2024_Brazil.csv` ou uma base tratada em `data/processed/`."
        )
        return

    st.sidebar.caption(source_message)
    filters = selected_filter_values(df)
    filtered = apply_filters(df, filters)

    if filtered.empty:
        st.warning("Nenhum registro encontrado com os filtros selecionados.")
        return

    tab_overview, tab_time, tab_geo, tab_stats, tab_business = st.tabs(
        [
            "Visão geral",
            "Análise temporal",
            "Análise geográfica",
            "Estatística e outliers",
            "Perguntas de negócio",
        ]
    )

    with tab_overview:
        render_metrics(filtered)
        left, right = st.columns(2)
        with left:
            render_plot(plot_daynight(filtered))
        with right:
            st.subheader("Frequências por categoria")
            category = st.selectbox("Coluna", ["daynight", "confidence", "type"], key="category_overview")
            st.dataframe(frequency_table(filtered, category), use_container_width=True, hide_index=True)

    with tab_time:
        render_plot(plot_monthly(filtered))
        st.dataframe(month_table(filtered), use_container_width=True)

    with tab_geo:
        render_plot(plot_geo_scatter(filtered))

    with tab_stats:
        left, right = st.columns(2)
        with left:
            render_plot(plot_frp_histogram(filtered))
        with right:
            render_plot(plot_frp_boxplot(filtered))

        corr_fig, corr_table = plot_correlation(filtered)
        render_plot(corr_fig)

        st.subheader("Coeficiente de variação")
        st.dataframe(variation_table(filtered), use_container_width=True)

        st.subheader("Percentis do FRP")
        st.dataframe(frp_percentiles(filtered).to_frame("frp"), use_container_width=True)

    with tab_business:
        st.header("Perguntas de negócio")
        business_answers(filtered, original_df)

        st.header("Conclusão rápida")
        st.write(
            "No geral, o dashboard mostra que os focos de calor tiveram uma concentração temporal forte "
            "em alguns meses de 2024, principalmente quando olhamos o gráfico mensal. O `frp` foi uma "
            "coluna importante porque ajuda a entender a intensidade dos focos e mostrou bastante variação "
            "proporcional. Os dados de satélite também ajudam muito no monitoramento ambiental, porque "
            "permitem observar padrões no tempo, no espaço e na intensidade dos registros."
        )


if __name__ == "__main__":
    main()
