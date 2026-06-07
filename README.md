<h1 align="center">Global Solution — SPACE CONECT</h1>

<h3 align="center">
  Global Solution de Data Science desenvolvida na <strong>FIAP</strong> com dados da <strong>NASA FIRMS / VIIRS S-NPP</strong>
</h3>

<h3 align="center">
  Análise de focos de calor no Brasil em 2024 com Python, Pandas, Streamlit e Machine Learning
</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.14-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit">
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-purple?style=for-the-badge&logo=pandas">
  <img src="https://img.shields.io/badge/Scikit--learn-ML-orange?style=for-the-badge&logo=scikitlearn">
  <img src="https://img.shields.io/badge/NASA%20FIRMS-VIIRS%20S--NPP-darkblue?style=for-the-badge">
</p>

---

## Objetivo do Projeto

O **SPACE CONECT** é uma Global Solution da FIAP voltada para a análise de focos de calor no Brasil em 2024.

O projeto usa dados da **NASA FIRMS / VIIRS S-NPP** para explorar padrões temporais, geográficos e estatísticos dos registros de fogo detectados por satélite.

A análise inclui:

- Limpeza e tratamento dos dados
- Remoção de outliers
- Estatística descritiva
- Visualizações gráficas
- Perguntas de negócio
- Dashboard em Streamlit
- Modelo Random Forest como complemento extra

---

## Estrutura do Projeto

```text
GS-1-Data_Science
│
├── app.py
├── preprocessing.ipynb
├── requirements.txt
├── README.md
│
├── data
│   ├── raw
│   │   └── viirs-snpp_2024_Brazil.csv
│   │
│   └── processed
│
└── src
    └── imgs
        ├── bernardo.png
        ├── foto_marco.jpg
        └── matheus.png
```

---

## Tecnologias Utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Plotly
- Streamlit
- Scikit-learn
- NASA FIRMS / VIIRS S-NPP

---

## Análises Realizadas

### Análise temporal

Foram analisados os focos de calor por mês e por dia, buscando entender em quais períodos de 2024 os registros ficaram mais concentrados.

### Análise geográfica

Foi criada uma visualização usando latitude e longitude para observar a distribuição espacial dos focos de calor.

### Estatística descritiva

Foram analisadas medidas como média, mediana, moda, desvio padrão, coeficiente de variação, quartis e percentis.

### Outliers

Os valores extremos foram analisados principalmente na coluna `frp`, que representa a potência radiativa do fogo.

### Perguntas de negócio

O notebook e o dashboard respondem perguntas sobre variabilidade, padrões temporais, concentração geográfica, valores extremos, correlação entre variáveis e frequência de categorias.

### Modelo complementar

Também foi criado um modelo com Random Forest para classificar a intensidade dos focos de calor em baixa, média e alta.

Essa parte foi feita como complemento extra da análise.

---

## Dashboard Streamlit

O projeto possui um dashboard interativo feito com **Streamlit**.

O app permite visualizar:

- Indicadores principais
- Filtros por mês, período, confiança e tipo
- Quantidade de focos por mês
- Distribuição por dia/noite
- Histograma e boxplot do `frp`
- Distribuição geográfica dos focos
- Matriz de correlação
- Perguntas de negócio

---

## Como Rodar o Projeto

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Adicionar a base de dados

Coloque o arquivo CSV original neste caminho:

```text
data/raw/viirs-snpp_2024_Brazil.csv
```

Se existir uma base tratada, ela pode ser colocada em:

```text
data/processed/
```

### 3. Rodar o notebook

Abra o arquivo `preprocessing.ipynb` no Jupyter Notebook ou no VS Code e execute as células.

### 4. Rodar o dashboard Streamlit

```bash
streamlit run app.py
```

Depois, acesse o endereço mostrado no terminal, normalmente:

```text
http://localhost:8501
```

---

## Observação Sobre os Dados

O arquivo CSV da NASA não deve ser enviado para o GitHub.

O `.gitignore` já ignora arquivos em:

```text
data/raw/
data/processed/
```

Assim, a base fica apenas localmente na máquina de quem está executando o projeto.

---

## Integrantes do Grupo

| [<img loading="lazy" src="./src/imgs/foto_marco.jpg" width=115><br><sub>Marco Aurélio</sub><br><sub>RM 563827</sub>](https://github.com/Arriatea) | [<img loading="lazy" src="./src/imgs/matheus.png" width=115><br><sub>Matheus Vasques</sub><br><sub>RM 563309</sub>](https://github.com/maatvasques) | [<img loading="lazy" src="./src/imgs/bernardo.png" width=115><br><sub>Bernardo Hanashiro</sub><br><sub>RM 565266</sub>](https://github.com/BernardoYuji) | [<sub>Mariana Egito</sub><br><sub>RM 562544</sub>](https://github.com/marianaegito) |
| :---: | :---: | :---: | :---: |

---
