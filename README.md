# Classificador de Criticidade de Bugs — Sistema Híbrido por Stacking

**Disciplina:** Reconhecimento de Padrões — UPE, 1º Período  
**Referência central:** Lamkanfi et al. (2010) — *Predicting the Severity of a Reported Bug*

---

## Índice

1. [Objetivo](#objetivo)
2. [Motivação e Revisão da Literatura](#motivação-e-revisão-da-literatura)
3. [Nossa Proposta](#nossa-proposta)
4. [Datasets](#datasets)
5. [Arquitetura do Sistema Híbrido](#arquitetura-do-sistema-híbrido)
6. [Baselines Comparados](#baselines-comparados)
7. [Processo de Seleção de Modelos](#processo-de-seleção-de-modelos)
8. [Métricas Avaliadas](#métricas-avaliadas)
9. [Estrutura do Projeto](#estrutura-do-projeto)
10. [Seções do Notebook](#seções-do-notebook)
11. [Setup e Execução](#setup-e-execução)
12. [Requisitos do Trabalho — Checklist](#requisitos-do-trabalho--checklist)
13. [O que ainda falta implementar](#o-que-ainda-falta-implementar)
14. [Tempos Estimados](#tempos-estimados)
15. [Referências](#referências)

---

## Objetivo

Dado o **texto** de um relatório de bug de software ou de uma vulnerabilidade CVE,
classificar automaticamente sua **criticidade** em três níveis: **Alta**, **Média** ou **Baixa**.

A triagem manual de bug trackers como o Jira do Apache Spark é cara e sujeita a
inconsistências — ferramentas com dezenas de milhares de itens acumulados dependem
de triagem prioritária para que falhas críticas não fiquem enterradas na fila.
A classificação automática de criticidade endereça diretamente esse gargalo.

**Contribuição técnica deste trabalho:** extensão do pipeline de Lamkanfi et al. (2010)
com representação híbrida (TF-IDF + embeddings de LLM) e meta-aprendizado por Stacking,
avaliada em dois domínios distintos: bugs de software e vulnerabilidades de segurança.

---

## Motivação e Revisão da Literatura

### O problema de desbalanceamento

Em datasets reais de bug trackers, bugs críticos são raros:
- Apache Spark: ~2–5% dos tickets são prioridade Blocker/Critical
- CVSS: vulnerabilidades de pontuação Alta (≥ 7.0) são minoria frente às Médias

Isso torna acurácia uma métrica enganosa — um modelo que sempre prevê "Média" acerta
~70–80% das vezes mas é inútil. Por isso usamos **F1-macro** como métrica principal e
aplicamos **SMOTE** dentro de cada fold de treino para mitigar o desbalanceamento.

### O que Lamkanfi et al. (2010) fizeram e suas limitações

Lamkanfi et al. treinaram NB, SVM, Decision Tree e k-NN com representação
**bag-of-words** em reports dos projetos Eclipse, Mozilla e GNOME, reportando
precisão/recall de 0.65–0.75 quando o treino tem ≥ 500 reports por classe.

**Limitações identificadas:**
- Bag-of-words não captura semântica (sinônimos, paráfrases, contexto)
- Classifica com modelos independentes — não explora complementaridade entre eles
- Não aborda o desbalanceamento de classes
- Avalia em único domínio (bugs de software, não vulnerabilidades de segurança)

### O que sabemos da literatura mais recente

| Evolução | Referência | Melhoria |
|---|---|---|
| Prioridade multi-fator (texto + metadados) | Tian et al. (DRONE, 2013) | Estende severidade para prioridade |
| CNN + RF com boosting | **Umer et al. (2019, Sensors)** | Híbrido neural/clássico supera modelos isolados |
| Fine-tuning BERT para bugs | Ali et al. (2024, JSS) | Estado-da-arte atual com transformers |
| NLP para CVSS em CVEs | **Spanos & Angelis (2018, JSS)** | Mesmo problema, domínio de segurança |

### Como Umer et al. (2019) e Spanos & Angelis (2018) agregam ao trabalho

#### Umer et al. (2019) — validação da abordagem híbrida

Umer et al. propõem uma arquitetura híbrida CNN + Random Forest + boosting para
classificação de severidade de bugs no dataset Eclipse. Os principais achados:

- Um classificador isolado (CNN sozinho, RF sozinho) é consistentemente superado pela
  combinação de ambos
- A representação neural (CNN extrai features do texto) combinada com o ensemble clássico
  (RF + boosting) captura padrões complementares que nem um nem outro capturam sozinhos
- A classe Alta (crítico) é a mais difícil — F1 individual inferior, mas o ensemble
  recupera boa parte do desempenho

**Como se relaciona com nosso trabalho:** nossa arquitetura é conceitualmente análoga.
Substituímos o CNN por embeddings congelados do Qwen3-0.6B (representação neural sem
fine-tuning) e em vez de combinar por boosting usamos **Stacking com meta-classificador**,
o que dá ao sistema flexibilidade para aprender pesos diferentes por classe.
Umer et al. nos fornece a justificativa empírica de que híbridos neural+clássico
**funcionam para esse problema** — reforçando a escolha do nosso design.

#### Spanos & Angelis (2018) — justificativa para o segundo dataset

Spanos & Angelis aplicam ML diretamente em descrições textuais de CVEs da NVD para
prever o score de severidade CVSS, e confirmam que o texto da descrição CVE carrega
informação preditiva suficiente para classificação automática sem features adicionais.

**Como se relaciona com nosso trabalho:** nós aplicamos a **mesma estratégia híbrida**
(TF-IDF + Embeddings + Stacking) em dois datasets — Apache Spark e CIRCL CVE.
Spanos & Angelis não validam nossa estratégia, mas validam a **escolha do domínio**:
mostram que classificar CVE por texto é uma tarefa solucionável antes de nós
tentarmos nossa abordagem nele. Em outras palavras, eles justificam por que o CIRCL CVE
é um segundo dataset legítimo para testar generalização do sistema.

---

Nosso trabalho posiciona-se entre os modelos clássicos (Lamkanfi) e o fine-tuning de
transformers (Ali et al.): usamos **embeddings congelados de LLM** (Qwen3-0.6B) sem
fine-tuning, combinados com TF-IDF via Stacking, explorando o melhor de dois mundos
com custo computacional viável para um projeto acadêmico.

---

## Nossa Proposta

| Dimensão | Lamkanfi (2010) | Este trabalho |
|---|---|---|
| Representação | Bag-of-words | TF-IDF sublinear **+** embeddings Qwen3-0.6B (1024 dim) |
| Estratégia | Modelos independentes | **Stacking** (meta-learning com OOF aninhado) |
| Desbalanceamento | Não tratado | SMOTE dentro de cada fold de treino |
| Datasets | Eclipse, Mozilla, GNOME | Apache Spark (Jira) + CIRCL CVE |
| Domínios | Bugs de software | Bugs de software **+** vulnerabilidades de segurança |
| Aceleração | CPU | GPU automática (CUDA) com fallback para CPU |
| Testes estatísticos | Nenhum | Friedman + Nemenyi + Wilcoxon |

---

## Datasets

### Dataset 1 — Apache Spark (Jira)

| Atributo | Detalhe |
|---|---|
| **Fonte** | `data/raw/spark_bugs.csv` (incluído no repositório) |
| **Origem** | Bug tracker Jira do projeto Apache Spark (open-source) |
| **Campo de texto** | `summary` + `description` concatenados (truncados a 1.000 chars) |
| **Tamanho original** | ~9.000 bugs |
| **Após undersampling** | ~4.000 bugs (1.333 por classe) |
| **Distribuição original** | Alta ~2–5% · Média ~70% · Baixa ~25% |

**Mapeamento de rótulos (prioridade Jira → criticidade):**

| Prioridade Jira | Criticidade |
|---|---|
| Blocker, Critical | Alta |
| Major | Média |
| Minor, Trivial | Baixa |

**Contexto:** prioridade Jira é atribuída pelo reportador e pode ser subjetiva.
Estudos anteriores (Lamkanfi 2010, Tian 2013) usam essa heurística como padrão-ouro
por ser a única anotação disponível em escala industrial.

---

### Dataset 2 — CIRCL CVE (HuggingFace)

| Atributo | Detalhe |
|---|---|
| **Fonte** | HuggingFace: `CIRCL/vulnerability-CVE` (download automático na 1ª execução) |
| **Origem** | Base de dados CVE mantida pela CIRCL (Computer Incident Response Center Luxembourg) |
| **Campo de texto** | Campo `description` da entrada CVE |
| **Tamanho original** | ~200.000 CVEs |
| **Após undersampling** | ~4.000 CVEs (1.333 por classe) |
| **Distribuição original** | Alta ~40–45% · Média ~35–40% · Baixa ~15–20% |

**Mapeamento de rótulos (CVSS Base Score → criticidade, padrão NIST/NVD):**

| CVSS Base Score | Criticidade |
|---|---|
| ≥ 7.0 | Alta |
| 4.0 – 6.9 | Média |
| < 4.0 | Baixa |

**Contexto:** o CVSS (Common Vulnerability Scoring System) é o padrão internacional
para quantificar a severidade de vulnerabilidades. Escores são calculados a partir
de métricas técnicas (vetor de ataque, complexidade, impacto de confidencialidade,
integridade e disponibilidade), tornando o rótulo mais objetivo do que prioridade Jira.

**Por que dois datasets?**  
Avaliação em dois domínios distintos (bugs de software × vulnerabilidades de segurança)
testa a **generalização** da abordagem além do dataset original de Lamkanfi, cumprindo
o requisito de utilização em mais de uma base de dados.

---

## Arquitetura do Sistema Híbrido

O sistema implementa **Stacking** (meta-aprendizado), uma das três estratégias de
ensemble definidas em Lorenzato (RecPad Aula 09) e formalizada por Wolpert (1992).

```
════════════════════════ NÍVEL 0 — Classificadores Base ════════════════════════

  Entrada: texto bruto do bug/CVE
      │
      ├─► [TF-IDF sublinear, 20k features]─► LogReg (C=busca)  ─► P(Alta) P(Méd) P(Bx) ──┐
      │                                                                 cols 0–2             │
      │                                                                                      │
      └─► [Qwen3-0.6B Embedding, 1024 dim] ─► RF (300 árvores)  ─► P(Alta) P(Méd) P(Bx) ──┤
                                           ├─► XGBoost (busca)  ─► P(Alta) P(Méd) P(Bx) ──┤
                                           └─► LightGBM (busca) ─► P(Alta) P(Méd) P(Bx) ──┤
                                                                       cols 3–11             │
                                                                                             │
════════════════════════ NÍVEL 1 — Meta-Classificador ═══════════════════════════           │
                                                                                             │
  Meta-features: 12 colunas ◄──────────────────────────────────────────────────────────────┘
     └─ TF-IDF (cols 0–2) + Embeddings (cols 3–11)
          │
          ▼
     LogReg meta (C=0.1)  ──►  classe final: Alta / Média / Baixa
```

**Onde acontece a hibridização:**  
As 12 meta-features combinam saídas de **duas representações fundamentalmente distintas**:
- Colunas 0–2: frequência de termos (esparso, ~20k dimensões, linear)
- Colunas 3–11: semântica densa (1024 dimensões, não-linear)

O meta-classificador aprende automaticamente o peso ideal de cada fonte por classe,
equivalente ao "Módulo de Combinação" no diagrama de Sistemas Inteligentes Híbridos
de Lorenzato (2024, slide 29).

**Anti-leakage — Out-of-Fold (OOF):**  
As meta-features de treino são geradas com inner CV (5 folds): cada amostra é predita
por um modelo que nunca a viu no treino, evitando que o meta-classificador memorize
os dados de base.

---

## Baselines Comparados

| # | Modelo | Representação | Biblioteca | Referência |
|---|---|---|---|---|
| 1 | Complement Naive Bayes | TF-IDF | `sklearn` | Lamkanfi (2010) |
| 2 | Regressão Logística | TF-IDF | `sklearn` | Lamkanfi (2010) |
| 3 | SVM Linear (calibrado) | TF-IDF | `sklearn` | Lamkanfi (2010) |
| 4 | Random Forest | Embeddings | `sklearn` | Breiman (2001) |
| 5 | **XGBoost** | Embeddings | **`xgboost`** | Chen & Guestrin (2016) |
| 6 | **LightGBM** | Embeddings | **`lightgbm`** | Ke et al. (2017) |

> Cumpre o requisito: **não é permitido usar apenas sklearn** — XGBoost e LightGBM
> são bibliotecas externas independentes; SentenceTransformer (embeddings) também.

---

## Processo de Seleção de Modelos

A especificação do processo de seleção segue três etapas encadeadas:

### Etapa 1 — Justificativa a priori (antes do experimento)
Os 6 baselines foram escolhidos para cobrir o espaço de modelos relevantes na literatura:
- NB, LogReg, SVM: replicam os baselines de Lamkanfi (2010)
- RF, XGBoost, LightGBM: extensão moderna sobre representação semântica

### Etapa 2 — Ajuste de hiperparâmetros (Seção 6.5 do notebook)
RandomizedSearchCV (20 iterações × 5 folds internos) busca os melhores hiperparâmetros
de LogReg, XGBoost e LightGBM antes da avaliação final. NB, SVM e RF são robustos com
defaults e não entram na busca.

### Etapa 3 — Comparação estatística pós-experimento (Seção 8 do notebook)
**Teste de Friedman** (não-paramétrico, multi-modelo, seguindo Demšar 2006):
- H₀: todos os modelos têm desempenho equivalente
- Se p < 0.05: **Nemenyi pós-hoc** compara todos os pares

Os resultados do Nemenyi informam:
- Quais modelos são estatisticamente equivalentes entre si
- Se modelos redundantes existem (baixa diversidade → candidatos a exclusão do stacking)
- Justificativa documentada para a composição final do stacking

> **Nota de pendência:** a célula de documentação da decisão (qual modelo foi incluído
> no stacking com base nos resultados Nemenyi, e por quê) ainda está pendente.
> Ver seção [O que ainda falta implementar](#o-que-ainda-falta-implementar).

---

## Métricas Avaliadas

Todas as métricas são reportadas como **média ± desvio padrão** sobre os 10 folds externos.

| Métrica | Fórmula resumida | Por que usar |
|---|---|---|
| **F1-macro** | Média do F1 por classe | **Métrica principal** — peso igual entre classes |
| Acurácia | Acertos / Total | Intuicível, mas enganosa com desbalanceamento |
| Precisão-macro | Média da precisão por classe | Custo de falsos positivos |
| Recall-macro | Média do recall por classe | Custo de falsos negativos (crítico para Alta) |
| MCC | Correlação de Matthews | Robusto ao desbalanceamento; intervalo [−1, 1] |
| Kappa de Cohen | Acordo além do acaso | 0 = aleatório · 1 = perfeito |
| F1-Alta | F1 da classe Alta | O modelo aprende a classe rara? |
| F1-Média | F1 da classe Média | — |
| F1-Baixa | F1 da classe Baixa | — |

---

## Estrutura do Projeto

```
Projeto/
├── data/
│   ├── raw/
│   │   └── spark_bugs.csv              # Dataset Spark (incluído)
│   ├── emb_spark_balanced.npy          # Cache de embeddings Spark (gerado na 1ª execução)
│   └── emb_circl_balanced.npy          # Cache de embeddings CIRCL (gerado na 1ª execução)
├── notebooks/
│   └── projeto_hibrido.ipynb           # Notebook principal (12 seções)
├── resultados/                          # Gerado automaticamente ao executar
│   ├── eda_overview.png
│   ├── preproc_truncamento.png
│   ├── preproc_tokens_*.png
│   ├── preproc_bigrams_*.png
│   ├── preproc_tfidf_*.png
│   ├── balanceamento_antes_depois.png
│   ├── embeddings_pca_*.png
│   ├── nemenyi_*.png
│   ├── confusion_*.png
│   ├── wilcoxon_*.png
│   ├── f1_comparacao_geral.png
│   ├── metricas_apache_spark.csv
│   └── metricas_circl_cve.csv
├── requirements.txt
└── README.md
```

---

## Seções do Notebook

| Seção | Conteúdo |
|---|---|
| 0. Título | Contexto, tabela comparativa vs Lamkanfi, estrutura |
| 1. Dados | Carregamento Spark e CIRCL, mapeamento de rótulos, estatísticas |
| 2. EDA | Distribuição de classes, comprimento dos textos por classe |
| 3. Pré-processamento | Truncamento, tokens por classe, bigramas, top TF-IDF features |
| 4. Balanceamento | Undersampling real-first + visualização antes/depois |
| 5. Embeddings | Qwen3-0.6B com cache em disco + PCA 2D para inspeção |
| 6. Framework | Validação cruzada 10-fold, função `compute_metrics`, todas as métricas |
| 6.5. Hiperparâmetros | RandomizedSearchCV para LogReg, XGBoost e LightGBM |
| 7. Baselines | 6 modelos base com SMOTE dentro do fold |
| 8. Seleção | Friedman + Nemenyi pós-hoc + tabela de ranks |
| 9. Stacking | Sistema híbrido com OOF aninhado (nested CV 10×5) |
| 10. Resultados | Tabela consolidada + matrizes de confusão + gráfico de barras |
| 11. Wilcoxon | Comparação final estacking vs melhor baseline, fold-a-fold |
| 12. Conclusão | Decisões justificadas, limitações, próximos passos |

---

## Dependências Principais

| Pacote | Versão mínima | Papel no projeto |
|---|---|---|
| `pandas` / `numpy` / `scipy` | 2.0 / 1.24 / 1.11 | Manipulação de dados, estatísticas (Friedman, Wilcoxon) |
| `scikit-learn` | 1.3 | LogReg, SVM, NB, RF, TF-IDF, métricas, validação cruzada |
| `imbalanced-learn` | 0.12 | SMOTE e `ImbPipeline` (SMOTE dentro do fold) |
| `xgboost` | 2.0 | Baseline XGBoost — cumpre requisito não-sklearn |
| `lightgbm` | 4.0 | Baseline LightGBM — cumpre requisito não-sklearn |
| `scikit-posthocs` | 0.9 | Teste de Nemenyi pós-hoc (Friedman) |
| `statsmodels` | 0.14 | Dependência interna do scikit-posthocs |
| `torch` | 2.0 | Detecção de GPU (CUDA) e backend dos embeddings |
| `sentence-transformers` | 3.0 | Geração de embeddings Qwen3-0.6B (1024 dim) |
| `datasets` | 2.16 | Download automático do CIRCL CVE do HuggingFace |
| `matplotlib` / `seaborn` | 3.7 / 0.12 | Todos os gráficos (EDA, PCA, Nemenyi, confusão, Wilcoxon) |
| `tqdm` | 4.65 | Barra de progresso na geração de embeddings |
| `truststore` | 0.9 | Fix de SSL em redes corporativas com CA local |
| `joblib` | 1.3 | Paralelismo interno do scikit-learn (`n_jobs=-1`) |
| `ipykernel` / `ipywidgets` / `jupyterlab` | — | Ambiente Jupyter para execução do notebook |

> `torch` é usado apenas para detecção de GPU (`torch.cuda.is_available()`). Se preferir
> rodar só em CPU sem instalar PyTorch completo, substitua a detecção por `DEVICE = 'cpu'`
> na primeira célula — os embeddings serão gerados via CPU pelo `sentence-transformers`.

---

## Setup e Execução

### Pré-requisitos

- Python 3.10+
- (Opcional) GPU NVIDIA com driver CUDA 11.8+

### 1. Criar e ativar o ambiente virtual

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Instalar dependências

```powershell
pip install -r requirements.txt
```

> **GPU (opcional):** instale o PyTorch com suporte CUDA antes das demais dependências:
> ```powershell
> pip install torch --index-url https://download.pytorch.org/whl/cu118
> pip install -r requirements.txt
> ```
> O notebook detecta a GPU automaticamente. XGBoost, LightGBM e os embeddings Qwen3
> usam CUDA se disponível; caso contrário, tudo roda em CPU sem nenhuma alteração de código.

### 3. Abrir o Jupyter

```powershell
jupyter lab
```

Abra `notebooks/projeto_hibrido.ipynb` e execute em ordem (**Kernel → Restart & Run All**).

> **Primeira execução:** o dataset CIRCL é baixado automaticamente do HuggingFace
> (~2 GB) e os embeddings são gerados (~8–17 min em CPU, ~1–2 min em GPU).
> Execuções seguintes carregam tudo do cache e são muito mais rápidas.

---

## Requisitos do Trabalho — Checklist

| Requisito | Status | Onde no projeto |
|---|---|---|
| Desenvolver um sistema híbrido | ✅ | Stacking TF-IDF + Embeddings (Seção 9) |
| Verificar literatura e propor alteração | ⚠️ Parcial | Tabela no título; falta seção dedicada no notebook |
| Experimentos comparativos (proposta vs baselines) | ✅ | 6 baselines + stacking, Seções 7–10 |
| Não usar apenas métodos sklearn | ✅ | XGBoost, LightGBM, SentenceTransformer |
| Testes de hipótese obrigatórios | ✅ | Friedman + Nemenyi (Seção 8) + Wilcoxon (Seção 11) |
| Mais de uma base de dados | ✅ | Apache Spark + CIRCL CVE |
| Tabela de métricas | ✅ | Seção 10: 9 métricas, salvo em CSV |
| Especificação do processo de seleção | ⚠️ Parcial | Nemenyi existe; falta célula de decisão documentada |

---

## O que ainda falta implementar

### Prioridade Alta (necessário para requisitos do trabalho)

#### 1. Célula de decisão pós-Nemenyi (Seção 8)

**O que falta:** após o heatmap do Nemenyi, adicionar uma célula markdown que
documente explicitamente a decisão de composição do stacking com base nos resultados:

> *"Os testes revelaram que X e Y são estatisticamente equivalentes (p > 0.05) enquanto
> Z é significativamente distinto (p < 0.05). Para maximizar diversidade no ensemble,
> selecionamos A, B, C e D como classificadores base do stacking, pois cobrem os dois
> grupos de feature representation (TF-IDF e Embeddings) e apresentam complementaridade
> estatística confirmada pelo Nemenyi."*

Sem essa célula, o requisito **"especificação do processo de seleção de modelos"**
fica implícito — o avaliador pode não perceber a conexão entre o Nemenyi e o stacking.

#### 2. Seção de revisão de literatura no notebook (antes da Seção 1)

**O que falta:** uma célula markdown entre o título e o carregamento de dados que
explicite:
- O que foi encontrado na literatura (Lamkanfi, Menzies, Tian, Umer, Spanos)
- Qual limitação específica cada referência possui
- Qual alteração este trabalho propõe em resposta

Isso torna auditável o cumprimento do requisito **"verificar trabalhos da literatura
e propor uma alteração"**.

---

### Prioridade Média (melhoria de alinhamento com os slides do curso)

#### 3. Baseline de Fusão (Voto Suave Ponderado)

**Contexto:** a Aula 09n (Lorenzato) define três estratégias de ensemble:
Métodos de Fusão · Métodos de Seleção · Meta-learning. O notebook implementa apenas
Meta-learning (Stacking). Adicionar um **voto suave ponderado** como baseline de Fusão
demonstra conhecimento das três estratégias e gera uma comparação direta
Meta-learning vs Fusão — um argumento mais forte para a adoção do Stacking.

**O que adicionar:** uma função `run_weighted_voting(df, emb, dataset_name)` que:
- Treina os 4 modelos base (LogReg+TF-IDF, RF+Emb, XGBoost+Emb, LightGBM+Emb)
- Usa o F1-macro de validação interna como peso: $w_i = \alpha_i / \sum \alpha_j$
- Combina as probabilidades ponderadas para a decisão final
- Avalia com o mesmo framework de 10 folds externos

**Resultado esperado:** Stacking deve superar Fusão, justificando a escolha.

---

### Prioridade Baixa (qualidade opcional)

#### 4. Informar o grupo na planilha

Tarefa administrativa — registrar o tema do projeto na planilha compartilhada da turma.

---

## Tempos Estimados

| Etapa | CPU (i7) | GPU (RTX 3050) |
|---|---|---|
| Embeddings Spark (~4k textos) | ~8 min | ~1 min |
| Embeddings CIRCL (~4k textos) | ~8 min | ~1 min |
| Busca de hiperparâmetros (2 datasets) | ~20 min | ~12 min |
| Baselines (2 datasets, 10 folds) | ~15 min | ~10 min |
| Stacking (2 datasets, nested 10×5 CV) | ~60 min | ~30 min |
| **Total (primeira execução)** | **~110 min** | **~55 min** |

> Embeddings são cacheados em `data/*.npy` — nas execuções seguintes, apenas os modelos
> são re-treinados.

---

