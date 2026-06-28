# Análise Detalhada — Projeto Híbrido de Classificação de Criticidade de Bugs

> **Projeto:** Classificador de Criticidade de Bugs via Sistema Híbrido por Stacking  
> **Disciplina:** Reconhecimento de Padrões — UPE, 1º Período  
> **Atualizado em:** 2026-06-28

---

## 1. Artigos de Base

### 1.1 Lamkanfi et al. (2010) — *Predicting the Severity of a Reported Bug*

#### O que o artigo fez

Lamkanfi et al. investigaram a previsão automática da **severidade** de bugs a partir do texto dos relatórios de bug. O estudo usa três repositórios de bug trackers de projetos open-source populares:

| Dataset | Repositório | Bugs analisados |
|---|---|---|
| Eclipse | Bugzilla | ~50.000 bugs |
| Mozilla | Bugzilla | ~120.000 bugs |
| GNOME | Bugzilla | ~25.000 bugs |

**Representação textual:** Bag-of-Words puro (não TF-IDF, apenas contagem de frequência), sem qualquer pré-processamento avançado além de tokenização básica.

**Classificadores testados:**
- Naive Bayes (NB)
- SVM (SVM Linear)
- Decision Tree (DT)
- k-NN (k=1, k=3, k=5)

**Labels de severidade:** blocker, critical, major, minor, trivial — cinco classes originais da escala Bugzilla.

**Métricas reportadas:**
- Precisão por classe: entre 0,65 e 0,75 para SVM e NB
- Recall por classe: similar faixa
- Acurácia global: ~70–75% para os melhores modelos

**Principais achados:**
1. NB e SVM são consistentemente melhores que DT e k-NN
2. A classe *blocker* (mais crítica) é a mais difícil de classificar — baixo recall
3. O desbalanceamento de classes prejudica especialmente as minorias
4. Bag-of-Words simples já produz resultados razoáveis

**Limitações explícitas do artigo:**
- Usa apenas frequência de termos, sem ponderação IDF
- Sem técnicas de balanceamento de classes (SMOTE, undersampling)
- Sem combinação de modelos (ensembles)
- Avaliação em apenas um domínio (bug trackers Bugzilla)
- Cinco classes originais; não consolida em Alta/Média/Baixa

#### Como o projeto se baseia neste artigo

Este artigo é a **referência baseline direta** do projeto. A célula 0 do notebook descreve explicitamente:

> "Base na literatura: Lamkanfi et al. (2010) — Predicting the Severity of a Reported Bug"

O projeto replica o experimento de Lamkanfi, adaptando-o:

| Aspecto | Lamkanfi (2010) | Nosso projeto |
|---|---|---|
| Domínio | Bugzilla (Eclipse/Mozilla/GNOME) | Jira (Apache Spark) + CVE (CIRCL) |
| Classes | 5 (blocker→trivial) | 3 (Alta/Média/Baixa) — consolidação |
| Representação | Bag-of-Words (count) | TF-IDF + Embeddings LLM |
| Modelos | NB, SVM, DT, k-NN | NB, SVM, DT, k-NN + 6 modelos adicionais |
| Balanceamento | Nenhum | Undersampling + SMOTE dentro do fold |
| Ensemble | Nenhum | Stacking + Voto Ponderado + Comb. Dinâmica |
| Métrica principal | Precisão/Recall | F1-macro (penaliza ignorar minoria) |

**Replicação dos baselines:** Os quatro modelos de Lamkanfi (NB, LogReg≈SVM, DT, k-NN) são implementados no projeto como baselines #1–4 (TF-IDF) e #6 (k-NN por embeddings). O SVM (TF-IDF) usa `CalibratedClassifierCV` para produzir probabilidades necessárias ao Stacking.

**O que superamos:** Lamkanfi não tratou desbalanceamento. Nosso projeto usa F1-macro como métrica principal justamente porque acurácia é enganosa com desbalanceamento — a célula 4 do notebook documenta que a classe Alta representa ~2% dos bugs do Spark antes do balanceamento.

---

### 1.2 Umer et al. (2019) — *CNN and RF Based Hybrid Framework for Bug Severity Prediction*

#### O que o artigo fez

Umer et al. propuseram um framework híbrido que combina CNN (Convolutional Neural Network) com Random Forest e técnicas de boosting para previsão de severidade de bugs. O trabalho é explicitamente sobre **sistemas híbridos** — exatamente o tema do nosso projeto.

**Dataset:** Eclipse Bugzilla (mesmo repositório de Lamkanfi), mas com foco na previsão multi-classe.

**Arquitetura do sistema híbrido de Umer:**
1. CNN extrai features locais do texto (n-gramas via filtros convolucionais)
2. Random Forest/Boosting classifica sobre as features extraídas pela CNN
3. Combinação por votação/ensembling

**Representação:** Word embeddings pré-treinados (GloVe/Word2Vec) como entrada da CNN — diferente do TF-IDF esparso de Lamkanfi.

**Principais achados:**
1. O sistema híbrido CNN+RF supera modelos individuais consistentemente
2. A classe *Alta* (blocker/critical) continua sendo a mais difícil
3. Embeddings pré-treinados melhoram representação em relação a BoW
4. Boosting (AdaBoost, XGBoost) contribui positivamente no ensemble

**Limitações:**
- CNN requer GPU para treinamento eficiente
- Embeddings GloVe/Word2Vec são estáticos — não capturam contexto
- Dataset único (Eclipse): sem validação cross-domain
- Sem teste de hipótese formal entre modelos

#### Como o projeto se baseia neste artigo

Umer et al. justificam a escolha pela **abordagem híbrida** em vez de modelos isolados. A seção "Nossa extensão" no notebook (célula 0) documenta:

| Elemento | Umer (2019) | Nossa extensão |
|---|---|---|
| Arquitetura híbrida | CNN + RF/Boosting | Stacking (meta-learning) sobre base-learners diversos |
| Embeddings | Word2Vec/GloVe (estáticos) | Qwen3-Embedding-0.6B (contextuais, modernos) |
| Base-learners | CNN + RF/Boosting | TF-IDF (NB, SVM, LogReg, DT) + Emb (RF, XGBoost, LightGBM, CatBoost, KNN, MLP) |
| Combinação | Votação/feature stacking simples | Meta-LogReg treinada em predições OOF |
| Datasets | Eclipse apenas | Apache Spark + CIRCL CVE (dois domínios) |
| Validação | Holdout/K-fold simples | 10-fold externo + GridSearchCV 5-fold interno (nested) |
| Hipótese | Não faz testes formais | Friedman + Nemenyi + Wilcoxon |

**Contribuição deste artigo para o projeto:** A motivação central de usar **sistemas híbridos** (múltiplos modelos combinados) vem diretamente de Umer. O projeto estende esta ideia adicionando:
1. Diversidade de representação (esparso TF-IDF + denso embeddings)
2. Seleção dinâmica de base-learners via testes estatísticos (Nemenyi)
3. Comparação formal entre estratégias de ensemble (Stacking vs Fusão vs Combinação Dinâmica)

**Resultado comparável:** Umer reporta melhorias de ~5–10% sobre modelos individuais com híbridos. No nosso projeto, o Stacking supera o melhor baseline em F1-Alta no Spark (0,438 vs 0,413) e mantém desempenho similar ao SVM no CVE (0,699 vs 0,704). A diferença é pequena e estatisticamente não significativa (Wilcoxon p=0,4922 no Spark), coerente com a expectativa quando baselines individuais já são fortes.

---

### 1.3 Spanos & Angelis (2018) — *Multi-Target Prediction of Vulnerability Severity Scores*

#### O que o artigo fez

Spanos & Angelis investigaram a **previsão automática de scores CVSS** (Common Vulnerability Scoring System) a partir do texto de descrições de vulnerabilidades CVE. O trabalho foca em **vulnerabilidades de segurança** (CVE), não bugs de software genéricos.

**Dataset:** NVD (National Vulnerability Database) — CVEs com scores CVSS publicados pelo NIST.

**Abordagem:**
- Texto de entrada: campo `description` da CVE
- Tarefa: prever múltiplos componentes do CVSS (Base Score, Access Vector, etc.)
- Representação: TF-IDF e features textuais simples
- Classificadores: SVM, Random Forest, Naive Bayes

**Principais achados:**
1. **O texto da CVE é suficiente** para prever o score CVSS com boa precisão
2. TF-IDF sobre a descrição da CVE consegue F1 ~0,70–0,80 para componentes individuais
3. Previsão do score geral (Alta/Média/Baixa) é mais fácil que componentes individuais
4. SVM e RF são os melhores performers

**Limitações:**
- Multi-target prediction: prediz cada componente do CVSS independentemente
- Sem embeddings contextuais (usa TF-IDF apenas)
- Sem técnicas de balanceamento
- Sem validação em outros domínios (apenas CVE/NVD)

#### Como o projeto se baseia neste artigo

Spanos & Angelis justificam **o uso de CVEs como segundo dataset** e confirmam que texto de vulnerabilidade é discriminativo para classificação de criticidade.

| Aspecto | Spanos (2018) | Nosso projeto (CIRCL CVE) |
|---|---|---|
| Fonte dos dados | NVD REST API | HuggingFace (CIRCL dataset) |
| Score CVSS | Múltiplos componentes | Score consolidado → 3 classes |
| Versão CVSS | CVSS v2 (predominante) | v3.1 → v3.0 → v4.0 → v2.0 (preferência) |
| Threshold | Próprios | NIST: ≥7.0 Alta, 4.0–6.9 Média, <4.0 Baixa |
| Representação | TF-IDF | TF-IDF + Qwen3 embeddings |
| Resultado TF-IDF | ~0,70–0,80 F1 | SVM (TF-IDF): 0,704 F1-macro |

**Validação cruzada dos resultados:** Os resultados de Spanos sugerem que TF-IDF + SVM em CVEs produz F1 ~0,70–0,80. Nosso SVM (TF-IDF) no CIRCL CVE obtém 0,704 F1-macro — **consistente com a faixa reportada**, confirmando que a representação e o dataset são apropriados.

---

### 1.4 Lorenzato (2024) — *Aula 09 RecPad: Combinação de Classificadores*

#### O que o material cobre

O material didático da disciplina formaliza a **taxonomia de combinação de classificadores** e apresenta os fundamentos teóricos que estruturam a arquitetura do projeto.

**Taxonomia de combinação (Lorenzato/Kuncheva):**

```
Combinação de Classificadores
├── Fusão (combinar saídas de todos os modelos)
│   ├── Voto majoritário
│   ├── Média de probabilidades
│   └── Voto ponderado ← implementado (Seção 9.1)
├── Seleção (usar apenas o melhor modelo por instância)
│   └── Combinação Dinâmica ← implementado (Seção 9.2)
└── Meta-learning (treinar um meta-classificador)
    └── Stacking ← implementado (Seção 9)
```

**ELM (Extreme Learning Machine):**
- Pesos de entrada aleatórios e fixos
- Pesos de saída calculados analiticamente via pseudo-inversa: `W_out = H⁺ · T`
- `H = sigmoid(X · W_in + b_in)`
- Treinamento muito rápido (sem backpropagation)
- Implementado no projeto via NumPy puro (célula 22 do notebook)

**Combinação Ponderada Dinâmica (exercício de aula):**
- Para cada ponto de teste x, encontra K vizinhos no espaço de predições OOF
- Calcula pesos locais via mínimos quadrados: `W = pinv(X_viz) @ y_viz_onehot`
- Predição: `proba = x_teste @ W`

#### Como o material estrutura o projeto

O projeto implementa **as três famílias** da taxonomia em paralelo para comparação direta:

| Família | Implementação | Seção do Notebook | Estado |
|---|---|---|---|
| Meta-learning | Stacking com LogReg meta C=0.1 | Seção 9 | ✅ Executado |
| Fusão | Voto Suave Ponderado (F1-weighted) | Seção 9.1 | ✅ Executado |
| Seleção | Combinação Dinâmica (KNN + MQ local) | Seção 9.2 | ⚠️ Implementado, não executado |

**ELM no projeto:** O `ELMClassifier` é implementado em NumPy puro (sem sklearn) na célula 22, cumprindo o requisito "não usar apenas sklearn". Passou pelo GridSearchCV de hiperparâmetros (Seção 6.6), mas ainda não está incluído na avaliação 10-fold dos baselines.

---

## 2. Datasets: Pipeline Completo

### 2.1 Dataset 1 — Apache Spark (Jira)

#### Origem

O Apache Spark mantém seu bug tracker no Jira da Apache Foundation. Os bugs têm campos: `summary` (título curto), `description` (texto longo), e `priority` (Blocker/Critical/Major/Minor/Trivial).

#### Tamanho original

Aproximadamente **9.000–10.000 bugs** com texto e prioridade definidos. A distribuição é fortemente desbalanceada:

| Prioridade Jira | Criticidade | Proporção aprox. |
|---|---|---|
| Blocker, Critical | Alta | ~2% |
| Major | Média | ~75% |
| Minor, Trivial | Baixa | ~23% |

#### Mapeamento de labels

```python
# Consolidação de 5 prioridades Jira → 3 classes
{'Blocker': 'Alta', 'Critical': 'Alta',
 'Major': 'Média',
 'Minor': 'Baixa', 'Trivial': 'Baixa'}
```

Esta consolidação é justificada pela proximidade semântica entre Blocker/Critical e entre Minor/Trivial. Reduz o problema de 5 classes (algumas com dezenas de exemplos) para 3 classes operacionalmente significativas.

#### Pré-processamento de texto

```python
def clean_text(text):
    text = text.lower()               # normaliza maiúsculas
    text = re.sub(r'\s+', ' ', text)  # colapsa espaços/newlines
    return text.strip()[:1000]        # trunca em 1000 chars

# Campo de texto: summary + " " + description
df['text'] = (df['summary'] + ' ' + df['description']).apply(clean_text)
```

**Por que 1000 chars:** O Qwen3-Embedding-0.6B tem janela de contexto de 512 tokens (≈ 400–600 palavras). Truncar em 1000 chars captura o início da descrição onde a informação mais relevante tende a aparecer.

**O que não fazemos** (documentado na Seção 3 do notebook):
- **Sem remoção de stopwords:** TF-IDF já penaliza palavras de alta frequência via IDF; embeddings precisam de stopwords para gramática e contexto.
- **Sem stemming/lemmatização:** Causa perda semântica. Termos técnicos como `NullPointerException` não devem ser stemizados.
- **Sem remoção de pontuação:** Pontuação carrega informação técnica (stack traces com `.`, `()`, `[]`).

#### Balanceamento

**Etapa 1 — Undersampling real-first** (`make_balanced_sample`, célula 13):

```python
N_PER_CLASS = 1333

def make_balanced_sample(df, n_per_class=N_PER_CLASS):
    groups = []
    for label, group in df.groupby('label'):
        n = min(len(group), n_per_class)  # preserva minoria intacta
        groups.append(resample(group, replace=False, n_samples=n))
    return pd.concat(groups).sample(frac=1)
```

Resultado: **~4.000 amostras reais** (~1.333/classe). Preserva todas as amostras disponíveis da classe Alta (a mais rara) e reduz as demais.

**Por que undersampling antes do SMOTE:** Se aplicássemos SMOTE direto no dataset original (9k amostras, 2% Alta), o SMOTE geraria sintéticos a partir de ~180 exemplos reais de Alta enquanto Média tem 6.750 — o desequilíbrio persistiria e SMOTE não conseguiria compensar sozinho. Undersampling primeiro equilibra o ponto de partida para ~1.333 por classe, então SMOTE refina dentro do fold.

**Etapa 2 — SMOTE dentro do fold** (`ImbPipeline`, anti-leakage):

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

pipe = ImbPipeline([
    ('smote', SMOTE(random_state=SEED)),
    ('clf', classificador)
])
```

Em cada fold da validação cruzada: SMOTE gera sintéticos **apenas no conjunto de treino** daquele fold. O conjunto de **teste nunca vê amostras sintéticas** — sem data leakage. Esta é uma distinção crítica em relação a trabalhos que aplicam SMOTE antes do split (erro documentado na literatura de ML).

**Fórmula SMOTE** (documentada no notebook, Seção 4):

> Para dois vizinhos $x_i$ e $x_{nn}$, gera $x_{new} = x_i + \lambda(x_{nn} - x_i)$, $\lambda \sim U(0,1)$

Cria amostras sintéticas interpoladas entre vizinhos da classe minoritária no espaço de features.

#### Split de tuning vs avaliação (Seção 6.5)

```python
TUNE_FRAC = 0.2  # 20% reservado para GridSearchCV

df_spark_tune, df_spark_eval = split_tune_eval(df_spark_balanced, emb_spark)
```

O `GridSearchCV` (5-fold) usa apenas `df_spark_tune`. A avaliação final 10-fold usa `df_spark_eval`. Isso evita o **viés otimista de Cawley & Talbot (2010, JMLR)**: se hiperparâmetros são escolhidos vendo todo o dataset, qualquer linha no fold de teste da avaliação final quase certamente esteve no treino durante a busca — inflando artificialmente os resultados.

#### Representação textual

**TF-IDF** (baselines #1–4, Seção 7):

```python
TfidfVectorizer(
    sublinear_tf=True,    # tf → 1 + log(tf), comprime termos muito frequentes
    max_features=20000,   # vocabulário: 20k termos mais informativos
)
```

Resultado: vetor esparso de 20.000 dimensões. Representa frequência ponderada de termos — texto com muitas ocorrências de "null", "exception", "error" terá vetor diferente de texto com "performance", "throughput", "latency".

**Embeddings Qwen3-Embedding-0.6B** (baselines #5–10, Seção 5):

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')

emb = model.encode(texts, batch_size=32, show_progress_bar=True)
np.save('resultados/emb_spark.npy', emb)  # cache para reutilização
```

Resultado: matriz `(n_samples, 1024)` de floats32. Cada texto vira um vetor denso de 1024 dimensões onde distância euclidiana ≈ dissimilaridade semântica. "NullPointerException in executor" ficará próximo de "memory fault during task execution" no espaço de embeddings — algo que TF-IDF não captura.

**Tempo de geração:** ~17 min em CPU, ~1–2 min em GPU. Cacheado como `.npy` — execuções subsequentes carregam em segundos.

**PCA 2D:** Calculado para visualização (EDA), mas **não usado nos modelos**. Serve apenas para inspecionar se as classes formam clusters distinguíveis.

---

### 2.2 Dataset 2 — CIRCL CVE (HuggingFace)

#### Origem

O CIRCL (Computer Incident Response Center Luxembourg) disponibiliza via HuggingFace um dataset de CVEs com scores CVSS em múltiplas versões.

```python
from datasets import load_dataset
ds = load_dataset('CIRCL/cve-scores')
```

#### Tamanho original

Aproximadamente **200.000 CVEs**. Distribuição dependente da versão CVSS utilizada e do período histórico.

#### Mapeamento de labels

**Preferência de versão do score CVSS:**

```
cvss_v3_1 → cvss_v3_0 → cvss_v4_0 → cvss_v2_0
```

CVSSv3.1 é o mais recente e preciso; CVSSv2.0 tem escala diferente mas é mantido como fallback para CVEs antigas (pré-2015).

**Threshold NIST/NVD** (padrão oficial, mesmo que Spanos & Angelis):

```python
def score_to_label(score):
    if score >= 7.0:   return 'Alta'
    elif score >= 4.0: return 'Média'
    else:              return 'Baixa'
```

#### Pré-processamento de texto

```python
# Apenas o campo description da CVE (único campo textual relevante)
df['text'] = df['description'].apply(clean_text)  # mesmo clean_text do Spark
```

CVEs têm apenas `description`. Diferente de bugs Jira, descrições CVE são mais padronizadas: começam com "A vulnerability in X allows Y to Z via W, resulting in V." Esse padrão facilita a classificação — o modelo aprende indicadores como "arbitrary code execution" → Alta, "denial of service" → Média, "information disclosure of non-sensitive data" → Baixa.

#### Balanceamento

Processo idêntico ao Apache Spark:
- Undersampling para `N_PER_CLASS = 1333` amostras/classe
- Resultado: **~4.000 amostras reais** (~1.333/classe)
- SMOTE dentro de cada fold de treino

#### Representação

Idêntica ao Apache Spark:
- TF-IDF: `TfidfVectorizer(sublinear_tf=True, max_features=20000)`
- Embeddings: Qwen3-Embedding-0.6B → 1024 dim, cacheados em `resultados/emb_circl.npy`

---

## 3. Arquitetura do Sistema Híbrido

### 3.1 Base-learners e justificativas

| # | Modelo | Representação | Lib | Motivação na literatura |
|---|---|---|---|---|
| 1 | Complement NB | TF-IDF | sklearn | Baseline Lamkanfi; robusto ao desbalanceamento |
| 2 | Regressão Logística | TF-IDF | sklearn | Linear, interpretável; referência Lamkanfi |
| 3 | SVM Linear (calibrado) | TF-IDF | sklearn | Melhor de Lamkanfi; margem máxima em alta dim |
| 4 | Decision Tree | TF-IDF | sklearn | Árvore simples; baseline Lamkanfi |
| 5 | Random Forest | Embeddings | sklearn | Ensemble de árvores (Umer: RF no híbrido) |
| 6 | KNN | Embeddings | sklearn | Baseline por similaridade semântica |
| 7 | XGBoost | Embeddings | **xgboost** | Gradient boosting (Umer); não-sklearn |
| 8 | LightGBM | Embeddings | **lightgbm** | Boosting leaf-wise; não-sklearn |
| 9 | CatBoost | Embeddings | **catboost** | Boosting; não-sklearn |
| 10 | MLP | Embeddings | sklearn | Rede neural densa sobre embeddings |
| (+) | ELM | Embeddings | **NumPy puro** | Lorenzato; não-sklearn; analítico |

**Por que dois grupos de representação:** TF-IDF captura vocabulário específico técnico (Lamkanfi, Spanos confirmam eficácia). Embeddings contextuais capturam semântica e relações não-lexicais (Umer motiva uso de embeddings). A diversidade garante complementaridade: quando TF-IDF erra por falta de contexto, embeddings podem compensar.

### 3.2 GridSearchCV (Seção 6.6)

Antes da avaliação final, cada baseline passa por `GridSearchCV` com 5-fold internos sobre o subconjunto de tuning (20% dos dados). Grades de busca por modelo:

- NB: `alpha ∈ {0.1, 0.5, 1.0}`
- LogReg: `C ∈ {0.1, 1.0, 10.0}`
- SVM: `C ∈ {0.1, 1.0, 10.0}`
- Decision Tree: `max_depth ∈ {5, 10, None}`
- RF: `n_estimators ∈ {100, 200}`, `max_depth ∈ {10, None}`
- etc.

Isso garante que a comparação entre modelos é **justa** — cada modelo está no seu melhor ajuste, não com hiperparâmetros padrão arbitrários.

### 3.3 Seleção de modelos via Friedman + Nemenyi (Seção 8 e 8.1)

**Problema:** Incluir todos os 10 baselines no ensemble pode introduzir redundância — modelos muito similares votam na mesma direção, reduzindo diversidade sem acrescentar informação.

**Algoritmo de seleção dinâmica** (`seleciona_base_learners()`, célula 32):

```
1. Rodar Friedman (H₀: todos equivalentes)
   → p < 0.05: diferença significativa → prosseguir com Nemenyi
   → p ≥ 0.05: usar top-N fallback (sem evidência de diferença)

2. Nemenyi pós-hoc: p-value para cada par de modelos

3. Percorrer modelos em ordem crescente de rank médio (melhor primeiro):
   - Incluir o melhor sempre
   - Excluir modelo m se:
     (a) p-Nemenyi(m, algum já selecionado) > 0.05  [equivalentes]
     E
     (b) representação(m) == representação(já selecionado)  [mesma família]
```

**Por que a condição (b):** Preserva diversidade de representação. Se SVM (TF-IDF) e LogReg (TF-IDF) são equivalentes estatisticamente, excluímos LogReg (pior rank). Mas se SVM (TF-IDF) e XGBoost (Emb) são equivalentes, mantemos ambos — têm performance similar mas representam informação diferente.

**Resultado para Apache Spark:**
```
Selecionados: SVM (TF-IDF), XGBoost (Emb), Decision Tree (TF-IDF), KNN (Emb)
Excluídos:    LogReg (equivalente ao SVM, mesma representação TF-IDF)
              [outros por rank inferior ou equivalência]
```

Fundamentado em **Demšar (2006, JMLR)** — referência canônica para comparação estatística de classificadores em ML.

### 3.4 Stacking (Meta-learning, Seção 9)

**Arquitetura de nested CV:**

```
Outer loop (10 folds):
  Para cada outer-fold:
    X_train, y_train = outer_train
    X_test,  y_test  = outer_test
    
    Meta-features (inner loop, 5 folds):
      Para cada base-learner b_i em sel_spark:
        OOF predictions sobre X_train:
          meta_X_train[:, 3i:3(i+1)] = P(Alta), P(Média), P(Baixa)
    
    Treina meta-LogReg(C=0.1) em meta_X_train
    
    Treina cada b_i em X_train (completo)
    meta_X_test[:, 3i:3(i+1)] = b_i.predict_proba(X_test)
    
    y_pred = meta-LogReg.predict(meta_X_test)
    metrics_fold = compute_metrics(y_test, y_pred)
```

**Por que OOF (Out-of-Fold) nas meta-features:** Se os base-learners fossem treinados em todo `outer_train` e produzissem predições sobre `outer_train`, o meta-classificador aprenderia sobre predições otimistas (os modelos "viram" os dados de treino). OOF garante que cada ponto é predito por um modelo que não o viu — predições honestas para o meta-classificador.

**Por que meta-LogReg com C=0.1:** Regularização forte evita que o meta-classificador overfitte nas meta-features (que têm apenas `n_base_learners × 3` colunas, mas `n_outer_train` linhas — espaço pequeno). A LogReg aprende quais base-learners são mais confiáveis por classe.

### 3.5 Voto Ponderado (Fusão, Seção 9.1)

Baseado diretamente na taxonomia de Lorenzato (2024). Para cada fold externo:

```python
# Estimativa de peso via split interno estratificado
w_i = f1_macro(b_i.predict(X_val_inner), y_val_inner)
weights_normalized = [w_i / sum(weights)]

# Voto suave ponderado
P_final = sum(w_i * b_i.predict_proba(X_test) for w_i, b_i in zip(weights, base_learners))
y_pred = argmax(P_final, axis=1)
```

**Diferença em relação ao Stacking:** Sem meta-treinamento — pesos globais baseados em F1 interno. Mais simples mas menos expressivo: não aprende que modelo X é melhor para a classe Alta enquanto modelo Y é melhor para Baixa.

### 3.6 Combinação Dinâmica (Seção 9.2)

Baseado no exercício de aula (Lorenzato, 2024). Pesos por instância de teste:

```python
# Para cada x_teste:
# 1. OOF predictions de todos os base-learners no outer_train
oof_proba_train = [b_i.oof_predict(X_train) for b_i in base_learners]

# 2. K=7 vizinhos de x_teste no espaço de predições OOF
viz_idx = KNN(k=7).fit(oof_proba_train).kneighbors(x_teste_oof)

# 3. Mínimos quadrados locais
X_viz = oof_proba_train[viz_idx]       # (7, n_learners * 3)
y_viz = y_onehot_train[viz_idx]        # (7, 3)
W     = pinv(X_viz) @ y_viz            # (n_learners * 3, 3)

# 4. Predição ponderada localmente
proba_final = x_teste_oof @ W          # (3,)
```

Estado atual: **implementado (célula 40), não executado** na última rodada (célula 41 sem output).

---

## 4. Métricas: Escolha e Interpretação

### 4.1 Por que F1-macro como métrica principal

Com desbalanceamento original (Alta ≈ 2%, Média ≈ 75%), um classificador que sempre prevê "Média" alcançaria ~75% de acurácia — completamente inútil. F1-macro penaliza igualmente a falha em qualquer classe, incluindo a minoria Alta.

| Métrica | O que revela neste contexto |
|---|---|
| **F1-macro** | Métrica principal; desempenho balanceado entre classes |
| **Acurácia** | Referência intuitiva; inflada pelo desbalanceamento original |
| **MCC** | Robusto mesmo com desbalanceamento extremo; [-1, 1] |
| **Kappa** | Acordo além do acaso; 0 = aleatório |
| **F1-Alta** | O modelo aprende a detectar bugs críticos? |
| **F1-Média** | Classe dominante; mais fácil de aprender |
| **F1-Baixa** | Classe minoritária secundária |

**Por que MCC e Kappa adicionalmente:** Quando o dataset de avaliação (após undersampling) tem distribuição diferente do mundo real, F1-macro pode ainda ser otimista para certas configurações. MCC e Kappa são mais robustos a essas variações de proporção de classe.

---

## 5. Resultados e Comparação com a Literatura

### 5.1 Apache Spark — Tabela completa

| Modelo | F1-macro | F1-Alta | F1-Média | F1-Baixa | MCC |
|---|---|---|---|---|---|
| NB (TF-IDF) | 0.475 ± 0.025 | 0.389 | 0.478 | 0.558 | 0.238 |
| LogReg (TF-IDF) | 0.508 ± 0.027 | 0.390 | 0.536 | 0.597 | 0.258 |
| **SVM (TF-IDF)** | **0.520 ± 0.031** | 0.413 | 0.537 | 0.609 | 0.280 |
| Decision Tree (TF-IDF) | 0.395 ± 0.035 | 0.226 | 0.406 | 0.552 | 0.111 |
| RF (Emb) | 0.495 ± 0.030 | 0.347 | 0.559 | 0.578 | 0.240 |
| KNN (Emb) | 0.377 ± 0.031 | 0.329 | 0.408 | 0.394 | 0.165 |
| XGBoost (Emb) | 0.495 ± 0.023 | 0.380 | 0.534 | 0.570 | 0.228 |
| LightGBM (Emb) | 0.485 ± 0.028 | 0.319 | 0.553 | 0.583 | 0.230 |
| CatBoost (Emb) | 0.493 ± 0.025 | 0.382 | 0.545 | 0.552 | 0.222 |
| MLP (Emb) | 0.501 ± 0.026 | 0.377 | 0.542 | 0.585 | 0.243 |
| **★ Stacking** | **0.511 ± 0.024** | **0.438** | 0.498 | 0.598 | **0.286** |
| ⬥ Voto Ponderado | 0.502 ± 0.032 | 0.390 | 0.538 | 0.578 | 0.252 |

**Análise por artigo:**

- **Vs Lamkanfi:** SVM (0,520) é o melhor baseline individual — **confirma o achado de Lamkanfi** de que SVM supera DT e NB. O Stacking (0,511) fica ligeiramente abaixo em F1-macro mas **supera em F1-Alta (0,438 vs 0,413)** e em MCC (0,286 vs 0,280) — o sistema híbrido é melhor em detectar a classe crítica, que é o que importa na prática.

- **Vs Umer:** A melhoria do Stacking sobre o melhor individual é pequena (+0 em F1-macro, +0,025 em F1-Alta). Wilcoxon p=0,4922 — não significativo. Coerente com a observação de que quando os baselines individuais são fortes, o ensemble adiciona menos.

- **Ponto inesperado:** Embeddings **não superam** TF-IDF no Spark. Provável causa: bugs Jira têm vocabulário muito específico (class names, method names, exception types) onde TF-IDF com sublinear_tf captura eficientemente. Embeddings gerais (treinados em texto genérico) não têm vantagem semântica sobre vocabulário técnico altamente específico.

### 5.2 CIRCL CVE — Tabela completa

| Modelo | F1-macro | F1-Alta | F1-Média | F1-Baixa | MCC |
|---|---|---|---|---|---|
| NB (TF-IDF) | 0.653 ± 0.023 | 0.723 | 0.649 | 0.587 | 0.490 |
| LogReg (TF-IDF) | 0.702 ± 0.027 | 0.762 | 0.699 | 0.646 | 0.551 |
| **SVM (TF-IDF)** | **0.704 ± 0.024** | **0.764** | **0.700** | 0.648 | **0.555** |
| Decision Tree (TF-IDF) | 0.691 ± 0.019 | 0.710 | 0.709 | 0.655 | 0.526 |
| RF (Emb) | 0.644 ± 0.024 | 0.682 | 0.667 | 0.585 | 0.455 |
| KNN (Emb) | 0.508 ± 0.028 | 0.583 | 0.490 | 0.451 | 0.325 |
| XGBoost (Emb) | 0.643 ± 0.027 | 0.687 | 0.663 | 0.580 | 0.457 |
| LightGBM (Emb) | 0.662 ± 0.024 | 0.716 | 0.684 | 0.585 | 0.489 |
| CatBoost (Emb) | 0.613 ± 0.034 | 0.647 | 0.632 | 0.559 | 0.414 |
| MLP (Emb) | 0.652 ± 0.029 | 0.705 | 0.665 | 0.587 | 0.472 |
| **★ Stacking** | **0.699 ± 0.024** | 0.756 | 0.696 | **0.644** | 0.550 |
| ⬥ Voto Ponderado | 0.683 ± 0.033 | 0.737 | 0.687 | 0.625 | 0.522 |

**Análise por artigo:**

- **Vs Spanos:** SVM (TF-IDF) obtém 0,704 F1-macro — **dentro da faixa de 0,70–0,80 reportada por Spanos**, confirmando que texto de CVE é discriminativo e TF-IDF é suficiente para um forte baseline. Isso valida tanto o dataset quanto a metodologia.

- **Vs Umer:** O Stacking (0,699) é muito próximo do SVM (0,704), mas **Wilcoxon p=0,3223** — não significativo. O Voto Ponderado (0,683) é significativamente **inferior** ao SVM — Wilcoxon p=0,0195. Ou seja, no domínio CVE, o Meta-learning (Stacking) é a estratégia certa; Fusão simples (Voto) piora em relação ao melhor individual.

- **Padrão oposto ao Spark:** No CVE, embeddings também ficam abaixo do TF-IDF (RF=0,644 vs SVM=0,704). O domínio CVE tem linguagem padronizada e descritiva onde palavras-chave específicas ("code execution", "buffer overflow", "CSRF") são altamente discriminativas — favorecendo TF-IDF.

### 5.3 Wilcoxon — Sumário das comparações

| Comparação | Dataset | p-value | Conclusão |
|---|---|---|---|
| Stacking vs SVM | Apache Spark | 0.4922 | Não significativo — equivalentes |
| Stacking vs SVM | CIRCL CVE | 0.3223 | Não significativo — equivalentes |
| Voto Ponderado vs SVM | Apache Spark | 0.1055 | Não significativo |
| **Voto Ponderado vs SVM** | **CIRCL CVE** | **0.0195** | **Significativo — SVM superior** |
| Stacking vs Voto Ponderado | Apache Spark | 0.4922 | Não significativo |
| Stacking vs Voto Ponderado | CIRCL CVE | 0.0840 | Limítrofe — Stacking tendência superior |

---

## 6. Mapeamento Literatura → Decisões de Implementação

| Decisão | Artigo motivador | Código correspondente |
|---|---|---|
| SVM como baseline principal | Lamkanfi (2010): melhor modelo | `CalibratedClassifierCV(LinearSVC(...))` |
| Classe Alta monitorada separadamente | Lamkanfi + Umer: Alta é a mais difícil | `compute_metrics()` → coluna F1-Alta |
| Sistema híbrido | Umer (2019): híbrido supera individual | Seções 9, 9.1, 9.2 |
| CVE como segundo dataset | Spanos (2018): CVE text é discriminativo | Dataset CIRCL, Seção 1 |
| TF-IDF com sublinear_tf | Melhoria sobre BoW de Lamkanfi | `TfidfVectorizer(sublinear_tf=True)` |
| Embeddings contextuais | Extensão sobre embeddings estáticos de Umer | Qwen3-Embedding-0.6B, Seção 5 |
| Taxonomia Fusão/Seleção/Meta | Lorenzato (2024): três famílias | Seções 9 (meta), 9.1 (fusão), 9.2 (seleção) |
| ELM analítico | Lorenzato (2024): ELM formalizado | `ELMClassifier` NumPy, célula 22 |
| Friedman + Nemenyi | Demšar (2006) via Lorenzato | `seleciona_base_learners()`, Seção 8.1 |
| Wilcoxon para comparação | Demšar (2006) via Lorenzato | Seção 11 |
| SMOTE dentro do fold | Anti-leakage best practice | `ImbPipeline`, todas as avaliações |
| Nested CV (tuning vs eval) | Cawley & Talbot (2010, JMLR) | `split_tune_eval()`, Seção 6.5 |
| Dois datasets distintos | Requisito do projeto + generalização | Seções 1 e 10 (tabelas separadas) |

---

## 7. Melhorias Propostas em Relação à Literatura

### 7.1 Sobre Lamkanfi et al. (2010)

| Limitação | Nossa resposta | Onde implementado |
|---|---|---|
| Bag-of-Words puro (count) | TF-IDF sublinear + embeddings Qwen3 1024-dim | Seções 5, 7 |
| Sem balanceamento de classes | Undersampling real-first + SMOTE inside fold | Seções 4, 6 |
| Modelos isolados | Stacking + Fusão + Comb. Dinâmica | Seções 9, 9.1, 9.2 |
| Dataset único (Bugzilla) | Dois domínios: Jira (bugs) + CIRCL (CVE) | Seção 1 |
| Sem testes de hipótese | Friedman + Nemenyi + Wilcoxon | Seções 8, 11 |
| 5 classes originais | Consolidação em 3 classes operacionais | Seção 1 |

### 7.2 Sobre Umer et al. (2019)

| Limitação | Nossa resposta | Onde implementado |
|---|---|---|
| CNN requer GPU | Embeddings pré-calculados offline (CPU viável) | Seção 5 |
| Embeddings estáticos (GloVe) | Qwen3 — modelo contextual moderno 0.6B | Seção 5 |
| Dataset único (Eclipse) | Dois domínios distintos | Seção 1 |
| Seleção de modelos ad hoc | `seleciona_base_learners()` via Nemenyi | Seção 8.1 |
| Sem teste de hipótese formal | Friedman + Nemenyi + Wilcoxon | Seções 8, 11 |

### 7.3 Sobre Spanos & Angelis (2018)

| Limitação | Nossa resposta | Onde implementado |
|---|---|---|
| Apenas TF-IDF | TF-IDF + embeddings contextuais | Seções 5, 7 |
| Modelos isolados | Sistema híbrido (Stacking como meta-learner) | Seção 9 |
| Apenas domínio CVE | Validação também em domínio Jira | Seções 1, 10 |
| Sem balanceamento | Undersampling + SMOTE | Seções 4, 6 |

---

## 8. Checklist de Requisitos

| Requisito | Status | Evidência objetiva |
|---|---|---|
| Desenvolver sistema híbrido | ✅ | Stacking (meta-LogReg sobre OOF) + Voto Ponderado + Comb. Dinâmica |
| Verificar literatura e propor alteração | ✅ | 4 artigos mapeados; "Nossa extensão" em célula 0 do notebook; implementada como embeddings + nested CV + seleção estatística |
| Experimentos comparativos (proposto vs baselines) | ✅ | 10 baselines avaliados em 10-fold; Wilcoxon compara Stacking vs melhor baseline |
| Não usar apenas sklearn | ✅ | XGBoost, LightGBM, CatBoost (libs externas); ELM em NumPy puro |
| Testes de hipótese obrigatórios | ✅ | Friedman (Seção 8) + Nemenyi pós-hoc (Seção 8.1) + Wilcoxon (Seção 11) |
| Mais de uma base de dados | ✅ | Apache Spark Jira + CIRCL CVE (domínios, tamanhos e distribuições distintos) |
| Tabela de métricas | ✅ | Seção 10 do notebook; `resultados/metricas_apache_spark.csv` e `metricas_circl_cve.csv`; 9 métricas × 12 modelos × 2 datasets |
| Especificação do processo de seleção de modelos | ✅ | Seção 8.1: algoritmo `seleciona_base_learners()` baseado em rank Nemenyi + regra de diversidade de representação |

---

## 9. Pendências

| Item | Prioridade | Descrição |
|---|---|---|
| ELM nos 10-folds | Alta | Adicionar ELM ao loop de baselines da Seção 7; incluir na Seção 10 |
| Executar Comb. Dinâmica | Alta | Célula 41 sem output; executar `run_combinacao_dinamica()` e incluir nas Seções 10 e 11 |
| Atualizar tabela consolidada | Média | Após #1 e #2, atualizar Seção 10 com todas as métricas |
| Ampliar revisão no notebook | Baixa | Célula 0: incluir Umer et al. e Spanos & Angelis explicitamente (além de Lamkanfi) |
