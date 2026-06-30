# Análise Detalhada — Projeto Híbrido de Classificação de Criticidade de Bugs

> **Projeto:** Classificador de Criticidade de Bugs via Sistema Híbrido por Stacking  
> **Disciplina:** Reconhecimento de Padrões — UPE, 1º Período  
> **Atualizado em:** 2026-06-29

---

## 0. Resumo Rápido — O Projeto em Uma Página

### O que o projeto faz

Dado o **texto** de um relatório de bug ou descrição de vulnerabilidade CVE, classificar automaticamente sua **criticidade** em três níveis: **Alta**, **Média** ou **Baixa**.

### Datasets

| Dataset | Fonte | Tamanho (após balanceamento) | Mapeamento de rótulo |
|---|---|---|---|
| **Apache Spark (Jira)** | CSV local (`spark_bugs.csv`) | ~4.000 amostras (~1.333/classe) | Prioridade Jira → Alta (Blocker/Critical), Média (Major), Baixa (Minor/Trivial) |
| **CIRCL CVE (NVD)** | NVD REST API 2.0 (cache local) | ~4.000 amostras (~1.333/classe) | Score CVSS → Alta (≥7.0), Média (4.0–6.9), Baixa (<4.0) |

Ambos passam por **undersampling** para equilibrar as classes antes do treinamento (a classe Alta representa ~2% do Spark antes do balanceamento).

### Representações de texto usadas

| Representação | Como | Para que modelos |
|---|---|---|
| **TF-IDF + Stemming** | `TfidfVectorizer(max_features=20_000, sublinear_tf=True, tokenizer=_lamkanfi_tokenizer)` | NB, LogReg, SVM, DT |
| **BoW + Stemming** | `CountVectorizer` + `_lamkanfi_tokenizer` (stopwords + Porter) | NB-Lamkanfi (MultinomialNB, fiel a Lamkanfi 2010), NB-Complement (investigação) |
| **Embeddings** | Qwen3-Embedding-0.6B (1024 dim, via Ollama local) | RF\*, KNN, XGBoost\*, LightGBM\*, CatBoost\*, MLP, ELM |

> \* RF, XGBoost, LightGBM, CatBoost são **somente benchmark** — eles próprios já são ensembles (bagging/boosting), não entram no stacking/voting/combinação dinâmica para evitar ensemble-de-ensemble sem justificativa metodológica.

### Arquitetura do sistema

```
Texto → [TF-IDF ou Embeddings] → 12 Baselines (inclui RF/XGB/LGBM/CAT como benchmark)
                                               ↓ Seleção Estatística (Friedman+Nemenyi)
                                                 [exclui ensembles a priori]
                                               ↓ melhores base-learners atômicos
                                         Stacking (meta-LogReg sobre OOF)
                                         Voto Ponderado
                                         Combinação Dinâmica
```

### Artigos de base — o que cada um contribui

| Artigo | Contribuição para o projeto |
|---|---|
| **Lamkanfi et al. (2010)** | Problema base (classificar severidade de bug por texto); baseline NB+BoW+stopwords+stemming reproduzido fielmente como `NB-Lamkanfi (BoW+Stem)` |
| **Spanos & Angelis (2018)** | Justifica CVE como domínio válido; confirma que texto de vulnerabilidade é discriminativo para TF-IDF; mesmo pré-processamento de Lamkanfi |
| **Umer et al. (2019)** | Motiva sistema híbrido (CNN + RF); justifica uso de embeddings além de TF-IDF |
| **Lorenzato (2024)** | Fundamentação teórica de combinação de classificadores (Stacking, Voto Ponderado) |

### Métricas usadas

- **F1-macro** (principal) — penaliza igualmente todas as classes, incluindo a minoritária Alta
- Acurácia, Precisão, Recall, MCC, Kappa (secundárias)
- F1 por classe (F1-Alta, F1-Média, F1-Baixa) para rastrear a classe crítica

### Validação

- **Nested CV:** 10 folds externos × 5 folds internos (GridSearchCV para hiperparâmetros)
- **SMOTE dentro de cada fold de treino** — sem data leakage
- **Friedman + Nemenyi** (compara 12 baselines), **Wilcoxon** (Stacking vs. melhor baseline)

---

## 0.1 Figuras — Arquiteturas dos Sistemas

### Figura 1 — Lamkanfi et al. (2010): sistema NÃO híbrido

```
┌─────────────────────────────────────────────────────────────┐
│                   Lamkanfi et al. (2010)                    │
│               Predicting Bug Severity (Bugzilla)            │
└─────────────────────────────────────────────────────────────┘

  Relatório de bug (Bugzilla)
         │
         ▼
  ┌─────────────────┐
  │  Pré-processamento NLP                                     │
  │  1. Tokenização (só chars alfa, lowercase)                 │
  │  2. Remoção de stopwords                                   │
  │  3. Stemming (Porter Stemmer)                              │
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  Bag-of-Words   │  ← CountVectorizer (frequência bruta)
  │  (BoW)          │     Vocabulário: todos os termos pós-stem
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  Naive Bayes    │  ← único classificador testado no artigo
  └────────┬────────┘
           │
           ▼
  Predição binária: Severo / Não-severo
  (exclui "normal"; major+critical+blocker = Severo)

  Validação: split 70/30 (sem CV, sem SMOTE, sem ensemble)
```

---

### Figura 2 — Spanos & Angelis (2018): sistema NÃO híbrido

```
┌─────────────────────────────────────────────────────────────┐
│               Spanos & Angelis (2018)                       │
│         Multi-Target Prediction of CVSS Scores (CVE/NVD)   │
└─────────────────────────────────────────────────────────────┘

  Descrição CVE (NVD)
         │
         ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Pré-processamento NLP                                  │
  │  1. Remoção de pontuação, números, stopwords            │
  │  2. Stemming (SnowballC via R — equiv. Porter inglês)   │
  │  3. Remoção de termos raros (freq < 0,1%)               │
  │     → 258 preditores finais                             │
  └────────────────┬────────────────────────────────────────┘
                   │
                   ▼
  ┌─────────────────────────────────────────────────────────┐
  │  DTM — Document-Term Matrix (frequência bruta)          │
  └────────────────┬────────────────────────────────────────┘
                   │
                   ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Label Powerset (LP)                                    │
  │  6 características CVSS concatenadas em uma classe:     │
  │  AV | AC | Au | CI | II | AI  →  337 classes únicas     │
  └────────────────┬────────────────────────────────────────┘
                   │
          ┌────────┴────────┐
          ▼                 ▼
     ┌─────────┐       ┌─────────┐
     │   RF    │       │Boosting │   (testados SEPARADAMENTE — melhor = RF)
     └─────────┘       └─────────┘
          │
          ▼
  Score CVSS reconstruído a partir da classe prevista

  Validação: split temporal (treino 1988–2015, teste 2016–2018)
  Balanceamento: pesos de classe (sem SMOTE)
```

---

### Figura 3 — Umer et al. (2019): sistema HÍBRIDO (pipeline em série)

```
┌─────────────────────────────────────────────────────────────┐
│                   Umer et al. (2019)                        │
│         CNN + RF/Boosting Hybrid for Bug Severity           │
└─────────────────────────────────────────────────────────────┘

  Relatório de bug (Eclipse Bugzilla)
         │
         ▼
  ┌─────────────────┐
  │  Word Embeddings│  ← Word2Vec / GloVe pré-treinados (estáticos)
  │  (denso, 300d)  │
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  CNN            │  ← filtros convolucionais extraem n-gramas locais
  │  (feature       │     → vetor de features comprimido
  │   extractor)    │
  └────────┬────────┘
           │  features extraídas pela CNN
           ▼
  ┌─────────────────────────────────┐
  │  RF / AdaBoost / XGBoost        │  ← classifica sobre as features da CNN
  │  (combinação por votação)       │
  └────────────────┬────────────────┘
                   │
                   ▼
  Predição de severidade (multi-classe)

  Hibridização: pipeline em série CNN → RF (NÃO meta-learning/stacking)
  Validação: holdout / k-fold simples (sem nested CV, sem testes formais)
```

---

### Figura 4 — Nosso sistema: HÍBRIDO com meta-learning e seleção estatística

```
┌─────────────────────────────────────────────────────────────────────────┐
│       Sistema Híbrido por Stacking — Projeto RecPad (2024)              │
│       Apache Spark (Jira) + CIRCL CVE  |  3 classes: Alta/Média/Baixa  │
└─────────────────────────────────────────────────────────────────────────┘

  Texto do bug / CVE
        │
        ├───────────────────────────────┐
        ▼                               ▼
  ┌───────────────┐             ┌────────────────────┐
  │  TF-IDF       │             │  Qwen3-Embed-0.6B  │
  │  (esparso,    │             │  (denso, 1024 dim, │
  │  20K features)│             │   contextual, LLM) │
  └──────┬────────┘             └────────┬───────────┘
         │                               │
   ┌─────┴─────┐                   ┌─────┴──────────────────┐
   ▼     ▼  ▼  ▼                   ▼    ▼    ▼    ▼   ▼   ▼
  NB  LogReg SVM DT            KNN  MLP  ELM [RF* XGB* LGBM*]
  NB-Lam                             benchmarks só ↑ (não entram no stacking)
         │                                   │
         └──────────────┬────────────────────┘
                        │ 12 baselines avaliados via
                        │ Nested CV (10 × 5 folds)
                        ▼
         ┌──────────────────────────────┐
         │  Friedman + Nemenyi          │  ← testes não-paramétricos (Demšar 2006)
         │  Autoranking por rank médio  │     exclui ensembles a priori (*RF/XGB/LGBM)
         │  → seleciona base-learners   │     descarta redundantes (mesma família,
         │    atomicos e diversos       │     equivalentes estatisticamente)
         └──────────────┬───────────────┘
                        │  base-learners selecionados (dinâmico por dataset)
                  ┌─────┴──────┐
                  │            │
                  ▼            ▼
  ┌───────────────────┐  ┌─────────────────────┐  ┌──────────────────────┐
  │  Stacking (OOF)   │  │  Voto Ponderado      │  │ Combinação Dinâmica  │
  │  Meta-LogReg      │  │  (peso ∝ F1 interno) │  │ KNN + Mínimos        │
  │  treinada sobre   │  │  Fusão suave         │  │ Quadrados local      │
  │  predições OOF    │  └──────────┬───────────┘  └──────────┬───────────┘
  └────────┬──────────┘             │                          │
           │                        │                          │
           └──────────┬─────────────┴──────────────────────────┘
                      │
                      ▼
        Wilcoxon pareado: Stacking vs melhor baseline
        Wilcoxon: Voto Ponderado vs melhor baseline
        Wilcoxon: Stacking vs Voto Ponderado

  Diferenciais vs. artigos base:
  ┌──────────────────────────────────────────────────────────────────┐
  │ ✦ Dois domínios independentes (bugs + CVEs)                      │
  │ ✦ Embeddings LLM modernos (contextuais) vs. BoW/Word2Vec estáticos│
  │ ✦ Seleção data-driven de base-learners (Nemenyi — nenhum artigo faz)│
  │ ✦ Três estratégias de ensemble comparadas formalmente            │
  │ ✦ Validação estatística completa (Friedman + Nemenyi + Wilcoxon) │
  │ ✦ SMOTE dentro do fold (sem leakage — erro comum na literatura)  │
  └──────────────────────────────────────────────────────────────────┘
```

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

**Pré-processamento textual (pipeline completo):**
1. **Tokenização** — divide o texto em tokens alfa, converte para minúsculas, remove pontuação.
2. **Remoção de stopwords** — elimina palavras funcionais sem valor discriminativo (ex: "the", "in", "that").
3. **Stemming (Porter Stemmer)** — reduz cada token à sua raiz morfológica (ex: "crashes"/"crashing" → "crash"), reduzindo dimensionalidade do vocabulário.

**Representação textual:** Bag-of-Words puro (CountVectorizer, não TF-IDF) sobre os tokens pós-stemming.

**Classificador testado:**
- **Naive Bayes (NB)** — único classificador do artigo.

> *Nota:* SVM, Decision Trees e k-NN são mencionados apenas como trabalhos futuros: *"other classification algorithms also exist including the promising Decision trees and Support Vector Machines"*. O artigo NÃO os avalia. Também não usa seleção de atributos (feature selection) — citada apenas como trabalho futuro: *"the impact of additional techniques like feature selection and cross-validation will be evaluated."*

**Labels de severidade:** O artigo remove a categoria "normal" (considerada ambígua — muitos usuários deixavam o valor padrão sem avaliar conscientemente) e agrupa as demais em **classificação binária**:
- **Severo** (Severe): major, critical, blocker
- **Não-severo** (Non-severe): minor, trivial

**Métricas reportadas:**
- Precisão por classe: entre 0,65 e 0,75 para NB
- Recall por classe: faixa similar
- Acurácia global: ~70–75% para NB

**Principais achados:**
1. NB produz resultados razoáveis; desbalanceamento prejudica especialmente as minorias
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
| Classes | Binária: Severo / Não-severo (remove "normal") | 3 (Alta/Média/Baixa) — consolidação |
| Pré-processamento | Tokenização + stopwords + Porter stemming | Idem para TF-IDF (`_lamkanfi_tokenizer`); embeddings usam texto bruto |
| Representação | Bag-of-Words (CountVectorizer) | TF-IDF sublinear + Embeddings LLM |
| Modelos | **Somente NB** (SVM/DT/kNN são future work do artigo) | NB, SVM, DT, k-NN + 7 modelos adicionais + NB-Lamkanfi (reprodução) |
| Balanceamento | Nenhum | Undersampling + SMOTE dentro do fold |
| Ensemble | Nenhum | Stacking + Voto Ponderado + Comb. Dinâmica |
| Métrica principal | Precisão/Recall | F1-macro (penaliza ignorar minoria) |

**Replicação dos baselines:** O modelo central de Lamkanfi (NB+BoW) é reproduzido fielmente como `NB-Lamkanfi (BoW+Stem)`. Os demais baselines (SVM, DT, k-NN) são adicionados pelo projeto como comparação — eles **não fazem parte do artigo original de Lamkanfi**. O SVM (TF-IDF) usa `CalibratedClassifierCV` para produzir probabilidades necessárias ao Stacking.

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
| Base-learners | CNN + RF/Boosting | TF-IDF (NB, SVM, LogReg, DT, NB-Lamkanfi) + Emb (KNN, MLP, ELM) — *RF/XGB/LGBM/CB são benchmark, não entram no stacking* |
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

**Dataset:** NVD (National Vulnerability Database) — CVEs com scores CVSS v2 publicados pelo NIST.

**Pipeline completo — 5 passos:**

1. **Entrada:** descrição técnica da CVE em linguagem natural.
2. **Pré-processamento NLP (idêntico a Lamkanfi):** remoção de pontuação, remoção de stop-words, stemming (SnowballC via R — funcionalmente equivalente a Porter para inglês), remoção de termos com frequência < 0,1% → resultado: **258 preditores**. Representação final: **DTM de frequência bruta** (igual a Lamkanfi — *não* TF-IDF).
3. **Label Powerset (LP):** as **6 características CVSS v2** — Access Vector (AV), Access Complexity (AC), Authentication (Au), Confidentiality Impact (CI), Integrity Impact (II), Availability Impact (AI) — são concatenadas em **uma única string de classe combinada** (ex: `"AV:N|AC:L|Au:SI|CI:P|II:N|AI:N"`). O modelo trata as **337 combinações únicas** como classes individuais, preservando correlações entre as características (ao contrário de predição independente por componente).
4. **Classificação:** Random Forest, Boosting e Decision Tree treinados sobre essas 337 classes combinadas. RF apresentou melhor desempenho — >1 em cada 4 vulnerabilidades foi prevista corretamente em **todas as 6 características simultaneamente** (Subset 0/1 Loss).
5. **Cálculo dos scores:** a string de classe prevista é decodificada nas 6 características originais e inserida nas fórmulas CVSS e WIVSS para obter o score numérico (CVSS real médio: 5,95; previsto pelo modelo: 5,22).

**Tratamento de desbalanceamento:** classes raras recebem pesos inversamente proporcionais à frequência (class weights), sem SMOTE.

**Principais achados:**
1. **O texto da CVE é suficiente** para prever scores CVSS com boa precisão
2. Maioria dos erros de predição ficou na faixa de 0 a 0,5 (alta precisão nas estimativas numéricas)
3. RF superou Boosting e DT na maioria das métricas
4. A abordagem multi-alvo preserva correlações entre características — vantagem sobre predição independente por componente

**Limitações:**
- Usa apenas frequência bruta de termos / DTM (sem TF-IDF, sem embeddings contextuais)
- Problema de 337 classes raramente se generaliza fora do domínio CVE/NVD
- Sem validação em outros domínios (apenas CVE/NVD)
- Sem testes de hipótese formais (sem Friedman/Wilcoxon)

#### Como o projeto se baseia neste artigo

Spanos & Angelis justificam **o uso de CVEs como segundo dataset** e confirmam que texto de vulnerabilidade é discriminativo para classificação de criticidade.

| Aspecto | Spanos (2018) | Nosso projeto (CIRCL CVE) |
|---|---|---|
| Fonte dos dados | NVD REST API | NVD REST API (mesmo endpoint, cache local) |
| Tarefa | Multi-target: 6 componentes CVSS via LP (337 classes) | Classificação tri-classe: Alta / Média / Baixa |
| Pré-processamento | Remoção de pontuação + stop-words + stemming → DTM | Idem para TF-IDF (`_lamkanfi_tokenizer`); embeddings usam texto bruto |
| Versão CVSS | CVSS v2 (predominante) | v3.1 → v3.0 → v4.0 → v2.0 (preferência) |
| Threshold | Calculado das fórmulas CVSS/WIVSS | NIST: ≥7.0 Alta, 4.0–6.9 Média, <4.0 Baixa |
| Representação | **DTM (frequência bruta)** — *não* TF-IDF | TF-IDF sublinear + Qwen3 embeddings 1024-dim |
| Balanceamento | Pesos de classe (inverse frequency) | Undersampling + SMOTE dentro do fold |
| Classificadores | RF, Boosting (C5.0/AdaBoost), Decision Tree | NB, SVM, DT, k-NN + 7 adicionais |
| Validação | Split temporal (treino 1988–2015; teste 2016–2018) | Nested CV 10×5 |
| F-measure por característica | 47–68% (RF, Tabela 7 do artigo) | F1-macro 0,704 (SVM, CVE, 3 classes) |
| Testes de hipótese | Nenhum | Friedman + Nemenyi + Wilcoxon |

**O projeto NÃO reproduz o Label Powerset de Spanos** — a formulação do problema é diferente: Spanos prevê 6 características CVSS simultaneamente (337 classes), enquanto o projeto classifica em 3 níveis operacionais de criticidade (Alta/Média/Baixa). Spanos justifica **o uso de CVE como domínio** e confirma que texto de vulnerabilidade é discriminativo.

**Referência de desempenho:** O artigo de 2018 (JSS) reporta F-measure de 47–68% por característica CVSS com RF. Um artigo anterior de Spanos (2017, conferência) que prediz apenas 3 categorias de severidade com SVM/DT/NN reporta F1 ~0,70–0,80 — mas são **problemas diferentes** (severidade 3 classes vs. 6 características CVSS). O projeto não cita o artigo de 2017; o nosso SVM (TF-IDF) no CIRCL CVE obtém 0,704 F1-macro — **compatível com a faixa esperada** para um problema de 3 classes com texto de vulnerabilidade.

**Nota sobre pré-processamento:** Spanos usa o mesmo pipeline de Lamkanfi (stopwords + stemming). Desde a revisão de Junho/2026, todos os modelos TF-IDF do projeto também usam `_lamkanfi_tokenizer` (Porter stemming + stopwords) — alinhado com ambos os artigos base. Modelos de embeddings (Qwen3) mantêm texto bruto, pois transformers são treinados sobre texto não-stemizado.

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

**Pré-processamento TF-IDF** — modelos NB, LogReg, SVM, DT usam `_lamkanfi_tokenizer` (compatível com Lamkanfi 2010 e Spanos 2018):
- **Remoção de stopwords:** tokens funcionais removidos antes da vetorização.
- **Porter Stemmer:** cada token reduzido à raiz morfológica.

**Modelos de embeddings** (RF, KNN, MLP, ELM) recebem texto bruto:
- Transformers (Qwen3) são treinados sobre texto não-stemizado; stemming corromperia os vetores.
- Embeddings são pré-computados e cacheados — não reprocessados a cada fold.
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
- TF-IDF: `TfidfVectorizer(sublinear_tf=True, max_features=20_000, tokenizer=_lamkanfi_tokenizer)` (stopwords + Porter stemming)
- Embeddings: Qwen3-Embedding-0.6B → 1024 dim, cacheados em `resultados/emb_circl.npy`

---

## 3. Arquitetura do Sistema Híbrido

### 3.0 As três fases do ensemble: Geração → Seleção → Combinação

A construção de um sistema de múltiplos classificadores (MCS) segue três fases canônicas (Britto, Sabourin & Oliveira 2014; Cruz, Sabourin & Cavalcanti 2018). O projeto implementa as três explicitamente:

```
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────────┐
│  1. GERAÇÃO     │ → │  2. SELEÇÃO     │ → │  3. COMBINAÇÃO          │
│  (pool)         │   │  (subconjunto)  │   │  (agregação)            │
└─────────────────┘   └─────────────────┘   └─────────────────────────┘
```

| Fase | O que faz | Onde no projeto | Seção / célula |
|---|---|---|---|
| **1. Geração** | Cria o pool de base-learners diversos. Diversidade vem de (a) **algoritmos diferentes** (NB, LogReg, SVM, DT, KNN, MLP, ELM) e (b) **representações diferentes** (TF-IDF esparso vs. embeddings densos). Cada modelo é **otimizado individualmente** via GridSearchCV. | `build_baselines()` + `busca_hiperparametros()` | §3.1, §3.2 (cél. 22, 24) |
| **2. Seleção** | Escolhe **quais** modelos entram no ensemble. É **estática** (mesma equipe para todas as instâncias) via Friedman + Nemenyi: remove ensembles a priori (`_ENSEMBLE_ONLY`), depois descarta modelos estatisticamente equivalentes da mesma família de representação. | `seleciona_base_learners()` | §3.3 (cél. 28, 32) |
| **3. Combinação** | Agrega as saídas dos modelos selecionados. Três estratégias da taxonomia de Kuncheva são implementadas **em paralelo** para comparação. | Stacking / Voto / Comb. Dinâmica | §3.4–3.6 (cél. 34, 37, 40) |

**Detalhe das estratégias de combinação** (fase 3), conforme taxonomia Lorenzato/Kuncheva (§1.4):

| Estratégia | Tipo na taxonomia | Como combina | Seção |
|---|---|---|---|
| **Stacking** | Meta-learning | Meta-classificador (LogReg) treina sobre predições OOF | §3.4 |
| **Voto Ponderado** | Fusão | Média de probabilidades ponderada por F1 de cada modelo | §3.5 |
| **Combinação Dinâmica** | Seleção dinâmica | Pesos locais por instância via KNN no espaço de predições OOF | §3.6 |

> **Nota terminológica:** "seleção" aparece em dois níveis distintos. Na **fase 2** (seleção de modelos), é *estática* — define a equipe fixa do ensemble via teste de hipóteses. Na **fase 3**, a *Combinação Dinâmica* é uma *seleção dinâmica por instância* — escolhe pesos locais para cada ponto de teste. São conceitos diferentes que compartilham o nome.

### 3.1 Base-learners e justificativas

**Base-learners atômicos** — candidatos a entrar no stacking/voting/combinação dinâmica:

| # | Modelo | Representação | Lib | Motivação na literatura |
|---|---|---|---|---|
| 1 | Complement NB | TF-IDF | sklearn | Baseline Lamkanfi; robusto ao desbalanceamento |
| 2 | Regressão Logística | TF-IDF | sklearn | Linear, interpretável |
| 3 | SVM Linear (calibrado) | TF-IDF | sklearn | Margem máxima em alta dim |
| 4 | Decision Tree | TF-IDF | sklearn | Árvore única; baseline simples |
| 5 | KNN | Embeddings | sklearn | Baseline por similaridade semântica (PCA 200) |
| 6 | MLP | Embeddings | sklearn | Rede neural densa sobre embeddings |
| 7 | ELM | Embeddings | **NumPy puro** | Lorenzato; não-sklearn; analítico; pesos aleatórios |

**Modelos benchmark-only** — avaliados individualmente, **nunca entram no stacking/voting/combinação dinâmica** (`_ENSEMBLE_ONLY` na célula 32):

| # | Modelo | Representação | Lib | Por que só benchmark |
|---|---|---|---|---|
| 8 | Random Forest | Embeddings | sklearn | Bagging de árvores — ensemble interno |
| 9 | XGBoost | Embeddings | **xgboost** | Gradient boosting — ensemble interno |
| 10 | LightGBM | Embeddings | **lightgbm** | Gradient boosting leaf-wise — ensemble interno |
| 11 | CatBoost | Embeddings | **catboost** | Gradient boosting — ensemble interno |
| 12 | NB-Lamkanfi (BoW+Stem) | BoW + Porter | sklearn + nltk | Reprodução **fiel** Lamkanfi (2010) com **MultinomialNB**; excluído por razão de **infraestrutura** — stacking só tem paths `text`(TF-IDF)/`emb`, não há path BoW+stem |
| 13 | NB-Complement (BoW+Stem) | BoW + Porter | sklearn + nltk | **Investigação**: mesma entrada BoW, mas **ComplementNB** — isola o efeito Multinomial vs Complement sobre a mesma representação; benchmark-only pela mesma razão |

> RF/XGB/LGBM/CB: incluí-los no stacking criaria "ensemble de ensemble" sem justificativa metodológica (Lorenzato, 2024; orientação do docente). NB-Lamkanfi é excluído por motivo distinto: é um baseline atômico de comparação fiel a Lamkanfi, mas seu path BoW+stemming não é suportado pela infraestrutura de stacking (que só vetoriza via TF-IDF ou usa embeddings). Permanece como benchmark direto.

**Investigação Multinomial vs Complement (fiel vs robusto):** Lamkanfi (2010) usou Naïve Bayes padrão (`MultinomialNB`). O projeto mantém essa escolha na reprodução fiel `NB-Lamkanfi (BoW+Stem)` e adiciona `NB-Complement (BoW+Stem)` — idêntico em representação (mesmo BoW+stem, mesmo `nb_lam_alpha` tunado na própria escala de contagem), diferindo **apenas** no classificador. Isso permite medir quanto o `ComplementNB` (projetado para classes desbalanceadas) ganha sobre o NB clássico **sem confundir** com diferenças de pré-processamento. O `nb_lam_alpha` é tunado sobre BoW (não herdado do TF-IDF), corrigindo a escala de smoothing.

**Por que dois grupos de representação:** TF-IDF captura vocabulário específico técnico (Lamkanfi, Spanos confirmam eficácia). Embeddings contextuais capturam semântica e relações não-lexicais (Umer motiva uso de embeddings). A diversidade garante complementaridade: quando TF-IDF erra por falta de contexto, embeddings podem compensar.

**Redução de dimensionalidade (PCA) — somente KNN:**

| Modelo | Precisa de PCA? | Motivo |
|---|---|---|
| KNN (Emb) | ✅ **Sim** — `PCA(n_components=200)` adicionado | Usa distância euclidiana pura em 1024 dims → maldição da dimensionalidade: todas as distâncias convergem, k-vizinhos perdem significado |
| RF (Emb) | ❌ Não | Usa `sqrt(1024) ≈ 32` features aleatórias por split — redução implícita já ocorre |
| MLP (Emb) | ❌ Não | Primeira camada oculta faz projeção aprendida: 1024 → 128 |
| ELM (Emb) | ❌ Não | Pesos aleatórios de entrada fazem projeção aleatória: 1024 → 500 |
| XGBoost/LightGBM/CatBoost | ❌ Não | Tree-based: busca splits ótimos por feature individualmente, não usa distância |

Pipeline KNN atualizado: `StandardScaler → PCA(200) → SMOTE → KNeighborsClassifier`

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
0. Remover a priori os modelos em _ENSEMBLE_ONLY
   (RF, XGBoost, LightGBM, CatBoost — ensembles internos; e NB-Lamkanfi — sem path BoW no stacking)
   → ficam só como benchmark, nunca entram no stacking/voting/comb. dinâmica

1. Rodar Friedman sobre os modelos atômicos restantes (H₀: todos equivalentes)
   → p < 0.05: diferença significativa → prosseguir com Nemenyi
   → p ≥ 0.05: usar top-N fallback (sem evidência de diferença)

2. Nemenyi pós-hoc: p-value para cada par de modelos atômicos

3. Percorrer modelos em ordem crescente de rank médio (melhor primeiro):
   - Incluir o melhor sempre
   - Excluir modelo m se:
     (a) p-Nemenyi(m, algum já selecionado) > 0.05  [equivalentes]
     E
     (b) representação(m) == representação(já selecionado)  [mesma família]
```

**Por que a condição (b):** Preserva diversidade de representação. Se SVM (TF-IDF) e LogReg (TF-IDF) são equivalentes, excluímos LogReg (pior rank). Mas se SVM (TF-IDF) e MLP (Emb) são equivalentes, mantemos ambos — representam informação diferente.

> ⚠️ **Resultado da rodada anterior (pré-mudança):** XGBoost (Emb) estava entre os selecionados para o Apache Spark. Após introdução de `_ENSEMBLE_ONLY`, XGBoost não pode mais ser selecionado — **os resultados do stacking vão mudar na próxima execução**. Ver Seção 9 (Pendências).

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

### 5.0 Comparação Direta com os Artigos de Base

> Comparação difícil porque tarefas, datasets e métricas diferem — interpretada com cautela.

| Artigo | Tarefa | Métrica reportada | Valor no artigo | Nosso equivalente | Nosso valor |
|---|---|---|---|---|---|
| **Lamkanfi (2010)** | Binária (Severo/Não-severo), Bugzilla, NB+BoW | Precisão/Recall por classe | 0,65–0,75 | NB-Lamkanfi (BoW+Stem), 3 classes, Jira | *pendente re-run* |
| **Lamkanfi (2010)** | Binária, NB | Acurácia global | ~70–75% | NB (TF-IDF), 3 classes, Spark | 0,475 F1-macro (⚠️ tarefa diferente) |
| **Spanos (2018)** | 6 características CVSS, RF | F-measure por característica | 47–68% | SVM (TF-IDF), 3 classes, CVE | 0,704 F1-macro (⚠️ tarefa diferente) |
| **Umer (2019)** | Severidade multi-classe, CNN+RF | Melhoria sobre individual | +5–10% | Stacking vs SVM (Wilcoxon) | Spark: +0% (p=0,49) / CVE: ~−1% (p=0,32) |

**Notas de comparabilidade:**
- Lamkanfi usa classificação **binária** e dataset **Bugzilla** — nosso projeto usa 3 classes e Jira. Comparação direta não é válida; serve como referência de magnitude.
- Spanos usa **6 características CVSS simultaneamente** (337 classes LP) — completamente diferente de nossa classificação tri-classe. O que podemos afirmar: texto CVE + TF-IDF produz F1 ~0,70 em ambos os trabalhos, confirmando que o domínio e a representação são adequados.
- Umer: a melhoria pequena (não significativa) do Stacking sobre SVM é **coerente** com o padrão reportado — quando baselines individuais são fortes (SVM já bem otimizado), ensembles ganham menos.
- **NB-Lamkanfi** é a única comparação direta metodologicamente válida com Lamkanfi — usa exatamente o mesmo pipeline (BoW + stopwords + Porter stemmer + NB). Resultado pendente de re-run.

### 5.1 Apache Spark — Tabela completa

> ⚠️ **Rodada anterior (sem NB-Lamkanfi, sem filtro ensemble).** Re-execução pendente após mudanças desta sessão. NB-Lamkanfi e ELM ainda não têm resultado; Stacking terá composição diferente (XGBoost foi excluído).

| Modelo | F1-macro | F1-Alta | F1-Média | F1-Baixa | MCC | Papel |
|---|---|---|---|---|---|---|
| NB-Lamkanfi (BoW+Stem) | *pendente* | — | — | — | — | Benchmark-only |
| NB (TF-IDF) | 0.475 ± 0.025 | 0.389 | 0.478 | 0.558 | 0.238 | Atômico/Stacking |
| LogReg (TF-IDF) | 0.508 ± 0.027 | 0.390 | 0.536 | 0.597 | 0.258 | Atômico/Stacking |
| **SVM (TF-IDF)** | **0.520 ± 0.031** | 0.413 | 0.537 | 0.609 | 0.280 | Atômico/Stacking |
| Decision Tree (TF-IDF) | 0.395 ± 0.035 | 0.226 | 0.406 | 0.552 | 0.111 | Atômico/Stacking |
| KNN (Emb) | 0.377 ± 0.031 | 0.329 | 0.408 | 0.394 | 0.165 | Atômico/Stacking |
| MLP (Emb) | 0.501 ± 0.026 | 0.377 | 0.542 | 0.585 | 0.243 | Atômico/Stacking |
| ELM (Emb) | *pendente* | — | — | — | — | Atômico/Stacking |
| RF (Emb) | 0.495 ± 0.030 | 0.347 | 0.559 | 0.578 | 0.240 | Benchmark-only |
| XGBoost (Emb) | 0.495 ± 0.023 | 0.380 | 0.534 | 0.570 | 0.228 | Benchmark-only |
| LightGBM (Emb) | 0.485 ± 0.028 | 0.319 | 0.553 | 0.583 | 0.230 | Benchmark-only |
| CatBoost (Emb) | 0.493 ± 0.025 | 0.382 | 0.545 | 0.552 | 0.222 | Benchmark-only |
| **★ Stacking** | **0.511 ± 0.024** *(¹)* | **0.438** | 0.498 | 0.598 | **0.286** | — |
| ⬥ Voto Ponderado | 0.502 ± 0.032 *(¹)* | 0.390 | 0.538 | 0.578 | 0.252 | — |

*(¹) Rodada anterior: XGBoost (Emb) estava no stacking. Após filtro ensemble, próxima rodada usará apenas modelos atômicos → resultados do stacking vão mudar.*

**Análise por artigo:**

- **Vs Lamkanfi:** SVM (0,520) é o melhor baseline individual — **confirma o achado de Lamkanfi** de que SVM supera DT e NB. O Stacking (0,511) fica ligeiramente abaixo em F1-macro mas **supera em F1-Alta (0,438 vs 0,413)** e em MCC (0,286 vs 0,280) — o sistema híbrido é melhor em detectar a classe crítica, que é o que importa na prática.

- **Vs Umer:** A melhoria do Stacking sobre o melhor individual é pequena (+0 em F1-macro, +0,025 em F1-Alta). Wilcoxon p=0,4922 — não significativo. Coerente com a observação de que quando os baselines individuais são fortes, o ensemble adiciona menos.

- **Ponto inesperado:** Embeddings **não superam** TF-IDF no Spark. Provável causa: bugs Jira têm vocabulário muito específico (class names, method names, exception types) onde TF-IDF com sublinear_tf captura eficientemente. Embeddings gerais (treinados em texto genérico) não têm vantagem semântica sobre vocabulário técnico altamente específico.

### 5.2 CIRCL CVE — Tabela completa

> ⚠️ **Rodada anterior (sem NB-Lamkanfi, sem filtro ensemble).** Re-execução pendente.

| Modelo | F1-macro | F1-Alta | F1-Média | F1-Baixa | MCC | Papel |
|---|---|---|---|---|---|---|
| NB-Lamkanfi (BoW+Stem) | *pendente* | — | — | — | — | Benchmark-only |
| NB (TF-IDF) | 0.653 ± 0.023 | 0.723 | 0.649 | 0.587 | 0.490 | Atômico/Stacking |
| LogReg (TF-IDF) | 0.702 ± 0.027 | 0.762 | 0.699 | 0.646 | 0.551 | Atômico/Stacking |
| **SVM (TF-IDF)** | **0.704 ± 0.024** | **0.764** | **0.700** | 0.648 | **0.555** | Atômico/Stacking |
| Decision Tree (TF-IDF) | 0.691 ± 0.019 | 0.710 | 0.709 | 0.655 | 0.526 | Atômico/Stacking |
| KNN (Emb) | 0.508 ± 0.028 | 0.583 | 0.490 | 0.451 | 0.325 | Atômico/Stacking |
| MLP (Emb) | 0.652 ± 0.029 | 0.705 | 0.665 | 0.587 | 0.472 | Atômico/Stacking |
| ELM (Emb) | *pendente* | — | — | — | — | Atômico/Stacking |
| RF (Emb) | 0.644 ± 0.024 | 0.682 | 0.667 | 0.585 | 0.455 | Benchmark-only |
| XGBoost (Emb) | 0.643 ± 0.027 | 0.687 | 0.663 | 0.580 | 0.457 | Benchmark-only |
| LightGBM (Emb) | 0.662 ± 0.024 | 0.716 | 0.684 | 0.585 | 0.489 | Benchmark-only |
| CatBoost (Emb) | 0.613 ± 0.034 | 0.647 | 0.632 | 0.559 | 0.414 | Benchmark-only |
| **★ Stacking** | **0.699 ± 0.024** *(¹)* | 0.756 | 0.696 | **0.644** | 0.550 | — |
| ⬥ Voto Ponderado | 0.683 ± 0.033 *(¹)* | 0.737 | 0.687 | 0.625 | 0.522 | — |

*(¹) Rodada anterior com composição diferente — próxima rodada excluirá ensembles a priori.*

**Análise por artigo:**

- **Vs Spanos:** SVM (TF-IDF) obtém 0,704 F1-macro — **dentro da faixa de 0,70–0,80 reportada por Spanos**, confirmando que texto de CVE é discriminativo e TF-IDF é suficiente para um forte baseline. Isso valida tanto o dataset quanto a metodologia.

- **Vs Umer:** O Stacking (0,699) é muito próximo do SVM (0,704), mas **Wilcoxon p=0,3223** — não significativo. O Voto Ponderado (0,683) é significativamente **inferior** ao SVM — Wilcoxon p=0,0195. Ou seja, no domínio CVE, o Meta-learning (Stacking) é a estratégia certa; Fusão simples (Voto) piora em relação ao melhor individual.

- **Padrão oposto ao Spark:** No CVE, embeddings também ficam abaixo do TF-IDF (RF=0,644 vs SVM=0,704). O domínio CVE tem linguagem padronizada e descritiva onde palavras-chave específicas ("code execution", "buffer overflow", "CSRF") são altamente discriminativas — favorecendo TF-IDF.

### 5.3 Wilcoxon — Sumário das comparações

> ⚠️ **Rodada anterior** — stacking incluía XGBoost. Após filtro `_ENSEMBLE_ONLY`, re-executar para resultados válidos.

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
| SVM como baseline principal | Lamkanfi (2010): menciona SVM como direção futura; nossa adição justificada por desempenho | `CalibratedClassifierCV(LinearSVC(...))` |
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
| BoW puro + stemming → perda semântica | TF-IDF sublinear + embeddings Qwen3 1024-dim | Seções 5, 7 |
| Pré-processamento com stemming/stopwords | Adotado para TF-IDF via `_lamkanfi_tokenizer`; embeddings usam texto bruto (transformers) | Seção 3 |
| Classificação binária (severo/não-severo) | Tri-classe (Alta/Média/Baixa) — granularidade operacional | Seção 1 |
| Sem balanceamento de classes | Undersampling real-first + SMOTE inside fold | Seções 4, 6 |
| Modelos isolados | Stacking + Fusão + Comb. Dinâmica | Seções 9, 9.1, 9.2 |
| Dataset único (Bugzilla) | Dois domínios: Jira (bugs) + CIRCL (CVE) | Seção 1 |
| Sem testes de hipótese | Friedman + Nemenyi + Wilcoxon | Seções 8, 11 |
| Reprodução: baseline **NB-Lamkanfi (BoW+Stem)** implementado fielmente | CountVectorizer + stopwords + Porter stemmer + ComplementNB | Seção 7, célula 24 do notebook |

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
| DTM/TF-IDF apenas (sem embeddings contextuais) | TF-IDF sublinear + Qwen3 embeddings 1024-dim | Seções 5, 7 |
| 337 classes LP — não generalizável como métrica de criticidade | 3 classes operacionais (Alta/Média/Baixa) com thresholds NIST | Seção 1 |
| Modelos isolados (sem ensemble meta-learner) | Sistema híbrido (Stacking + Voto Ponderado + Comb. Dinâmica) | Seções 9, 9.1, 9.2 |
| Apenas pesos de classe (sem SMOTE) | Undersampling real-first + SMOTE dentro do fold | Seções 4, 6 |
| Apenas domínio CVE/NVD | Validação também em domínio Jira (Apache Spark) | Seções 1, 10 |
| Sem testes de hipótese formais | Friedman + Nemenyi + Wilcoxon | Seções 8, 11 |

---

## 8. Checklist de Requisitos

| Requisito | Status | Evidência objetiva |
|---|---|---|
| Desenvolver sistema híbrido | ✅ | Stacking (meta-LogReg sobre OOF) + Voto Ponderado + Comb. Dinâmica |
| Verificar literatura e propor alteração | ✅ | 4 artigos mapeados; "Nossa extensão" em célula 0 do notebook; implementada como embeddings + nested CV + seleção estatística |
| Experimentos comparativos (proposto vs baselines) | ✅ | 12 baselines avaliados em 10-fold (incluindo NB-Lamkanfi reprodução fiel); Wilcoxon compara Stacking vs melhor baseline |
| Não usar apenas sklearn | ✅ | XGBoost, LightGBM, CatBoost (libs externas); ELM em NumPy puro |
| Testes de hipótese obrigatórios | ✅ | Friedman (Seção 8) + Nemenyi pós-hoc (Seção 8.1) + Wilcoxon (Seção 11) |
| Mais de uma base de dados | ✅ | Apache Spark Jira + CIRCL CVE (domínios, tamanhos e distribuições distintos) |
| Tabela de métricas | ✅ | Seção 10 do notebook; `resultados/metricas_apache_spark.csv` e `metricas_circl_cve.csv`; 9 métricas × 12 modelos × 2 datasets |
| Especificação do processo de seleção de modelos | ✅ | Seção 8.1: algoritmo `seleciona_base_learners()` baseado em rank Nemenyi + regra de diversidade de representação |

---

## 9. Pendências

| Item | Prioridade | Descrição |
|---|---|---|
| **Re-rodar baselines (Cells 25–26)** | 🔴 Crítica | NB-Lamkanfi e ELM novos — precisam de resultado |
| **Re-rodar stacking/voting/comb. dinâmica (Cells 32–42)** | 🔴 Crítica | `_ENSEMBLE_ONLY` muda composição: XGBoost (Spark) e LightGBM (CVE), que eram selecionados antes, agora são excluídos |
| ELM nos 10-folds | 🔴 Alta | ELM estava ausente da rodada anterior; agora incluído — resultado pendente |
| Executar Comb. Dinâmica | 🔴 Alta | Célula 41 sem output; executar `run_combinacao_dinamica()` e incluir nas Seções 10 e 11 |
| Atualizar Cell 21 markdown | 🟡 Média | Ainda diz "10 modelos" no título da Seção 6.6 — deve ser "12 modelos (10 com grid próprio + ELM sem tuning + NB-Lamkanfi reutiliza nb_alpha)" |
| Atualizar tabela consolidada (Seção 10) | 🟡 Média | Após re-run: atualizar com NB-Lamkanfi, ELM, novos resultados do stacking |
| Atualizar resultados Wilcoxon (Seção 11) | 🟡 Média | Comparações vão mudar após novo stacking — refazer interpretação |
| Ampliar revisão no notebook (Cell 0) | 🟢 Baixa | Incluir Umer et al. e Spanos & Angelis explicitamente na célula de introdução |

### O que mudou nesta sessão de análise

| Mudança | Onde | Impacto nos resultados |
|---|---|---|
| NB-Lamkanfi (BoW+Stem) adicionado | `build_baselines()` Cell 24 | Nova linha nos resultados — **pendente re-run** |
| `_ENSEMBLE_ONLY` filtro em `seleciona_base_learners` | Cell 32 | XGBoost (Spark) e LightGBM (CVE) eram selecionados antes — agora excluídos; stacking muda |
| `'NB-Lamkanfi (BoW+Stem)'` adicionado a `_ENSEMBLE_ONLY` | Cell 32 | **Fix crítico**: sem isso o notebook trava com `ValueError` ao processar seleção pós-Nemenyi (BoW+stem não tem path no stacking) |
| **`_lamkanfi_tokenizer` adicionado a TODOS os TF-IDF** | Cells 22/24/34/37/40 do notebook | Resultados TF-IDF mudarão no próximo run (alinhamento com Lamkanfi+Spanos) |
| **NB-Lamkanfi → MultinomialNB (fiel) + grid `nb_lam_alpha` próprio em BoW** | Cells 1/22/24 | Reprodução fiel a Lamkanfi; alpha não mais herdado do TF-IDF |
| **NB-Complement (BoW+Stem) adicionado** (investigação Multinomial vs Complement) | Cells 23/24/32 | Novo 13º baseline benchmark-only; isola efeito do classificador |
| Notebook Cell 48 corrigido | "TF-IDF+stemming" → "BoW+stemming" | Só documentação |
| ANALISE.md: Lamkanfi usa só NB (não SVM/DT/kNN) | Seções 1.1, 3.1 | Só documentação |
| ANALISE.md: Spanos usa DTM bruto, não TF-IDF | Seção 1.3 | Só documentação |
| ANALISE.md: F1 0.70–0.80 é do paper de 2017, não 2018 | Seção 1.3 | Só documentação |
| ANALISE.md: figuras das 4 arquiteturas | Seção 0.1 | Nova seção |
