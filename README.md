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
9. [Resultados e Discussão (Segunda Rodada)](#resultados-e-discussão-segunda-rodada)
10. [Estrutura do Projeto](#estrutura-do-projeto)
11. [Seções do Notebook](#seções-do-notebook)
12. [Setup e Execução](#setup-e-execução)
13. [Requisitos do Trabalho — Checklist](#requisitos-do-trabalho--checklist)
14. [O que ainda falta implementar](#o-que-ainda-falta-implementar)
15. [Tempos Estimados](#tempos-estimados)
16. [Referências](#referências)
17. [Resultados Estatísticos — Wilcoxon](#resultados-estatísticos--wilcoxon)

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

O sistema implementa **três** estratégias de ensemble definidas em
Lorenzato (RecPad Aula 09): **Meta-learning** (Stacking, formalizado por Wolpert 1992),
**Fusão** (Voto Suave Ponderado) e **Combinação Dinâmica** (KNN + mínimos quadrados local).
Todas operam sobre **o mesmo subconjunto de base-learners**, selecionado dinamicamente pelo
teste de Nemenyi (Seção 8.1), permitindo comparação direta entre as três estratégias.

```
════════════ SELEÇÃO DINÂMICA (Seção 8.1, pós-Nemenyi) ════════════
  10 baselines avaliados → Friedman + Nemenyi → subconjunto de base-learners
  (mantém diversidade de representação, descarta redundância da mesma família)
                                │
                                ▼  (mesma lista alimenta os dois ensembles)
════════════════ NÍVEL 0 — Classificadores Base selecionados ════════════════

  Entrada: texto bruto do bug/CVE
      │
      ├─► [TF-IDF sublinear, 20k features] ─► base-learners 'text'  ─► P(Alta) P(Méd) P(Bx) ──┐
      │                                                                                          │
      └─► [Qwen3-0.6B Embedding, 1024 dim] ─► base-learners 'emb'   ─► P(Alta) P(Méd) P(Bx) ──┤
                                                                                                 │
       (k base-learners selecionados → 3·k meta-features / probabilidades)                       │
                                                                                                 │
══════════ NÍVEL 1 — Combinação (duas estratégias comparadas) ════════════════                   │
                                                                                                 │
  Meta-features: 3·k colunas ◄──────────────────────────────────────────────────────────────────┘
     │
     ├─► [Meta-learning]    LogReg meta (C=0.1) treinado nas predições OOF        ─► classe final
     │
     ├─► [Fusão]            P_final = Σ wᵢ·Pᵢ , wᵢ = F1ᵢ / Σ F1ⱼ (argmax)      ─► classe final
     │
     └─► [Comb. Dinâmica]  KNN (k=7) no espaço OOF → W = pinv(X_viz) @ y_viz    ─► classe final
                            (pesos locais por mínimos quadrados, Lorenzato 2024)
```

**Onde acontece a hibridização:**  
As `3·k` meta-features combinam saídas de **duas representações fundamentalmente distintas**
(quando a seleção inclui modelos das duas famílias):
- blocos de modelos TF-IDF: frequência de termos (esparso, ~20k dimensões, linear)
- blocos de modelos Embeddings: semântica densa (1024 dimensões, não-linear)

O meta-classificador aprende automaticamente o peso ideal de cada fonte por classe,
equivalente ao "Módulo de Combinação" no diagrama de Sistemas Inteligentes Híbridos
de Lorenzato (2024, slide 29).

**Anti-leakage — Out-of-Fold (OOF):**  
As meta-features de treino são geradas com inner CV (5 folds): cada amostra é predita
por um modelo que nunca a viu no treino, evitando que o meta-classificador memorize
os dados de base.

---

## Baselines Comparados

| #  | Modelo | Representação | Biblioteca | Avaliado em 10-fold |
|----|---|---|---|---|
| 1  | Complement Naive Bayes | TF-IDF | `sklearn` | ✅ |
| 2  | Regressão Logística | TF-IDF | `sklearn` | ✅ |
| 3  | SVM Linear (calibrado) | TF-IDF | `sklearn` | ✅ |
| 4  | Decision Tree | TF-IDF | `sklearn` | ✅ |
| 5  | Random Forest | Embeddings | `sklearn` | ✅ |
| 6  | k-NN | Embeddings | `sklearn` | ✅ |
| 7  | **XGBoost** | Embeddings | **`xgboost`** | ✅ |
| 8  | **LightGBM** | Embeddings | **`lightgbm`** | ✅ |
| 9  | **CatBoost** | Embeddings | **`catboost`** | ✅ |
| 10 | MLP (rede neural rasa) | Embeddings | `sklearn` | ✅ |
| 11 | **ELM** (Extreme Learning Machine) | Embeddings | `numpy` | ⚠️ GridSearch OK; 10-fold pendente |

> Cumpre o requisito: **não é permitido usar apenas sklearn** — XGBoost, LightGBM e
> CatBoost são bibliotecas externas independentes; ELM é implementado diretamente em
> NumPy (sem sklearn); SentenceTransformer (embeddings) também.

**Nota sobre o ELM:** a classe `ELMClassifier` está implementada em NumPy puro e seu
`GridSearchCV` de hiperparâmetros foi executado (Seção 6.5). Porém a avaliação 10-fold
completa (Seção 7) ainda usa apenas os 10 modelos acima — o ELM precisa ser adicionado
ao laço principal de baselines e o notebook re-executado.

**Justificativa das adições:**
- **Decision Tree** e **k-NN** completam os quatro classificadores originais de
  Lamkanfi et al. (2010) (que usava NB, SVM, Decision Tree e k-NN), tornando a
  replicação do baseline clássico fiel ao trabalho de referência.
- **CatBoost** (Prokhorenkova et al., 2018) entra como terceiro algoritmo de
  *gradient boosting* externo, ao lado de XGBoost e LightGBM, ampliando a diversidade
  de boosting sobre a representação semântica.
- **MLP sobre embeddings** é o análogo de classificador único neural ao híbrido
  CNN+RF de Umer et al. (2019): como substituímos o CNN por embeddings congelados do
  Qwen3, uma rede neural rasa (MLP) sobre esses embeddings é a contraparte natural de
  "classificador neural isolado" para fins de comparação.
- **ELM (Extreme Learning Machine)** — pesos de entrada aleatórios e fixos; pesos de
  saída calculados por pseudo-inversa (mínimos quadrados). Implementado conforme
  exercício de aula (Lorenzato, 2024). Pendente de integração ao loop de avaliação.

> Com os 10 baselines avaliados, a composição dos ensembles **não é fixada a priori**: é
> **selecionada dinamicamente** a partir do teste de Nemenyi (Seção 8.1 do notebook).
> Ver [Processo de Seleção de Modelos](#processo-de-seleção-de-modelos).

---

## Processo de Seleção de Modelos

A especificação do processo de seleção segue três etapas encadeadas:

### Etapa 1 — Justificativa a priori (antes do experimento)
Os 10 baselines foram escolhidos para cobrir o espaço de modelos relevantes na literatura:
- NB, LogReg, SVM, Decision Tree, k-NN: replicam os baselines de Lamkanfi (2010)
- RF, XGBoost, LightGBM, CatBoost: extensão moderna (ensembles/boosting) sobre representação semântica
- MLP: contraparte de classificador neural isolado sobre embeddings (cf. Umer et al., 2019)

### Etapa 2 — Ajuste de hiperparâmetros (Seção 6.5 do notebook)
`GridSearchCV` (5 folds internos) busca os melhores hiperparâmetros de todos os
baselines ajustáveis (LogReg, SVM, NB, Decision Tree, RF, k-NN, XGBoost, LightGBM,
CatBoost, MLP, ELM) antes da avaliação final, com SMOTE dentro de cada fold do search.

### Etapa 3 — Comparação estatística pós-experimento (Seção 8 do notebook)
**Teste de Friedman** (não-paramétrico, multi-modelo, seguindo Demšar 2006):
- H₀: todos os modelos têm desempenho equivalente
- Se p < 0.05: **Nemenyi pós-hoc** compara todos os pares

### Etapa 4 — Seleção dinâmica de base-learners (Seção 8.1 do notebook)
Uma célula de decisão **deriva automaticamente** a composição dos ensembles a partir
do Nemenyi (não fixa modelos a priori):
- percorre os modelos por rank médio crescente (melhor primeiro);
- **exclui** um modelo se ele for estatisticamente equivalente (Nemenyi p > 0,05) a um
  modelo já selecionado, de melhor rank e **mesma representação** (TF-IDF/Embeddings) —
  descartando redundância da mesma família e preservando diversidade de representação;
- se o Friedman não for significativo, recorre ao *fallback* top-N por rank médio.

A lista selecionada (que pode diferir entre Spark e CIRCL) alimenta **tanto o Stacking
quanto o Voto Ponderado**, garantindo comparação justa sobre base idêntica. Isso
documenta explicitamente a *especificação do processo de seleção de modelos*.

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

## Resultados e Discussão (Segunda Rodada)

Resultados completos em `resultados/` (tabelas `metricas_*.csv` e gráficos de
confusão, Nemenyi e Wilcoxon). Os CSVs contêm 10 baselines + 2 ensembles (12 linhas cada).

### Visão comparativa — Spark vs CIRCL

| Métrica (Stacking) | Apache Spark | CIRCL CVE |
|---|---|---|
| F1-macro | 0,511 ± 0,024 | 0,699 ± 0,024 |
| MCC | 0,286 ± 0,030 | 0,550 ± 0,031 |
| Kappa | 0,277 ± 0,031 | 0,548 ± 0,032 |
| F1-Alta | 0,438 ± 0,032 | 0,756 ± 0,022 |
| Melhor baseline isolado | SVM (TF-IDF), F1-macro 0,520 | SVM (TF-IDF), F1-macro 0,704 |

**CIRCL CVE supera Apache Spark de forma consistente** em todas as métricas — diferença de
~0,18 em F1-macro e quase o dobro em MCC/Kappa. A causa mais provável não são os embeddings
(o PCA mostra sobreposição forte de classes nos dois datasets), e sim a **qualidade do rótulo**:
o CVSS é calculado a partir de atributos técnicos objetivos (vetor de ataque, impacto), enquanto a
prioridade Jira é atribuída subjetivamente pelo reportador, e o mapeamento `Major → Média` comprime
~70% dos dados originais numa única classe "guarda-chuva" que se sobrepõe textualmente com as
demais.

### Compatibilidade com a literatura

- **Spark/Jira — abaixo do esperado.** Lamkanfi et al. (2010) reportam precisão/recall de
  **0,65–0,75** no Eclipse/Mozilla/GNOME com ≥ 500 reports/classe. Nosso Spark (~1.333/classe)
  entrega apenas ~0,51–0,53 — explicável pela natureza mais subjetiva e ambígua da prioridade
  Jira frente ao campo de severidade estruturado usado por Lamkanfi.
- **CIRCL/CVSS — consistente, sem benchmark quantitativo direto.** Spanos & Angelis (2018) são
  citados no projeto apenas qualitativamente ("o texto da CVE carrega informação preditiva
  suficiente"); não reportam uma métrica diretamente comparável. Nosso F1-macro ~0,70 confirma a
  tese deles na prática, e cai dentro da própria faixa 0,65–0,75 de Lamkanfi — ou seja, bate com o
  que a literatura clássica (pré-transformers) considera um resultado sólido nesse tipo de tarefa.
- **Desvio interessante de Umer et al. (2019):** Umer afirma que a classe Alta/crítica é
  consistentemente a mais difícil. Isso se confirma no Spark (F1-Alta é a menor das três classes),
  mas **se inverte no CIRCL** (F1-Alta é a maior). Hipótese: no CIRCL a classe Alta não é rara na
  distribuição original (~40–45% antes do undersampling) e tem vocabulário técnico muito
  distintivo ("remote code execution", "buffer overflow" etc.), enquanto no Spark ela é rara e
  textualmente ambígua.

### O ganho do Stacking é estatisticamente significativo?

Não, em F1-macro. O Wilcoxon Stacking vs SVM (TF-IDF) **não é significativo** em nenhum dos dois
datasets (Spark p=0,4922; CIRCL p=0,3223), e em F1-macro puro o Stacking até perde levemente para
o SVM nos dois casos. O ganho real do Stacking aparece em **MCC, Kappa e F1-Alta**, não em
F1-macro/acurácia. No Spark, a classe Alta sobe de recall 46,8% (SVM) para 65,1% (Stacking), mas
às custas da classe Média (recall 60,7% → 43,2%) — é uma troca de prioridade, não um ganho líquido.

Achado mais preocupante: no CIRCL, o **Voto Ponderado perde de forma estatisticamente
significativa para o SVM puro** (Wilcoxon p=0,0195, F1-macro 0,683 vs 0,704) — o método de fusão
simples fica atrás do melhor baseline isolado nesse dataset.

### O valor de 0,70 (F1-macro, CIRCL) é bom?

Sim, com ressalvas de escopo:
- É mais que o dobro do acaso (0,33 para 3 classes balanceadas).
- O Kappa de 0,548 cai na faixa "moderado a substancial" (Landis & Koch) — acordo real e útil,
  mas não "quase perfeito".
- Bate a faixa histórica de Lamkanfi (0,65–0,75) para precisão/recall.
- Provavelmente fica **abaixo** do que um fine-tuning de transformer (Ali et al., 2024 — citado
  como estado-da-arte atual) alcançaria no mesmo dataset — esperado, já que o projeto usa
  embeddings congelados sem fine-tuning, posicionado deliberadamente entre Lamkanfi (clássico) e
  Ali (fine-tuning).
- Útil como **ferramenta de priorização assistida** (ordenar fila para revisão humana); ainda
  não suficiente para **triagem totalmente autônoma** — ~30% de erro por classe é alto demais
  quando o custo de classificar uma vulnerabilidade crítica como "Média" é grande.

### Limitações metodológicas identificadas

- **Reuso de folds na seleção de base-learners:** a seleção via Nemenyi (Seção 8.1) usa os
  mesmos 10 folds externos que depois servem para comparar Stacking/Voto contra os baselines —
  dupla utilização dos dados de teste, infla a confiança no resultado final. O ideal seria uma
  validação aninhada (seleção em CV interna, separada da avaliação final).
- **KNN (Emb)** é consistentemente o pior modelo nos dois datasets — provável maldição da
  dimensionalidade em embeddings de 1024-d sem redução dimensional.
- **Decision Tree (TF-IDF)** no Spark tem MCC ≈ 0,111 (quase aleatório) — discutível mantê-lo
  como baseline competitivo.

### Sugestões para a próxima rodada

1. Tratar severidade como problema **ordinal** (Baixa < Média < Alta) ou aplicar threshold
   tuning por classe em vez de argmax simples — recupera parte do recall de Média no Spark sem
   custo de implementação.
2. Aplicar custo assimétrico/threshold mais agressivo para a classe Alta (falso negativo em
   bug/CVE crítico custa mais que falso positivo).
3. Separar a seleção de base-learners (Nemenyi) da avaliação final via CV aninhada, para que
   ganhos futuros do ensemble possam ser comparados com confiança estatística real.
4. Remover ou substituir KNN (Emb) por uma variante com redução de dimensionalidade (PCA/UMAP)
   antes do KNN.
5. Calibrar probabilidades antes do Voto Ponderado (hoje os pesos são apenas `F1ᵢ / Σ F1ⱼ`, sem
   calibração) — pode reduzir a perda significativa frente ao SVM no CIRCL.
6. Para CIRCL: testar embeddings de domínio de segurança (ex. SecBERT/CySecBERT) ou metadados
   da CVE (categoria CWE, vetor de ataque) como features complementares ao texto.
7. Para Spark: considerar fine-tuning leve do modelo de embeddings em corpus de bug reports, já
   que o Qwen3-0.6B genérico pode não capturar bem o jargão de Jira.

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
│   └── projeto_hibrido.ipynb           # Notebook principal (Seções 0–12, c/ 8.0/8.1/9.1)
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

| Seção | Conteúdo | Estado |
|---|---|---|
| 0. Título | Contexto, tabela comparativa vs Lamkanfi, estrutura | ✅ |
| 1. Dados | Carregamento Spark e CIRCL, mapeamento de rótulos, estatísticas | ✅ |
| 2. EDA | Distribuição de classes, comprimento dos textos por classe | ✅ |
| 3. Pré-processamento | Truncamento, tokens por classe, bigramas, top TF-IDF features | ✅ |
| 4. Balanceamento | Undersampling real-first + visualização antes/depois | ✅ |
| 5. Embeddings | Qwen3-0.6B com cache em disco + PCA 2D para inspeção | ✅ |
| 6. Framework | Validação cruzada 10-fold, função `compute_metrics`, todas as métricas | ✅ |
| 6.5. Hiperparâmetros | GridSearchCV (5 folds) para os 10 baselines + ELM | ✅ |
| 7. Baselines | **10** modelos avaliados em 10-fold com SMOTE (ELM no loop pendente) | ⚠️ |
| 8. Seleção | Friedman + Nemenyi pós-hoc + tabela de ranks | ✅ |
| 8.0. Registry | Factory compartilhado de base-learners (Stacking + Voto + Comb. Din.) | ✅ |
| 8.1. Decisão | Seleção dinâmica dos base-learners a partir do Nemenyi | ✅ |
| 9. Stacking | Meta-learning híbrido com OOF aninhado (nested CV 10×5) | ✅ |
| 9.1. Voto Ponderado | Fusão por voto suave ponderado (mesmos base-learners) | ✅ |
| 9.2. Comb. Dinâmica | KNN + mínimos quadrados local — código implementado, execução pendente | ⚠️ |
| 10. Resultados | Tabela consolidada (Stacking + Voto) + matrizes de confusão + barras | ✅ |
| 11. Wilcoxon | Stacking/Voto vs SVM + cruzamentos; Comb. Dinâmica pendente | ⚠️ |
| 12. Conclusão | Decisões justificadas, resultados estatísticos, alinhamento com literatura | ✅ |

---

## Dependências Principais

| Pacote | Versão mínima | Papel no projeto |
|---|---|---|
| `pandas` / `numpy` / `scipy` | 2.0 / 1.24 / 1.11 | Manipulação de dados, estatísticas (Friedman, Wilcoxon) |
| `scikit-learn` | 1.3 | LogReg, SVM, NB, RF, TF-IDF, métricas, validação cruzada |
| `imbalanced-learn` | 0.12 | SMOTE e `ImbPipeline` (SMOTE dentro do fold) |
| `xgboost` | 2.0 | Baseline XGBoost — cumpre requisito não-sklearn |
| `lightgbm` | 4.0 | Baseline LightGBM — cumpre requisito não-sklearn |
| `catboost` | 1.2 | Baseline CatBoost — cumpre requisito não-sklearn |
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
| Desenvolver um sistema híbrido | ✅ | Stacking TF-IDF + Embeddings (Seção 9) + Voto Ponderado (Seção 9.1) |
| Verificar literatura e propor alteração | ✅ | README: revisão completa (Lamkanfi, Umer, Spanos, Ali); Notebook cell 0: "O que Lamkanfi fez" + tabela "Nossa extensão" — alteração proposta e implementada (Seções 9/9.1) |
| Experimentos comparativos (proposta vs baselines) | ✅ | 10 baselines + Stacking + Voto Ponderado, Seções 7–10 |
| Não usar apenas métodos sklearn | ✅ | XGBoost, LightGBM, CatBoost (libs externas); SentenceTransformer (Qwen3); ELM em NumPy |
| Testes de hipótese obrigatórios | ✅ | Friedman + Nemenyi (Seção 8) + Wilcoxon (Seção 11) |
| Mais de uma base de dados | ✅ | Apache Spark (bugs) + CIRCL CVE (vulnerabilidades) |
| Tabela de métricas | ✅ | Seção 10: 9 métricas × 12 modelos, salvo em CSV (`resultados/metricas_*.csv`) |
| Especificação do processo de seleção | ✅ | Nemenyi → seleção dinâmica em 4 etapas (Seção 8.1); descrita neste README |

---

## O que ainda falta implementar

### ✅ Concluído

#### ~~1. Célula de decisão pós-Nemenyi (Seção 8)~~ — **FEITO**

Implementado na **Seção 8.1**: `seleciona_base_learners()` deriva dinamicamente o
subconjunto de base-learners a partir do Nemenyi (mantém diversidade de representação,
descarta redundância da mesma família; *fallback* top-N se o Friedman não for
significativo).

#### ~~3. Baseline de Fusão (Voto Suave Ponderado)~~ — **FEITO**

Implementado na **Seção 9.1**: `run_weighted_voting()` com pesos `wᵢ = F1ᵢ / Σ F1ⱼ`.

#### ~~5. ELM (Extreme Learning Machine)~~ — **FEITO**

Implementado na **Seção 7** como 11º baseline: `ELMClassifier` em NumPy puro (pesos de
entrada aleatórios, pesos de saída via pseudo-inversa). Hiperparâmetro `hidden_neurons`
buscado via GridSearchCV em [200, 500, 1000]. Integrado ao registry de base-learners
(Seção 8.0) e elegível para seleção nos ensembles.

#### ~~6. Combinação Dinâmica (KNN + MQ local)~~ — **FEITO**

Implementado na **Seção 9.2**: `run_combinacao_dinamica()` coleta predições OOF dos
base-learners via 5-fold interno, usa KNN (k=7) no espaço de probabilidades OOF para
encontrar vizinhos de cada ponto de teste, e calcula pesos locais por mínimos quadrados
(`W = pinv(X_viz) @ y_viz_onehot`). Comparado via Wilcoxon na Seção 11.

---

### Prioridade Alta (pendente)

#### 2. Seção de revisão de literatura expandida no notebook

**O que falta:** expandir a célula 0 do notebook para incluir:
- Resumo de Umer et al. (2019) e o que o híbrido CNN+RF demonstrou
- Resumo de Spanos & Angelis (2018) e a justificativa do domínio CVE
- Tabela explícita: limitação identificada → alteração proposta neste trabalho

Atualmente cell 0 menciona apenas Lamkanfi. O README tem a seção completa, mas
a banca pode exigir que o notebook também documente a revisão.

#### 3. Completar avaliação do ELM

**O que falta:** adicionar `ELM (Emb)` ao dicionário `baselines` na Seção 7
e re-executar o notebook. O `ELMClassifier` está implementado e seu GridSearchCV
foi rodado (Seção 6.5).

#### 4. Executar Combinação Dinâmica e incluir no Wilcoxon

**O que falta:** executar a célula 41 (`run_combinacao_dinamica`) e re-executar
a Seção 10 (tabela consolidada) e a Seção 11 (Wilcoxon) com os resultados dela.
O código está completo em `run_combinacao_dinamica()`.

---

### Prioridade Baixa

#### 5. Informar o grupo na planilha

Tarefa administrativa — registrar o tema do projeto na planilha compartilhada da turma.

---

## Tempos Estimados

| Etapa | CPU (i7) | GPU (RTX 3050) |
|---|---|---|
| Embeddings Spark (~4k textos) | ~8 min | ~1 min |
| Embeddings CIRCL (~4k textos) | ~8 min | ~1 min |
| Busca de hiperparâmetros (11 baselines, 2 datasets) | ~40 min | ~20 min |
| Baselines (11 modelos, 2 datasets, 10 folds) | ~28 min | ~17 min |
| Stacking (2 datasets, nested 10×5 CV) | ~60 min | ~30 min |
| Voto Ponderado (2 datasets, 10 folds) | ~15 min | ~8 min |
| Comb. Dinâmica (2 datasets, 10 folds × 5 OOF) | ~35 min | ~18 min |
| **Total (primeira execução)** | **~195 min** | **~95 min** |

> Embeddings são cacheados em `data/*.npy` — nas execuções seguintes, apenas os modelos
> são re-treinados. ELM tem treinamento analítico (pseudo-inversa), portanto muito rápido
> individualmente; o custo adicional vem do GridSearchCV. Combinação Dinâmica exige 5-fold
> OOF por fold externo, tornando-a ~2× mais custosa que o Voto Ponderado.

---

## Referências

- **Lamkanfi, A., Demeyer, S., Giger, E., Goethals, B. (2010).** *Predicting the
  Severity of a Reported Bug.* MSR 2010. — referência central; baselines NB, SVM,
  Decision Tree e k-NN sobre bag-of-words.
- **Breiman, L. (2001).** *Random Forests.* Machine Learning, 45(1). — Random Forest.
- **Chen, T., Guestrin, C. (2016).** *XGBoost: A Scalable Tree Boosting System.* KDD 2016.
- **Ke, G. et al. (2017).** *LightGBM: A Highly Efficient Gradient Boosting Decision
  Tree.* NeurIPS 2017.
- **Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., Gulin, A. (2018).**
  *CatBoost: unbiased boosting with categorical features.* NeurIPS 2018.
- **Umer, Q., Liu, H., Sultan, Y. (2019).** *Emotion Based Automated Priority
  Prediction for Bug Reports.* IEEE Access / Sensors. — híbrido CNN+RF; motiva a
  contraparte neural (MLP sobre embeddings).
- **Spanos, G., Angelis, L. (2018).** *A multi-target approach to estimate software
  vulnerability characteristics and severity scores.* Journal of Systems and Software.
- **Wolpert, D. H. (1992).** *Stacked Generalization.* Neural Networks, 5(2). —
  formalização do Stacking (meta-learning).
- **Demšar, J. (2006).** *Statistical Comparisons of Classifiers over Multiple Data
  Sets.* JMLR 7. — Friedman + Nemenyi + Wilcoxon como protocolo de comparação.
- **Lorenzato (2024).** *Sistemas Inteligentes Híbridos / Métodos de Ensemble*,
  RecPad — Aula 09 (UPE). — taxonomia Fusão · Seleção · Meta-learning; exercícios de
  ELM e Combinação Ponderada (base para Seções 7 e 9.2).
- **Huang, G.-B., Zhu, Q.-Y., Siew, C.-K. (2006).** *Extreme Learning Machine: Theory
  and Applications.* Neurocomputing, 70(1–3). — formalização do ELM.

---

## Resultados Estatísticos — Wilcoxon

Resultados da segunda rodada (`resultados/20260622_235327/`). Comparações realizadas
fold-a-fold (Wilcoxon pareado, bicaudal, α = 0.05).

| Comparação | Apache Spark | CIRCL CVE |
|---|---|---|
| Stacking vs SVM (TF-IDF) | p = 0.4922 — não significativo | p = 0.3223 — não significativo |
| Voto Ponderado vs SVM (TF-IDF) | p = 0.1055 — não significativo | **p = 0.0195 — SVM superior** |
| Stacking vs Voto Ponderado | p = 0.4922 — não significativo | p = 0.0840 — não significativo |

> **Pendente:** Combinação Dinâmica (Seção 9.2 — código pronto, execução não realizada)
> e ELM (código pronto, integração ao loop de 10-fold pendente) serão incluídos
> nesta tabela após a próxima rodada de execução completa do notebook.

**Interpretação atual (10 baselines + 2 ensembles):** nenhum ensemble supera o SVM
(TF-IDF) de forma estatisticamente significativa em F1-macro. O único resultado
significativo (CIRCL, Voto Ponderado vs SVM, p = 0.0195) indica que a fusão simples
fica *abaixo* do baseline — resultado negativo documentado, consistente com a literatura
que alerta para a fragilidade de ensembles mal calibrados em domínios com distribuição
de classes desbalanceada.

---

