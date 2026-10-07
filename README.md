
# Estudo TabPFN-3 em Dados Biomédicos e Comportamentais

## Comparação de desempenho do TabPFN-3 com modelos de Machine Learning em bases de dados da área da saúde

Este repositório contém os dados e materiais utilizados no estudo **"Comparação de desempenho do TabPFN-3 com modelos consagrados em bases de dados na área da saúde"**, desenvolvido por João Luiz Junho Pereira, Davi dos Reis Soares e Rayane Braga Machado, da Universidade Federal de Itajubá (UNIFEI).

O trabalho apresenta uma comparação sistemática entre o **TabPFN-3** e sete algoritmos clássicos de classificação em quinze bases de dados biomédicas e comportamentais.

---

## Objetivo

O objetivo do estudo é avaliar o desempenho do TabPFN-3 em comparação com algoritmos tradicionais de Machine Learning em diferentes conjuntos de dados relacionados à saúde.

Foram avaliados oito algoritmos:

- Regressão Logística (LogReg)
- k-Nearest Neighbors (kNN)
- Gaussian Naive Bayes (GaussianNB)
- Support Vector Classifier (SVC)
- Rede Neural Artificial (ANN / MLP)
- Árvore de Decisão (Decision Tree)
- XGBoost
- TabPFN-3

Os modelos clássicos tiveram seus hiperparâmetros otimizados utilizando o **Optuna**, enquanto o TabPFN-3 foi utilizado sem otimização específica de hiperparâmetros, por meio de inferência em contexto (*in-context learning*).

---

## Bases de dados

Foram utilizadas 15 bases de dados públicas provenientes principalmente do **UCI Machine Learning Repository** e do **Kaggle**.

| Base de dados | Área de aplicação | Fonte |
|---|---|---|
| DARWIN | Neurologia / diagnóstico assistido de Alzheimer | UCI |
| Drug Induced Autoimmunity Prediction | Toxicologia / farmacologia | UCI |
| Bone Marrow Transplant: Children | Hematologia pediátrica | UCI |
| Cervical Cancer (Risk Factors) | Oncologia / ginecologia | UCI |
| Cirrhosis Patient Survival Prediction | Hepatologia | UCI |
| Differentiated Thyroid Cancer Recurrence | Oncologia | UCI |
| Echocardiogram | Cardiologia | UCI |
| Estimation of Obesity Levels | Nutrição / saúde pública | UCI |
| Heart Failure Clinical Records | Cardiologia | UCI |
| Hepatitis | Hepatologia | UCI |
| Neurofibromatosis Type 1 (NF1) | Genética clínica | UCI |
| Parkinsons | Neurologia | UCI |
| Chronic Kidney Disease (Risk Factors) | Nefrologia | UCI |
| Student Depression Dataset | Psicologia / saúde mental | Kaggle |
| Student Stress Monitoring Datasets | Psicologia | Kaggle |

As bases foram selecionadas com diferentes tamanhos amostrais, quantidade de atributos e níveis de desbalanceamento, permitindo avaliar os modelos em cenários heterogêneos.

---

## Dados disponíveis neste repositório

As bases utilizadas no estudo estão organizadas em diretórios individuais:

```text
DADOS DO ESTUDO/
├── DARWIN/
├── Drug Induced Autoimmunity Prediction/
├── Bone marrow transplant_ children/
├── Cervical Cancer (Risk Factors)/
├── Cirrhosis Patient Survival Prediction/
├── Differentiated Thyroid Cancer Recurrence/
├── Echocardiogram/
├── Estimation of Obesity Levels Based On Eating Habits and Physical Condition/
├── Heart Failure Clinical Records/
├── Hepatitis/
├── Neurofibromatosis Type 1_ Clinical Symptoms of Familial and Sporadic Cases/
├── Parkinsons/
├── Risk Factor Prediction of Chronic Kidney Disease/
├── Student Depression Dataset/
└── Student Stress Monitoring Datasets/
````

Algumas bases são disponibilizadas diretamente em formatos como `.csv`, enquanto outras estão acompanhadas dos arquivos originais disponibilizados pelos respectivos repositórios.

---

## Pré-processamento

Para cada conjunto de dados, as variáveis foram separadas em numéricas e categóricas.

O pré-processamento utilizado nos modelos clássicos foi realizado por meio de um `ColumnTransformer`, contendo:

### Variáveis numéricas

- Imputação de valores ausentes pela mediana;
- Padronização utilizando `RobustScaler`.

### Variáveis categóricas

- Imputação de valores ausentes pela moda;
- Codificação One-Hot;
- Tratamento de categorias desconhecidas.

O pré-processamento foi incorporado a um `Pipeline` do Scikit-learn para garantir que o ajuste dos transformadores ocorresse exclusivamente nos dados de treinamento de cada fold, evitando vazamento de informação (*data leakage*).

### Prevenção de vazamento de dados

Durante o pré-processamento, alguns atributos foram removidos das bases de dados devido ao risco de **vazamento de dados (*data leakage*)**.

Esses atributos poderiam conter informações diretamente relacionadas ao resultado que se deseja prever ou informações que somente estariam disponíveis após o evento de interesse. A utilização dessas variáveis poderia permitir que os algoritmos obtivessem informações sobre a variável-alvo de forma inadequada, influenciando artificialmente o processo de decisão e resultando em uma estimativa de desempenho superior àquela esperada em uma aplicação real.

Dessa forma, os atributos identificados como potenciais fontes de vazamento foram eliminados antes da etapa de treinamento dos modelos.

A relação dos atributos removidos não é apresentada de forma centralizada neste README, pois as exclusões são especificadas individualmente em cada código, na seção **"Importando Data-Set"**, permitindo verificar diretamente quais variáveis foram removidas em cada conjunto de dados.

Essa etapa foi realizada de forma individual para cada base, considerando as características e o significado das respectivas variáveis.

Também foram removidas variáveis que apresentavam relação lógica direta com a variável-alvo quando necessário. Um exemplo foi a remoção da variável `survival time` da base Bone Marrow Transplant: Children.

## Modelos avaliados

### 1. Regressão Logística

Modelo estatístico utilizado para classificação binária.

### 2. k-Nearest Neighbors

Algoritmo baseado na proximidade entre observações no espaço de características.

### 3. Gaussian Naive Bayes

Classificador probabilístico baseado no Teorema de Bayes e na hipótese de independência condicional entre os atributos.

### 4. Support Vector Classifier

Modelo baseado na construção de uma fronteira de decisão que maximiza a margem entre as classes.

### 5. Artificial Neural Network

Rede neural artificial implementada por meio do `MLPClassifier` do Scikit-learn.

### 6. Decision Tree

Modelo baseado em uma sequência hierárquica de regras de decisão.

### 7. XGBoost

Algoritmo de *gradient boosting* baseado em árvores de decisão.

### 8. TabPFN-3

Modelo fundacional baseado na arquitetura Transformer, desenvolvido especificamente para dados tabulares. O modelo utiliza aprendizado em contexto (*in-context learning*) e não foi submetido ao processo de otimização de hiperparâmetros utilizado nos modelos clássicos.

---

## Otimização de hiperparâmetros

Os modelos clássicos foram otimizados utilizando o **Optuna**.

Foram realizados **25 trials** para cada algoritmo e conjunto de dados.

A função objetivo utilizada foi baseada na acurácia média obtida durante a validação cruzada estratificada.

Os espaços de busca incluíram diferentes hiperparâmetros específicos de cada algoritmo, como:

* `C`, `solver` e outros parâmetros para Regressão Logística;
* `n_neighbors`, `weights` e `p` para kNN;
* `var_smoothing` para GaussianNB;
* `C`, `kernel` e `gamma` para SVC;
* número de camadas, número de neurônios, função de ativação e taxa de aprendizado para ANN;
* profundidade e critérios de divisão para Decision Tree;
* número de estimadores, profundidade, taxa de aprendizado e regularização para XGBoost.

O TabPFN-3 não foi otimizado via Optuna.

---

## Validação

A avaliação dos modelos foi realizada utilizando **validação cruzada estratificada de 5 folds**:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

A estratificação foi utilizada para preservar a proporção das classes em cada partição.

O mesmo particionamento foi utilizado durante a otimização dos hiperparâmetros e durante a validação final.

---

## Métricas

Foram utilizadas as seguintes métricas:

* Accuracy
* Recall
* Precision
* AUC-ROC

Para a acurácia também foi calculado o desvio-padrão entre os folds.

O Recall e a Precision foram incluídos devido à presença de bases com desbalanceamento entre as classes.

Em algumas bases, a AUC-ROC do TabPFN-3 não pôde ser calculada de maneira consistente devido às características da implementação utilizada para acesso ao modelo, sendo registrada como `NaN`.

---

## Resultados principais

Os melhores algoritmos em termos de acurácia média para cada conjunto de dados foram:

| Base de dados                            | Melhor algoritmo | Acurácia |
| ---------------------------------------- | ---------------: | -------: |
| DARWIN                                   |          XGBoost |    0.908 |
| Drug Induced Autoimmunity Prediction     |         TabPFN-3 |    0.855 |
| Bone Marrow Transplant: Children         |         TabPFN-3 |    0.829 |
| Cervical Cancer (Risk Factors)           |              kNN |    0.938 |
| Cirrhosis Patient Survival Prediction    |          XGBoost |    0.813 |
| Differentiated Thyroid Cancer Recurrence |          XGBoost |    0.890 |
| Echocardiogram                           |          XGBoost |    0.918 |
| Estimation of Obesity Levels             |         TabPFN-3 |    0.935 |
| Heart Failure Clinical Records           |          XGBoost |    0.853 |
| Hepatitis                                |              ANN |    0.871 |
| Neurofibromatosis Type 1 (NF1)           |              ANN |    0.574 |
| Parkinsons                               |         TabPFN-3 |    0.949 |
| Chronic Kidney Disease                   |         TabPFN-3 |    1.000 |
| Student Depression Dataset               |         TabPFN-3 |    0.848 |
| Student Stress Monitoring Datasets       |          XGBoost |    0.937 |

O XGBoost e o TabPFN-3 apresentaram os maiores números de vitórias individuais, com seis bases cada.

---

## Médias globais

Os resultados médios dos modelos nas quinze bases foram:

| Modelo       | Accuracy média | Recall médio | Precision média | AUC média | Desvio da Accuracy |
| ------------ | -------------: | -----------: | --------------: | --------: | -----------------: |
| TabPFN-3     |      0.8666257 |    0.6625432 |       0.7387334 |       NaN |          0.0203539 |
| XGBoost      |      0.8612767 |    0.6731634 |       0.7830739 | 0.8602751 |          0.0288367 |
| LogReg       |      0.8470828 |    0.6274824 |       0.7304316 | 0.8252008 |          0.0288831 |
| SVC          |      0.8459480 |    0.6286239 |       0.6881865 | 0.8036378 |          0.0289764 |
| ANN          |      0.8384319 |    0.6315109 |       0.7141142 | 0.8012290 |          0.0308110 |
| DecisionTree |      0.8337619 |    0.6410159 |       0.7001208 | 0.7988617 |          0.0256163 |
| kNN          |      0.8263071 |    0.5899693 |       0.7568853 | 0.8096259 |          0.0328366 |
| GaussianNB   |      0.7385866 |    0.5428563 |       0.6945702 | 0.8036935 |          0.0538152 |

O TabPFN-3 apresentou a maior acurácia média entre os modelos avaliados, enquanto o XGBoost apresentou o maior Recall e a maior Precision média.

---

## Análise estatística

Foi utilizado o **teste de Friedman** para avaliar se existiam diferenças estatisticamente significativas entre os desempenhos dos oito algoritmos considerando as quinze bases como blocos.

O teste resultou em:

```text
χ²r = 40.07
graus de liberdade = 7
p ≈ 1.22 × 10⁻⁶
```

Como o resultado apresentou significância estatística, foi aplicado o teste post-hoc de **Nemenyi**.

Os resultados indicaram que:

* XGBoost apresentou rank médio de 2.57;
* TabPFN-3 apresentou rank médio de 2.80;
* XGBoost e TabPFN-3 foram estatisticamente equivalentes entre si;
* A diferença crítica foi de 2.71;
* GaussianNB e Decision Tree apresentaram desempenho inferior aos modelos de maior rank.

---

## Tecnologias utilizadas

O desenvolvimento computacional utilizou principalmente:

* Python
* Pandas
* NumPy
* Scikit-learn
* Optuna
* XGBoost
* TabPFN
* Matplotlib

O acesso ao TabPFN-3 foi realizado por meio do `tabpfn-client`.

---

## Estrutura geral do estudo

Dados
  │
  ▼
Pré-processamento
  │
  ├── Variáveis numéricas
  │     ├── Imputação pela mediana
  │     └── RobustScaler
  │
  └── Variáveis categóricas
        ├── Imputação pela moda
        └── One-Hot Encoding
  │
  ▼
Validação cruzada estratificada
5 folds
  │
  ├───────────────┐
  ▼               ▼
Modelos clássicos  TabPFN-3
  │               │
  ▼               ▼
Optuna           In-context
25 trials        learning
  │               │
  └───────┬───────┘
          ▼
Avaliação
  │
  ├── Accuracy
  ├── Recall
  ├── Precision
  └── AUC-ROC
          │
          ▼
Teste de Friedman
          │
          ▼
Teste de Nemenyi

---

## Referências das bases de dados

### DARWIN
FONTANELLA, Francesco. DARWIN. Irvine: UCI Machine Learning Repository, 2022. Disponível em: https://doi.org/10.24432/C55D0K. Acesso em: 11 jul. 2026.
Explicações das Variáveis: 
CILIA, Nicole D. et al. Diagnosing Alzheimer’s disease from on-line handwriting: A novel dataset and performance benchmarking. Engineering Applications of Artificial Intelligence, v. 111, p. 104822, 2022. Disponível em: https://doi.org/10.1016/j.engappai.2022.104822. Acesso em: 7 out. 2026.

### Drug Induced Autoimmunity Prediction
HUANG, Xiaojie. Drug induced autoimmunity prediction. Irvine: UCI Machine Learning Repository, 2025. Disponível em: https://doi.org/10.24432/C5332M. Acesso em: 11 jul. 2026.
Explicações das Variáveis: Arquivo *RDKit_ChemDes.xlsx* na mesma citação do banco de dados na seção Dataset Files.

### Bone Marrow Transplant: Children
SIKORA, Marek; WRÓBEL, Łukasz; GUDYŚ, Adam. Bone marrow transplant: children. Irvine: UCI Machine Learning Repository, 2020. Disponível em: https://doi.org/10.24432/C5NP6Z. Acesso em: 11 jul. 2026.
Explicações das Variáveis: Explicado na citação acima na seção Variables Table

### Cervical Cancer (Risk Factors)
FERNANDES, Kelwin; CARDOSO, Jaime; FERNANDES, Jessica. Cervical cancer (risk factors). Irvine: UCI Machine Learning Repository, 2017. Disponível em: https://doi.org/10.24432/C5Z310. Acesso em: 11 jul. 2026.
Explicações das Variáveis: Explicado na citação acima na seção Additional Variable Information

### Cirrhosis Patient Survival Prediction
DICKSON, E. et al. Cirrhosis patient survival prediction. Irvine: UCI Machine Learning Repository, 1989. Disponível em: https://doi.org/10.24432/C5R02G. Acesso em: 11 jul. 2026.
Explicações das Variáveis: Explicado na citação acima na seção Additional Variable Information

### Differentiated Thyroid Cancer Recurrence
BORZOOEI, Shiva; TAROKHIAN, Aidin. Differentiated thyroid cancer recurrence. Irvine: UCI Machine Learning Repository, 2023. Disponível em: https://doi.org/10.24432/C5632J. Acesso em: 11 jul. 2026.
Explicações das Variáveis: BORZOOEI, Shiva et al. Machine learning for risk stratification of thyroid cancer patients: a 15-year cohort study. European Archives of Oto-Rhino-Laryngology, v. 281, n. 4, p. 2095-2104, 2024. Disponível em: https://doi.org/10.1007/s00405-023-08299-w. Acesso em: 7 out. 2026.

### Echocardiogram
UCI MACHINE LEARNING REPOSITORY. Echocardiogram. Irvine: University of California, 1988. Disponível em: https://doi.org/10.24432/C5QW24. Acesso em: 11 jul. 2026.
Explicações das Variáveis: Explicado na citação acima na seção Additional Variable Information

### Estimation of Obesity Levels
UCI MACHINE LEARNING REPOSITORY. Estimation of obesity levels based on eating habits and physical condition. Irvine: University of California, 2019. Disponível em: https://doi.org/10.24432/C5H31Z. Acesso em: 11 jul. 2026.
Explicações das Variáveis: PALECHOR, Fabio Mendoza; MANOTAS, Alexis Gutierrez. Dataset for estimation of obesity levels based on eating habits and physical condition in individuals from Colombia, Peru and Mexico. Data in Brief, v. 25, p. 104344, 2019. Disponível em: https://doi.org/10.1016/j.dib.2019.104344. Acesso em: 7 out. 2026.

### Heart Failure Clinical Records
UCI MACHINE LEARNING REPOSITORY. Heart failure clinical records. Irvine: University of California, 2020. Disponível em: https://doi.org/10.24432/C5Z89R. Acesso em: 11 jul. 2026.
Explicações das Variáveis:Explicado na citação acima na seção Additional Variable Information

### Hepatitis
UCI MACHINE LEARNING REPOSITORY. Hepatitis. Irvine: University of California, 1983. Disponível em: https://doi.org/10.24432/C5Q59J. Acesso em: 11 jul. 2026.
Explicações das Variáveis:Explicado na citação acima na seção Additional Variable Information

### Neurofibromatosis Type 1
SHARAFI, P. et al. Neurofibromatosis Type 1; clinical symptoms of familial and sporadic cases. Turk Hijyen ve Deneysel Biyoloji Dergisi, 2025. Disponível em: https://doi.org/10.5505/TurkHijyen.2025.06337. Acesso em: 7 out. 2026. 
Explicações das Variáveis:Explicado na citação acima na seção Additional Variable Information

### Parkinsons
LITTLE, Max. Parkinsons. Irvine: UCI Machine Learning Repository, 2007. Disponível em: https://doi.org/10.24432/C59C74. Acesso em: 11 jul. 2026
Explicações das Variáveis:Explicado na citação acima na seção Additional Variable Information

### Chronic Kidney Disease
ISLAM, Md. Ashiqul; AKTER, Shamima. Risk factor prediction of chronic kidney disease. Irvine: UCI Machine Learning Repository, 2020. Disponível em: https://doi.org/10.24432/C5WP64. Acesso em: 11 jul. 2026.. 
Explicações das Variáveis: ISLAM, Md Ashiqul et al. Risk factor prediction of chronic kidney disease based on machine learning algorithms. In: INTERNATIONAL CONFERENCE ON INTELLIGENT SUSTAINABLE SYSTEMS (ICISS), 3., 2020, Thoothukudi. Proceedings [...]. IEEE, 2020. p. 952-957. Disponível em: https://doi.org/10.1109/ICISS49785.2020.9315878. Acesso em: 7 out. 2026.

### Student Depression Dataset
MOHAMMED, Saifeldeen. Student depression dataset. [S. l.]: Kaggle Code, 2024. Disponível em: https://www.kaggle.com/code/saifeldeenmohammedm/student-depression-dataset. Acesso em: 11 jul. 2026. 
Explicações das Variáveis: Especificados no Notebook da citação acima.

### Student Stress Monitoring Datasets
OVI, Md Sultanul Islam. Student stress monitoring datasets. [S. l.]: Kaggle Datasets, 2024. Disponível em: https://www.kaggle.com/datasets/mdsultanulislamovi/student-stress-monitoring-datasets. Acesso em: 11 jul. 2026.
Explicações das Variáveis: Especificados no Notebook da citação acima.

---

## Referências principais

Grinsztajn, L. et al.
TabPFN-3: Technical Report. arXiv preprint arXiv:2605.13986 (2026).

Akiba, T., Sano, S., Yanase, T., Ohta, T. & Koyama, M.
Optuna: A Next-generation Hyperparameter Optimization Framework (2019).

Chen, T.
XGBoost: A Scalable Tree Boosting System (2016).

Breiman, L.
Random Forests. Machine Learning 45, 5–32 (2001).

Cover, T. & Hart, P.
Nearest Neighbor Pattern Classification. IEEE Transactions on Information Theory 13, 21–27 (1967).

Quinlan, J. R.
Induction of Decision Trees. Machine Learning 1, 81–106 (1986).

Fawcett, T.
An Introduction to ROC Analysis. Pattern Recognition Letters 27, 861–874 (2006).

---

## Autores

**João Luiz Junho Pereira**
Instituto de Engenharia Mecânica
Universidade Federal de Itajubá — UNIFEI

**Davi dos Reis Soares**
Instituto de Engenharia de Produção e Gestão
Universidade Federal de Itajubá — UNIFEI

**Rayane Braga Machado**
Instituto de Engenharia Mecânica
Universidade Federal de Itajubá — UNIFEI

---

## Observação

Os dados disponibilizados neste repositório são provenientes dos repositórios públicos indicados nas referências. Os arquivos foram organizados para permitir a reprodução e consulta dos conjuntos de dados utilizados no estudo.

Para informações detalhadas sobre o processo de pré-processamento, otimização, validação, métricas e análise estatística, consulte o artigo associado a este repositório.

```
```
