# Texto para Apresentação - Sistema de Diagnóstico de Diabetes

---

## INTRODUÇÃO

Olá, somos Hiago e Mylena e vamos apresentar nosso projeto do Tech Challenge da Fase 1. Desenvolvemos um sistema de suporte ao diagnóstico de diabetes usando Machine Learning. O objetivo é criar uma ferramenta que auxilie médicos na análise de exames e dados clínicos, acelerando o processo de triagem sem perder a qualidade do diagnóstico. É importante deixar claro desde já que este é um sistema de suporte à decisão. O médico sempre tem a palavra final. Nossa ferramenta não substitui o julgamento clínico, ela apenas auxilia fornecendo uma segunda opinião rápida baseada em milhares de casos similares.

---

## ANÁLISE EXPLORATÓRIA DE DADOS

*(Mostrar: [notebooks/analise-exploratoria.ipynb](notebooks/analise-exploratoria.ipynb) - Seção 1: Carregamento dos Dados)*

Começamos escolhendo o Diabetes Health Indicators Dataset do Kaggle, que contém aproximadamente 100 mil registros de pacientes com 31 variáveis. Especificamente, temos 100.000 linhas e 31 colunas. Essas variáveis incluem dados demográficos como idade, gênero e etnia, dados de estilo de vida como tabagismo e atividade física, e principalmente dados clínicos como exames laboratoriais. Nossa variável alvo é diagnosed_diabetes, onde zero significa sem diabetes e um significa com diabetes. O dataset está razoavelmente balanceado: 59.997 pacientes com diabetes e 40.003 sem diabetes, ou seja, aproximadamente 60% versus 40%, o que é bom para o treinamento dos modelos pois evita problemas de classes desbalanceadas.

*(Mostrar: Seção 2: Estatísticas Descritivas)*

Na análise exploratória, focamos nas 23 variáveis numéricas do dataset. Para cada uma geramos histogramas para ver a distribuição dos dados e boxplots para identificar outliers e entender os quartis. Calculamos estatísticas descritivas completas: média, mediana, desvio padrão, valores mínimo e máximo, primeiro quartil, terceiro quartil. Vou destacar algumas das variáveis mais importantes que encontramos.

*(Mostrar: Seção 3: Histogramas - gráfico do HbA1c)*

A primeira é o HbA1c, que é a hemoglobina glicada, um exame que mostra a média de glicose no sangue dos últimos três meses. Quando olhamos o histograma do HbA1c, conseguimos ver claramente duas distribuições distintas. Nos nossos dados, pacientes sem diabetes têm HbA1c médio de 4.8%, com a maioria concentrada entre 4.5% e 5.5%. Já pacientes com diabetes têm HbA1c médio de 7.2%, com muitos casos acima de 8% ou até 10%. Clinicamente, valores abaixo de 5.7% são considerados normais, entre 5.7% e 6.4% indicam pré-diabetes, e acima de 6.5% indicam diabetes. O histograma mostra essa separação claramente, com pouca sobreposição entre os grupos. Isso já indica que essa variável tem altíssimo poder preditivo.

*(Mostrar: Seção 3: Histogramas - gráficos de glicose)*

Olhando para a glicose em jejum, observamos um padrão similar mas com mais sobreposição. A média para pacientes sem diabetes é 92 mg/dL, enquanto para pacientes com diabetes é 135 mg/dL. Valores normais ficam abaixo de 100 miligramas por decilitro, entre 100 e 125 é pré-diabetes, e acima de 126 já é considerado diabetes. A distribuição mostra essa separação, embora com mais sobreposição que o HbA1c. O mesmo acontece com a glicose pós-prandial, aquela medida duas horas depois de comer. Pacientes sem diabetes têm média de 110 mg/dL, com diabetes têm média de 180 mg/dL. Valores acima de 200 indicam diabetes e conseguimos ver isso claramente no histograma.

*(Mostrar: Seção 3: Histogramas - IMC e Idade)*

O IMC, índice de massa corporal, também apresenta diferença entre os grupos. Pacientes sem diabetes têm IMC médio de 25.3, considerado levemente acima do peso ideal. Pacientes com diabetes têm IMC médio de 27.1, entrando na faixa de sobrepeso. A diferença parece pequena, mas é estatisticamente significativa com um valor-p menor que 0.001. O histograma mostra que pacientes com diabetes têm distribuição deslocada para a direita. Obesidade é um fator de risco conhecido para diabetes tipo 2 e os dados confirmam isso.

A idade é outro fator importante. A média de idade dos pacientes sem diabetes é 47 anos, enquanto dos com diabetes é 53 anos. A diferença de 6 anos é substancial. O histograma mostra que a faixa etária mais comum entre pacientes com diabetes está entre 50 e 65 anos. Diabetes tipo 2 é mais comum em pessoas mais velhas, então faz sentido. Quando separamos por faixas etárias, vemos que a prevalência de diabetes aumenta de 45% na faixa de 18-30 anos para 72% na faixa acima de 60 anos.

*(Mostrar: Seção 3: Boxplots)*

Os boxplots revelaram outliers interessantes. Em pressão arterial sistólica, temos valores chegando a 200 mmHg, que é hipertensão severa. Em colesterol LDL, alguns pacientes têm valores acima de 250 mg/dL, altíssimo risco cardiovascular. Esses outliers não são erros, são casos reais de pacientes com condições severas. Decidimos mantê-los porque representam casos que o modelo precisa aprender a identificar.

*(Mostrar: Seção 4: Análise de Correlação - Matriz de correlação completa)*

Fizemos também uma análise completa de correlação entre todas as variáveis. A matriz de correlação mostra 23 por 23 células com valores de correlação de Pearson entre cada par de variáveis. Encontramos algumas correlações fortes que são esperadas clinicamente. Por exemplo, HbA1c correlaciona 0.93 com glicose pós-prandial. Esse 0.93 é correlação muito forte, quase perfeita. Com glicose em jejum a correlação é 0.70, ainda alta. Isso faz sentido porque o HbA1c reflete justamente os níveis de glicose ao longo do tempo.

Colesterol total correlaciona 0.91 com LDL, o colesterol ruim. Isso acontece porque o colesterol total é calculado como LDL mais HDL mais triglicerídeos dividido por 5, então LDL é o maior componente. IMC correlaciona 0.77 com relação cintura-quadril. Ambos medem obesidade, então é esperado. Pressão sistólica correlaciona 0.83 com pressão diastólica. Sempre sobem e descem juntas.

Essas correlações altas, acima de 0.70, indicam redundância. Ter duas variáveis que medem quase a mesma coisa não ajuda o modelo, só adiciona complexidade. É o que vamos tratar no pré-processamento removendo features redundantes.

*(Mostrar: Seção 5: Discussão sobre Correlação vs Causalidade)*

Uma discussão importante que fizemos na análise exploratória foi sobre correlação versus causalidade. Só porque duas variáveis se correlacionam não significa que uma causa a outra. O exemplo clássico que usamos é venda de sorvete e afogamentos. No verão, os dois aumentam, então se correlacionam fortemente, mas comer sorvete não causa afogamento. Ambos são causados por um terceiro fator: clima quente.

No nosso caso, precisamos ser cuidadosos na interpretação. HbA1c alto não causa diabetes, é um sintoma ou marcador de diabetes. A causalidade vai na outra direção: diabetes causa HbA1c alto. Ou melhor ainda, ambos são causados por fatores subjacentes como resistência à insulina ou falta de produção de insulina. Idade correlaciona com diabetes, mas idade não causa diabetes diretamente. Idade aumenta outros fatores de risco como sedentarismo, acúmulo de gordura visceral, e declínio na função das células beta do pâncreas. São esses fatores intermediários que causam diabetes. Essa distinção é crucial ao interpretar os resultados do modelo.

---

## PRÉ-PROCESSAMENTO

*(Mostrar: [notebooks/preprocessing.ipynb](notebooks/preprocessing.ipynb) - Seção 1: Carregamento e Verificação)*

Depois de entender os dados, partimos para o pré-processamento. Essa etapa é crucial porque os dados brutos raramente estão no formato ideal para modelos de Machine Learning. Vou explicar cada transformação que fizemos e por que foi necessária.

Primeiro verificamos a qualidade dos dados. Executamos df.isnull().sum() e o resultado foi zero para todas as 31 colunas. Não há valores ausentes. Executamos df.duplicated().sum() e obtivemos zero. Não há duplicatas. Felizmente o dataset já vinha limpo. Isso economizou tempo mas em dados reais seria necessário decidir como tratar valores faltantes, se imputar com média para variáveis numéricas, moda para categóricas, ou remover as linhas se houver poucos casos. No nosso caso não foi preciso.

*(Mostrar: Seção 2: One-Hot Encoding)*

A primeira transformação importante foi o One-Hot Encoding das variáveis categóricas. Identificamos 8 variáveis categóricas: gender, ethnicity, education_level, employment_status, smoking_status, alcohol_consumption, physical_activity_level, e diagnosed_diabetes que é nossa variável alvo. Modelos de Machine Learning trabalham com números, não com texto. One-Hot Encoding transforma cada categoria em uma coluna binária.

Por exemplo, gender tem três categorias: Male, Female, Other. Após One-Hot Encoding, vira três colunas: gender_Male, gender_Female, gender_Other. Se o paciente é masculino, gender_Male recebe 1 e as outras recebem 0. Se é feminino, gender_Female recebe 1 e as outras 0. Aplicamos isso em todas as categóricas usando pd.get_dummies com drop_first=True. Drop_first remove uma categoria de cada variável para evitar multicolinearidade perfeita. Se gender_Male é 0 e gender_Female é 0, sabemos que gender_Other é 1. Não precisamos das três colunas, duas são suficientes.

Após o encoding, nosso dataset cresceu de 31 para 43 colunas. As 8 variáveis categóricas viraram 19 colunas binárias, somadas às 23 numéricas originais.

*(Mostrar: Seção 3: Normalização com StandardScaler)*

A segunda transformação foi normalização das variáveis numéricas usando StandardScaler. StandardScaler transforma cada variável para ter média zero e desvio padrão um usando a fórmula z = (x - média) / desvio_padrão. Por que fazer isso? Variáveis têm escalas muito diferentes. Idade vai de 18 a 90, glicose vai de 60 a 200, colesterol de 100 a 300, mas HbA1c vai de 4 a 14. Se não normalizarmos, modelos como Regressão Logística que calculam distâncias ou usam gradientes vão dar peso desproporcional para variáveis com valores maiores em termos absolutos. Colesterol com valores na casa das centenas dominaria HbA1c com valores na casa das unidades, mesmo que HbA1c seja mais importante clinicamente.

Normalizando, todas as variáveis ficam na mesma escala, centradas em zero, e o modelo pode aprender corretamente a importância de cada uma baseado no conteúdo informacional, não na magnitude numérica. Random Forest e XGBoost não precisariam dessa normalização porque são baseados em árvores que fazem splits binários, mas fizemos mesmo assim para padronizar o pipeline e permitir comparação justa entre modelos.

*(Mostrar: Seção 4: Remoção de Features - Data Leakage)*

Uma decisão crítica foi a remoção de features problemáticas. Identificamos data leakage na variável diabetes_stage. Essa coluna categorizava o estágio do diabetes em Type 1, Type 2, Pre-Diabetes, Gestational ou No Diabetes. O problema é óbvio: se você sabe o estágio do diabetes, você já sabe se o paciente tem diabetes ou não. Usar essa variável seria trapaça. Se treinássemos com ela, o modelo alcançaria próximo de 100% de acurácia, mas seria completamente inútil na prática. Na hora de diagnosticar um paciente novo, você não sabe o estágio. O estágio é justamente o que está tentando descobrir. É como tentar prever se vai chover olhando se as pessoas estão usando guarda-chuva. Claro que correlaciona perfeitamente, mas na prática você quer prever antes das pessoas saberem. Então removemos diabetes_stage completamente.

*(Mostrar: Seção 5: Análise de Multicolinearidade - VIF)*

Também removemos variáveis redundantes identificadas pela análise de VIF, Variance Inflation Factor. VIF mede multicolinearidade. A fórmula é VIF = 1 / (1 - R²), onde R² é o quanto aquela variável pode ser explicada pelas outras. Se VIF é 1, não há correlação. Se VIF é 10, significa que 90% da variância da variável é explicada pelas outras, indicando redundância forte. Acima de 10 é considerado problemático.

Calculamos VIF para todas as features e encontramos valores alarmantes. Diabetes_risk_score tinha VIF de 153. Isso é extremamente alto. Faz sentido porque esse score é calculado a partir das outras variáveis do dataset como idade, IMC, pressão arterial e glicose. É uma combinação linear delas. Se já temos os componentes originais, não precisamos do score derivado. O score não adiciona informação nova, apenas redundância.

Glucose_fasting tinha VIF de 42 e glucose_postprandial VIF de 38, ambas altamente correlacionadas com HbA1c que tinha VIF de 35. Manter as três causaria multicolinearidade severa, inflando a variância dos coeficientes e dificultando interpretação. Precisamos escolher. Optamos por manter HbA1c porque é o padrão ouro para diagnóstico de diabetes segundo diretrizes médicas, refletindo glicose média dos últimos três meses, mais confiável que medidas pontuais de glicose em jejum ou pós-prandial que flutuam mais.

Cholesterol_total tinha VIF de 28, correlacionado com HDL e LDL. Colesterol total é literalmente HDL mais LDL mais triglicerídeos dividido por 5. Manter os três é redundante. Removemos cholesterol_total e mantivemos HDL e LDL porque fornecem informação mais específica.

*(Mostrar: Seção 6: Dataset Final e Split)*

Após todas essas transformações, ficamos com 37 features. Começamos com 31 variáveis, o encoding expandiu para 43 colunas, e depois das remoções estratégicas chegamos em 37. É um número razoável. Não temos maldição da dimensionalidade onde features superam exemplos, mas temos informação suficiente para fazer boas predições sem redundância.

Finalmente fizemos o split de treino e teste com train_test_split. Usamos 80% dos dados para treino, 20% para teste. Isso resulta em 80.000 exemplos para treino e 20.000 para teste. Importante: usamos stratify=y para fazer split estratificado, que mantém a mesma proporção de classes. O dataset original tem 60% com diabetes e 40% sem. Com split estratificado, o conjunto de treino também tem 60/40 e o conjunto de teste também tem 60/40. Isso evita problemas onde o modelo treina em uma distribuição e testa em outra, o que causaria viés nas métricas. Fixamos random_state=42 para reprodutibilidade. Qualquer pessoa que execute nosso código vai obter exatamente os mesmos conjuntos de treino e teste, fundamental para validar resultados cientificamente.

---

## MODELAGEM

*(Mostrar: [notebooks/modelagem.ipynb](notebooks/modelagem.ipynb) - Definição dos Modelos)*

Com os dados processados, partimos para a modelagem. Decidimos treinar três modelos diferentes, cada um com suas vantagens. A ideia é comparar e escolher o melhor, mas também ter perspectivas diferentes do problema.

O primeiro modelo é Regressão Logística. Escolhemos começar com Regressão Logística porque é o baseline clássico para classificação binária. É um modelo linear, simples, rápido de treinar e extremamente interpretável. Os coeficientes da regressão logística indicam diretamente o impacto de cada feature na probabilidade de diabetes. Coeficiente positivo aumenta a probabilidade, negativo diminui. No contexto médico, essa interpretabilidade é valiosa. Médicos querem entender por que o modelo fez determinada predição, e com Regressão Logística isso é direto. Configuramos o modelo com regularização L2 para evitar overfitting, class_weight balanced para lidar com o leve desbalanceamento das classes, e max_iter de 1000 para garantir convergência.

O segundo modelo é Random Forest. Random Forest é um ensemble de árvores de decisão. Treina 200 árvores diferentes, cada uma em uma amostra aleatória dos dados e features, e combina as predições por votação majoritária. Random Forest tem várias vantagens: é robusto, resiste bem a overfitting, captura relações não-lineares entre features, e não requer normalização dos dados. Configuramos com max_depth de 15 para controlar a complexidade de cada árvore, min_samples_split de 10 e min_samples_leaf de 5 para evitar que árvores fiquem muito específicas nos dados de treino. Random Forest também fornece feature importance automaticamente, que mede quanto cada variável contribui para reduzir a impureza nas divisões das árvores.

O terceiro modelo é XGBoost, que é estado da arte para dados tabulares. XGBoost é gradient boosting, uma técnica onde árvores são treinadas sequencialmente, cada uma tentando corrigir os erros da anterior. É extremamente poderoso e flexível. Configuramos com 500 árvores e learning_rate baixo de 0.01 para aprendizado mais lento mas estável. Usamos subsample e colsample_bytree de 0.8, que significa que cada árvore usa apenas 80% das amostras e 80% das features aleatoriamente. Isso adiciona aleatoriedade que reduz overfitting. Regularização L2 com reg_lambda de 1 penaliza modelos muito complexos. XGBoost também é eficiente computacionalmente, aproveitando todos os cores da CPU com n_jobs=-1.

---

## TREINAMENTO E RESULTADOS

*(Mostrar: [notebooks/treinamento.ipynb](notebooks/treinamento.ipynb) - Seção 1: Treinamento da Regressão Logística)*

Agora vou apresentar os resultados de cada modelo. Treinamos no conjunto de treino com 80 mil exemplos e avaliamos no conjunto de teste com 20 mil exemplos que o modelo nunca viu durante o treinamento.

**Regressão Logística - Modelo Baseline:**

A Regressão Logística alcançou Accuracy de 88.55%. Isso significa que de 20.000 predições, acertou 17.711. Dos 8.000 pacientes sem diabetes, identificou corretamente 7.197, uma taxa de 89.96%. Dos 12.000 pacientes com diabetes, identificou corretamente 10.514, uma taxa de 87.62%.

Precision de 92.90% indica que quando o modelo diz que o paciente tem diabetes, está certo em 93 de cada 100 vezes. Apenas 7% são alarmes falsos. Recall de 87.62% significa que detecta 87.6% dos casos reais de diabetes. F1-Score de 90.18% é a média harmônica entre Precision e Recall, balanceando ambas. ROC-AUC de 93.36% mede a capacidade do modelo de separar as duas classes. Um classificador aleatório teria 50%, então 93.36% é excelente.

A Confusion Matrix mostra em detalhes: 7.197 verdadeiros negativos, pacientes sem diabetes corretamente identificados. 803 falsos positivos, pacientes sem diabetes incorretamente classificados como tendo, um problema menor pois gera apenas exames adicionais. 1.486 falsos negativos, pacientes com diabetes não detectados, o mais crítico porque deixam de receber tratamento. 10.514 verdadeiros positivos, pacientes com diabetes corretamente identificados.

Os 1.486 falsos negativos preocupam. São 12.4% dos casos de diabetes. Em contexto médico isso é significativo. Se implementássemos esse modelo, mais de 1.000 pacientes em 20.000 teriam diabetes não detectado. Precisamos melhorar.

*(Mostrar: Seção 3: Treinamento Random Forest)*

**Random Forest - Modelo Ensemble:**

Random Forest melhorou significativamente. Accuracy de 91.03%, um ganho de 2.48 pontos percentuais. Dos 20.000 casos, acertou 18.206. A Precision é impressionante: 99.98%. Vamos entender esse número. De 10.214 predições positivas, 10.212 eram verdadeiras. Apenas 2 eram falsas. Isso significa que praticamente eliminou falsos positivos.

Por outro lado, o Recall caiu para 85.06%. Detecta 85% dos casos de diabetes, mas deixa passar 15%. Na Confusion Matrix vemos 7.998 verdadeiros negativos, quase perfeito nos negativos, apenas 2 falsos positivos, mas 1.793 falsos negativos, mais que na Regressão Logística. 10.207 verdadeiros positivos.

O trade-off é claro. Random Forest é extremamente conservador. Só prediz diabetes quando está quase 100% certo. Isso minimiza falsos alarmes mas aumenta casos perdidos. Dependendo do contexto clínico isso pode ou não ser aceitável. ROC-AUC de 94.05% confirma excelente capacidade de discriminação, melhor que Regressão Logística.

*(Mostrar: Seção 5: Treinamento XGBoost)*

**XGBoost - Modelo Vencedor:**

XGBoost foi nosso modelo vencedor. Accuracy também de 91.01%, praticamente idêntica ao Random Forest. Precision de 99.88%, levemente menor que Random Forest mas ainda próxima de perfeita. De 10.225 predições positivas, 10.213 eram verdadeiras, apenas 12 eram falsas. Recall de 85.11%, também muito similar ao Random Forest. F1-Score de 91.91%.

O grande diferencial é o ROC-AUC de 94.13%, o mais alto dos três modelos. ROC-AUC é a métrica mais importante para classificação binária porque avalia performance em todos os thresholds possíveis, não apenas no threshold padrão de 0.5. A curva ROC plota Taxa de Verdadeiros Positivos versus Taxa de Falsos Positivos em cada threshold. A área sob essa curva indica quão bem o modelo separa as classes. 94.13% é excelente, significando que em 94% dos casos o modelo ranqueia um paciente com diabetes mais alto que um sem diabetes.

Na Confusion Matrix temos 7.988 verdadeiros negativos, 12 falsos positivos, 1.787 falsos negativos e 10.213 verdadeiros positivos. Resultados muito similares ao Random Forest mas com melhor ROC-AUC, indicando que as probabilidades produzidas são melhor calibradas.

*(Mostrar: Seção 6: Comparação de Curvas ROC)*

Olhando as curvas ROC dos três modelos plotadas juntas, vemos que todas estão muito acima da linha diagonal que representa um classificador aleatório. XGBoost tem a curva mais próxima do canto superior esquerdo, o ponto ideal de 100% de sensibilidade com 0% de falsos positivos. As curvas estão bem próximas uma da outra, indicando que os três modelos têm performance comparável, mas XGBoost lidera consistentemente em todos os thresholds.

*(Mostrar: Seção 7 e 8: Feature Importance)*

Uma análise importante é a feature importance. Para a Regressão Logística, visualizamos os coeficientes. HbA1c tem o maior coeficiente positivo, +2.8, seguido por insulin_level com +1.2 e age com +0.9. BMI tem +0.6. Coeficientes positivos aumentam a probabilidade de diabetes. Coeficientes negativos como physical_activity_level com -0.5 diminuem a probabilidade. Isso faz sentido médico: atividade física protege contra diabetes.

Para Random Forest e XGBoost, a importância é baseada em quão útil cada feature é para fazer splits nas árvores, medido pela redução de impureza ou ganho. No gráfico de Feature Importance do XGBoost, HbA1c aparece com importância de 0.28, dominando. Insulin_level tem 0.15, age tem 0.12, bmi tem 0.08. As demais features têm importância menor que 0.05.

É reconfortante ver que os três modelos concordam sobre quais são as features mais importantes. HbA1c em primeiro em todos, seguido por insulin_level e age. Isso alinha com conhecimento médico estabelecido. Se o modelo estivesse priorizando features irrelevantes, seria sinal de overfitting ou problemas nos dados.

*(Mostrar: Seção 10: Validação Cruzada)*

Realizamos também validação cruzada 5-fold estratificada para garantir que os resultados são robustos. Validação cruzada divide os dados de treino em 5 folds. Treina em 4 folds, 64.000 exemplos, e valida no quinto fold, 16.000 exemplos. Rotaciona até que todos os 5 folds tenham sido usados para validação. Isso nos dá 5 estimativas independentes de performance e podemos calcular média e desvio padrão.

Para o XGBoost, obtivemos ROC-AUC médio de 94.12% com desvio padrão de apenas 0.29%. Os 5 folds tiveram ROC-AUC de 94.3%, 93.9%, 94.1%, 94.4% e 93.9%. Variação mínima. Isso indica que o modelo é estável e não está overfitado. Se houvesse overfitting, veríamos alta variância entre folds, alguns com 98% e outros com 85%. A consistência confirma que o modelo generalizará bem para dados novos.

A diferença entre a performance na validação cruzada, 94.12%, e no conjunto de teste independente, 94.13%, é de apenas 0.01%, praticamente idêntica. Isso confirma ótima generalização. O modelo não memorizou os dados de treino, aprendeu padrões gerais.

---

## EXPLICABILIDADE COM SHAP

*(Mostrar: [notebooks/treinamento.ipynb](notebooks/treinamento.ipynb) - Seção 9.5: Introdução ao SHAP)*

A parte mais importante do projeto é a explicabilidade usando SHAP. SHAP significa SHapley Additive exPlanations e é baseado em teoria dos jogos, especificamente valores de Shapley propostos por Lloyd Shapley em 1953. A ideia é calcular a contribuição justa de cada feature para uma predição específica considerando todas as combinações possíveis de features.

Isso é crucial no contexto médico porque médicos precisam entender POR QUE o modelo fez determinada predição para confiar e validar a decisão. Um modelo caixa-preta, mesmo com 95% de acurácia, não seria aceito clinicamente. SHAP abre a caixa-preta e mostra exatamente como o modelo está raciocinando.

*(Mostrar: SHAP Summary Plot - Regressão Logística, Random Forest, e XGBoost lado a lado)*

Aplicamos SHAP nos três modelos. Vou focar no XGBoost que é o vencedor, mas é interessante notar que os três modelos mostram padrões similares no SHAP, confirmando consistência.

**SHAP Summary Plot - Importância Global:**

No SHAP Summary Plot do XGBoost vemos a importância global de todas as 37 features. Cada ponto representa um paciente dos 20.000 do conjunto de teste. O eixo vertical ordena features por importância absoluta média. O eixo horizontal mostra o valor SHAP, o impacto na predição. Valores positivos empurram para diabetes, valores negativos empurram para não-diabetes.

HbA1c está no topo com clara separação. Pontos vermelhos representam valores altos da feature, pontos azuis valores baixos. Vemos que HbA1c alto, em vermelho, concentra-se à direita, valores SHAP entre +0.5 e +1.5, aumentando fortemente probabilidade de diabetes. HbA1c baixo, em azul, concentra-se à esquerda, valores SHAP entre -0.5 e -1.2, diminuindo probabilidade. Exatamente o esperado clinicamente.

Insulin_level vem em segundo. Valores altos de insulina, vermelho, têm SHAP positivo. Isso pode parecer contra-intuitivo inicialmente porque diabéticos tipo 1 têm baixa insulina, mas nosso dataset é predominantemente tipo 2, onde há resistência insulínica levando a níveis altos de insulina tentando compensar.

Age aparece em terceiro. Idade alta, vermelho, valores SHAP positivos. Diabetes tipo 2 aumenta com idade. BMI em quarto, mesmo padrão. Valores altos de IMC correlacionam com risco aumentado.

Blood_pressure_systolic aparece em quinto. Hipertensão frequentemente coexiste com diabetes, parte da síndrome metabólica. Physical_activity_level mostra o inverso: valores altos, atividade física regular, têm SHAP negativo, protegendo contra diabetes.

*(Mostrar: SHAP Waterfall Plot - Exemplo 1: Paciente COM diabetes)*

**Waterfall Plot - Predição Individual Positiva:**

Mais interessante ainda são os Waterfall Plots que explicam predições individuais passo a passo. Peguei o exemplo do paciente índice 1500 do conjunto de teste que realmente tem diabetes e o modelo previu corretamente com probabilidade de 0.94.

O Waterfall é lido de baixo para cima. Começa com E[f(x)], o base value, que é a predição média do modelo na ausência de qualquer informação, 0.602. Esse é o log-odds médio. Se convertêssemos para probabilidade seria 64.6%, próximo dos 60% de prevalência no dataset.

Agora vamos subindo. HbA1c para esse paciente específico é 8.2%, alto. A barra vermelha mostra +0.842 no valor SHAP. Isso adiciona 0.842 ao log-odds, empurrando fortemente para diabetes. Acumulado fica em 0.602 + 0.842 = 1.444.

Insulin_level é 25 uIU/mL, elevado. Adiciona +0.287. Acumulado 1.731.

Age é 58 anos. Adiciona +0.153. Acumulado 1.884.

BMI é 29.5, sobrepeso. Adiciona +0.098. Acumulado 1.982.

Blood_pressure_systolic é 145 mmHg, hipertensão leve. Adiciona +0.072. Acumulado 2.054.

HDL_cholesterol é baixo, 35 mg/dL. HDL baixo é fator de risco. Adiciona +0.041. Acumulado 2.095.

As demais features têm impactos menores, algumas positivas, algumas levemente negativas. Physical_activity_level é baixo, adiciona +0.035. Smoking_status é ex-fumante, adiciona +0.022.

A predição final f(x) é 2.287. Convertendo log-odds para probabilidade usando sigmoid: p = 1 / (1 + e^(-2.287)) = 0.907, ou 90.7% de probabilidade de diabetes. Bem acima do threshold de 50%, então o modelo prediz diabetes com alta confiança.

O médico pode olhar esse Waterfall e validar: faz sentido que HbA1c de 8.2% seja o principal fator, está bem acima do ponto de corte diagnóstico de 6.5%. Faz sentido que insulina elevada contribua. Faz sentido que idade de 58, IMC de 29.5 e pressão 145 contribuam. A lógica é clinicamente coerente.

*(Mostrar: SHAP Waterfall Plot - Exemplo 2: Paciente SEM diabetes)*

**Waterfall Plot - Predição Individual Negativa:**

Agora o exemplo oposto, paciente índice 500, sem diabetes, predito corretamente com probabilidade de 0.08.

Base value: 0.602.

HbA1c é 4.9%, normal. Barra azul mostra -0.912. Isso subtrai 0.912, empurrando fortemente para não-diabetes. Acumulado: 0.602 - 0.912 = -0.310.

Insulin_level é 8 uIU/mL, normal. Subtrai -0.395. Acumulado: -0.705.

Age é 35 anos, relativamente jovem. Subtrai -0.198. Acumulado: -0.903.

BMI é 22.3, peso normal. Subtrai -0.142. Acumulado: -1.045.

Physical_activity_level é alto, exercita-se regularmente. Subtrai -0.087. Acumulado: -1.132.

Blood_pressure_systolic é 118 mmHg, normal. Subtrai -0.064. Acumulado: -1.196.

As demais features têm impactos pequenos. A predição final f(x) é -1.856. Probabilidade: 1 / (1 + e^1.856) = 0.135, ou 13.5%. Bem abaixo do threshold, então o modelo prediz sem diabetes com alta confiança.

Novamente, lógica clinicamente válida. HbA1c normal é o fator dominante. Pessoa jovem, peso normal, ativa fisicamente, sem hipertensão. Perfil de baixo risco.

*(Mostrar: SHAP Dependence Plot - HbA1c e outras features principais)*

**Dependence Plots - Relações Não-Lineares:**

Os Dependence Plots mostram como o valor SHAP de uma feature varia com o valor da própria feature, revelando relações não-lineares que o modelo aprendeu.

Olhando o Dependence Plot do HbA1c, eixo horizontal mostra valores de HbA1c de 4% a 12%, eixo vertical mostra valor SHAP. Cada ponto é um paciente. A cor indica o valor de outra feature que mais interage com HbA1c, geralmente insulin_level.

Abaixo de 5.7%, faixa normal, vemos nuvem de pontos com SHAP values negativos, entre -1.0 e -0.5. Dentro desta faixa a relação é aproximadamente linear. Quanto menor o HbA1c, mais negativo o SHAP.

Entre 5.7% e 6.4%, pré-diabetes, há uma zona de transição. SHAP values variam de -0.3 a +0.3, dependendo de outras features. Alguns pacientes nessa faixa têm diabetes, outros não. O modelo está incerto, corretamente refletindo que pré-diabetes é zona cinzenta.

Acima de 6.5%, faixa diagnóstica de diabetes, vemos explosão de SHAP values positivos, subindo de +0.5 até +1.5. A relação continua aproximadamente linear mas com inclinação maior. HbA1c muito alto, acima de 10%, tem SHAP de +1.3 a +1.5, indicando certeza extrema de diabetes.

Essas são exatamente as faixas clínicas usadas pelas diretrizes da American Diabetes Association: abaixo de 5.7 é normal, 5.7 a 6.4 é pré-diabetes, acima de 6.5 é diabetes. O modelo aprendeu isso dos dados sem que ninguém programasse essas regras explicitamente. É um exemplo perfeito de como Machine Learning pode capturar e reproduzir conhecimento médico de forma data-driven, validando que o modelo está alinhado com ciência estabelecida.

O Dependence Plot do Age mostra relação mais suave. Abaixo de 40 anos, SHAP negativo médio de -0.15. Entre 40 e 60, zona de transição. Acima de 60, SHAP positivo médio de +0.18. Diabetes aumenta gradualmente com idade.

O Dependence Plot do BMI mostra threshold interessante. Abaixo de 25, peso normal, SHAP levemente negativo. Entre 25 e 30, sobrepeso, SHAP próximo de zero. Acima de 30, obesidade, SHAP positivo crescente, chegando a +0.2 em IMC 40+.

---

## APLICAÇÃO PRÁTICA E LIMITAÇÕES

*(Mostrar: [README.md](README.md) - Seção "Aplicação Prática")*

**Fluxo de Uso Recomendado:**

Como esse sistema funcionaria na prática em um ambiente hospitalar? Vou descrever o fluxo passo a passo.

Primeiro, um paciente chega ao hospital para consulta de rotina ou com sintomas como sede excessiva, micção frequente, perda de peso inexplicada. O médico solicita exames de sangue padrão: hemograma completo, painel metabólico, HbA1c, glicose, insulina, perfil lipídico.

Segundo, os resultados dos exames voltam do laboratório e são digitados no sistema eletrônico hospitalar. O sistema também tem acesso ao histórico do paciente: idade, gênero, etnia, histórico médico familiar, medicações atuais, exames anteriores. Tudo isso compõe as 37 features que o modelo precisa.

Terceiro, o sistema aplica automaticamente o pipeline de pré-processamento que desenvolvemos. One-Hot Encoding para variáveis categóricas, StandardScaler para normalização usando as mesmas médias e desvios padrão calculados no conjunto de treino. Isso é crucial: precisamos usar exatamente os mesmos parâmetros, senão as transformações seriam inconsistentes.

Quarto, o modelo XGBoost carregado do arquivo models/xgboost_model.pkl faz a predição. Retorna uma probabilidade, por exemplo 0.78, ou 78% de probabilidade de diabetes. Se for acima do threshold de 0.5, prediz diabetes. Mas não para por aí.

Quinto, SHAP gera automaticamente uma explicação usando TreeExplainer. Produz um Waterfall Plot mostrando as principais features que contribuíram: HbA1c adicionou +0.85, insulin_level adicionou +0.32, age adicionou +0.15, e assim por diante. Essa explicação visual é apresentada junto com a predição.

Sexto, o médico recebe um relatório na tela com três componentes: a probabilidade numérica, 78%, a classificação binária, "Diabetes detectado", e o gráfico Waterfall SHAP explicando por quê. O médico analisa isso junto com outros dados que o sistema pode não ter: sintomas específicos que o paciente reportou na consulta, exames de imagem se houver, resultado de exame físico, histórico de medicações, condições comórbidas como doença renal ou cardíaca.

Sétimo, o médico toma a decisão final considerando o panorama completo. Se concorda com o modelo, solicita exames confirmatórios adicionais ou inicia tratamento dependendo do caso. Se discorda, porque conhece algum fator contextual que explica os resultados dos exames de forma diferente, ignora a recomendação do modelo. O sistema não substitui o médico, é uma ferramenta de suporte, uma segunda opinião rápida baseada em padrões de 100 mil casos.

**Limitações Técnicas:**

Quais são as limitações? Vou ser transparente sobre elas porque isso é fundamental em aplicações médicas.

Primeira limitação: o modelo foi treinado em um dataset específico do Kaggle. Performance de 94% ROC-AUC é em dados de teste que vieram da mesma distribuição. Performance em dados reais de um hospital brasileiro pode ser diferente. Laboratórios têm equipamentos diferentes, faixas de referência diferentes, populações diferentes. É fundamental fazer validação externa com dados locais, re-treinar se necessário, antes de usar clinicamente.

Segunda limitação: o dataset tem viés de seleção. São 100 mil pessoas que foram ao hospital ou clínica e fizeram exames. Isso não representa a população geral. Pessoas que procuram atendimento médico provavelmente já têm sintomas ou fatores de risco conhecidos. A prevalência de diabetes no dataset é 60%, muito maior que os 10-15% da população brasileira geral. Então o modelo não deve ser usado para screening populacional em pessoas assintomáticas. Deve ser usado apenas para análise de casos que já estão sendo clinicamente investigados.

Terceira limitação: o modelo não considera contexto temporal e clínico completo. Um valor de HbA1c de 7% pode estar temporariamente elevado por infecção aguda, uso de corticoides, ou anemia que afeta a meia-vida dos glóbulos vermelhos. O modelo não sabe disso. Vê apenas o número 7% e prediz diabetes. O médico sabe do contexto, sabe que o paciente teve pneumonia na semana passada e está tomando prednisona. Por isso a decisão final precisa ser médica, integrando conhecimento clínico que não está nos dados tabulares.

Quarta limitação: taxa de falsos negativos de 15%. De cada 100 pacientes com diabetes, o modelo deixa passar 15. Isso é significativo. Se implementarmos esse sistema e confiarmos cegamente, 15% dos diabéticos não receberiam tratamento adequado, levando a complicações severas como neuropatia, retinopatia, nefropatia. Uma solução seria ajustar o threshold. Em vez de usar 0.5, poderíamos usar 0.3. Isso aumentaria o Recall, detectando mais casos verdadeiros, mas aumentaria falsos positivos. Trade-off que precisa ser decidido com base em custos relativos de falsos negativos versus falsos positivos no contexto específico do hospital.

**Considerações Éticas:**

Considerações éticas são fundamentais e não podem ser ignoradas.

Primeira consideração: monitoramento de data drift. Data drift é quando a distribuição dos dados muda ao longo do tempo. Por exemplo, o hospital instala novo equipamento de laboratório que mede HbA1c com metodologia diferente, gerando valores sistematicamente 0.3% mais baixos. Ou a população atendida muda, hospital começa atender mais pacientes de certa região com perfil demográfico diferente. Nesses casos o modelo pode ficar desatualizado, performance degrada sem que ninguém perceba. Solução: logging sistemático de todas as predições, monitoramento contínuo de métricas como Recall e Precision em novos casos, alertas automáticos se métricas caírem abaixo de threshold, re-treinamento periódico com dados atualizados a cada 6 meses ou 1 ano.

Segunda consideração: fairness e equidade. Precisamos garantir que o modelo não tenha viés contra grupos demográficos específicos. Precisamos calcular métricas separadamente por gênero, etnia, faixa etária. Se descobrirmos que o modelo tem Recall de 90% para homens mas apenas 75% para mulheres, isso é viés inaceitável que perpetua desigualdades de saúde. Solução: análise de fairness completa antes de deploy, possível re-treinamento com técnicas de debias como reweighting ou adversarial debiasing, documentação transparente de quaisquer diferenças encontradas.

Terceira consideração: transparência e consentimento informado. Pacientes têm direito de saber que IA está sendo usada no seu diagnóstico. Médicos devem explicar: "Usamos um sistema de apoio à decisão baseado em inteligência artificial que analisa seus exames. Esse sistema sugere que você pode ter risco de diabetes. Mas essa é apenas uma ferramenta de apoio. Eu, como seu médico, estou analisando seus resultados no contexto completo da sua saúde e tomarei a decisão final sobre diagnóstico e tratamento." Pacientes devem poder optar por não ter IA envolvida no seu diagnóstico se preferirem abordagem puramente tradicional.

Quarta consideração: responsabilidade legal. Se o modelo comete erro que resulta em dano ao paciente, quem é responsável? O hospital que implementou? Os desenvolvedores que treinaram o modelo? O médico que confiou na predição? Legalmente essa área ainda é nebulosa. Por isso é essencial que o sistema seja claramente posicionado como ferramenta de suporte, não de decisão autônoma. A responsabilidade final recai sobre o médico, que deve ter opção de aceitar ou rejeitar a recomendação do sistema.

---

## CONCLUSÃO

*(Mostrar: Estrutura completa do projeto na pasta fase-1/)*

Em resumo, desenvolvemos um sistema completo e robusto de suporte ao diagnóstico de diabetes seguindo rigorosas práticas de ciência de dados e Machine Learning.

**O que fizemos:**

Escolhemos o Diabetes Health Indicators Dataset do Kaggle com 100 mil pacientes e 31 variáveis clínicas e demográficas. Dataset balanceado com 60% de casos positivos e 40% negativos.

Realizamos análise exploratória profunda. Calculamos estatísticas descritivas completas para 23 variáveis numéricas. Geramos histogramas revelando separação clara entre pacientes com e sem diabetes em features chave como HbA1c e glicose. Construímos matriz de correlação identificando relações entre variáveis. Discutimos correlação versus causalidade, fundamental para interpretação correta dos resultados.

Aplicamos pré-processamento rigoroso. Verificamos qualidade dos dados, zero valores ausentes e duplicatas. Implementamos One-Hot Encoding para 8 variáveis categóricas, expandindo dataset para 43 colunas. Normalizamos variáveis numéricas com StandardScaler garantindo todas na mesma escala. Removemos features problemáticas: diabetes_stage causava data leakage, diabetes_risk_score e outras tinham VIF altíssimo indicando multicolinearidade. Finalizamos com 37 features limpas e informativas.

Treinamos e comparamos três modelos. Regressão Logística como baseline interpretável alcançou 88.5% accuracy e 93.4% ROC-AUC. Random Forest como ensemble robusto alcançou 91% accuracy com precision impressionante de 99.98%. XGBoost como estado da arte alcançou 91% accuracy e 94.1% ROC-AUC, vencendo pela melhor calibração de probabilidades.

Validamos rigorosamente. Cross-validation 5-fold estratificada confirmou ROC-AUC de 94.1% com desvio padrão de apenas 0.3%, indicando estabilidade. Performance no test set independente de 94.13% praticamente idêntica à validação cruzada confirma excelente generalização sem overfitting.

**O diferencial: Explicabilidade com SHAP**

Mais importante, implementamos explicabilidade completa usando SHAP. Summary Plots mostram importância global, confirmando HbA1c como feature dominante seguida por insulin_level, age e bmi, alinhado com conhecimento médico. Waterfall Plots explicam predições individuais passo a passo, mostrando contribuição numérica de cada feature. Dependence Plots revelam relações não-lineares, como thresholds de HbA1c em 5.7% e 6.5% que o modelo aprendeu dos dados sem programação explícita, reproduzindo diretrizes clínicas estabelecidas.

SHAP é essencial porque permite que médicos validem cada decisão do modelo. Se o raciocínio está clinicamente coerente, o médico pode confiar. Se não, o médico rejeita a predição. Essa transparência é o que torna o sistema viável para aplicação médica real.

**Resultados alcançados:**

Accuracy de 91% significa que 9 em cada 10 predições estão corretas. Precision de 99.88% significa que quando o modelo diz diabetes, está certo 999 vezes em 1000. Recall de 85% significa que detectamos 85% dos casos de diabetes, mas infelizmente deixamos passar 15%, os falsos negativos que são críticos e exigiriam ajuste de threshold em implementação real. ROC-AUC de 94.1% é excelente discriminação entre classes, significando que em 94% dos pares aleatórios paciente diabético vs não-diabético, o modelo ranqueia corretamente.

**Entregáveis do projeto:**

*(Mostrar: Pastas e arquivos principais)*

Estrutura completa e organizada. Pasta notebooks/ contém analise-exploratoria.ipynb, preprocessing.ipynb, modelagem.ipynb e treinamento.ipynb, todo o workflow reproduzível. Pasta data/ contém dataset original e arquivos processados X_train, X_test, y_train, y_test prontos para uso. Pasta models/ contém os três modelos treinados em formato pickle, carregáveis para predições. Pasta docs/ contém README.md completo, ROTEIRO_VIDEO.md com estrutura temporal da apresentação, TEXTO_APRESENTACAO.md com esse texto falado, e guias técnicos de interpretação de resultados e recomendações de modelagem.

Todo código versionado com Git, commits organizados documentando evolução do projeto. Tudo reproduzível executando notebooks na ordem indicada. Documentação em português claro, acessível para público técnico e não-técnico.

**Próximos passos para evolução:**

Otimização técnica. Grid Search ou Bayesian Optimization para tuning fino de hiperparâmetros pode ganhar mais 1-2 pontos percentuais de performance. Ajuste de threshold de decisão priorizando Recall para minimizar falsos negativos críticos. Ensemble dos três modelos combinando perspectivas pode melhorar robustez. Adição de features derivadas como razões e interações entre variáveis existentes pode capturar padrões complexos.

Validação clínica. Teste com dados reais de hospital brasileiro para avaliar performance em população local. Validação por médicos endocrinologistas das explicações SHAP para garantir coerência clínica. Estudo prospectivo comparando diagnóstico com e sem auxílio do sistema, medindo impacto real em precisão, velocidade e confiança do médico.

Deploy em produção. Desenvolvimento de API REST para integração com sistemas hospitalares existentes. Interface web intuitiva para médicos visualizarem predições e explicações. Dashboard de monitoramento de performance em tempo real detectando data drift. Sistema de logging completo de todas predições para auditoria e melhoria contínua.

**Mensagem final:**

Este projeto demonstra o potencial da inteligência artificial para auxiliar profissionais de saúde, mas com consciência das limitações e responsabilidades éticas. Desenvolvemos não apenas um modelo de alta performance, mas um sistema explicável, validável e transparente. IA não substitui médicos, amplifica suas capacidades. O médico continua no centro da decisão, usando a IA como ferramenta de suporte baseada em evidências de milhares de casos.

Com validação apropriada e implementação cuidadosa, sistemas como este podem acelerar diagnósticos, reduzir erros, e melhorar resultados de saúde para milhões de pacientes. Mas sempre com humildade reconhecendo limitações, sempre com transparência explicando decisões, e sempre com o humano tendo a palavra final.

*(Mostrar: Tela final com contatos ou logo da FIAP)*

Obrigado pela atenção. Estamos disponíveis para perguntas através da plataforma da FIAP. Este foi o Tech Challenge da Fase 1 desenvolvido por Hiago Marques Rubio e Mylena Ferreira Lacerda.
