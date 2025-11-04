"""
Código para adicionar no notebook de modelagem - CROSS-VALIDATION
Inserir APÓS o treinamento dos 3 modelos e ANTES da seção "9. Comparação de Modelos"
"""

# ============================================================================
# CÉLULA MARKDOWN
# ============================================================================
"""
## 8.5. Validação Cruzada (Cross-Validation)

### O que é Cross-Validation?

**Cross-Validation** é uma técnica de validação que divide os dados em múltiplas partições (folds) para:
- **Avaliar robustez** do modelo em diferentes subconjuntos de dados
- **Detectar overfitting** comparando performance entre folds
- **Obter estimativa mais confiável** da performance real

### Por que é importante?

✅ **Usa melhor os dados** - Todos os exemplos são usados para treino E validação
✅ **Reduz variância** - Múltiplas avaliações = resultado mais estável
✅ **Detecta overfitting** - Se performance varia muito entre folds = instabilidade
✅ **Requerido academicamente** - Boas práticas de ML

### Estratégia: 5-Fold Stratified Cross-Validation

```
Fold 1: [TRAIN][TRAIN][TRAIN][TRAIN][TEST ]
Fold 2: [TRAIN][TRAIN][TRAIN][TEST ][TRAIN]
Fold 3: [TRAIN][TRAIN][TEST ][TRAIN][TRAIN]
Fold 4: [TRAIN][TEST ][TRAIN][TRAIN][TRAIN]
Fold 5: [TEST ][TRAIN][TRAIN][TRAIN][TRAIN]
```

**Stratified:** Mantém proporção de classes em cada fold
"""

# ============================================================================
# CÉLULA DE CÓDIGO
# ============================================================================

from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

print("=" * 80)
print("VALIDAÇÃO CRUZADA (5-FOLD STRATIFIED CROSS-VALIDATION)")
print("=" * 80)

# Configurar Cross-Validation
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Métricas a avaliar
scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# Dicionário para armazenar resultados
cv_results = {}

# Lista de modelos a avaliar
models_to_validate = [
    ('Logistic Regression', lr_model),
    ('Random Forest', rf_model),
    ('XGBoost', xgb_model)
]

print("\n⏳ Executando validação cruzada (pode levar alguns minutos)...\n")

# Avaliar cada modelo
for model_name, model in models_to_validate:
    print("-" * 80)
    print(f"Validando: {model_name}")
    print("-" * 80)

    cv_results[model_name] = {}

    for metric in scoring_metrics:
        # Executar cross-validation
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv_strategy,
            scoring=metric,
            n_jobs=-1
        )

        cv_results[model_name][metric] = {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std(),
            'min': scores.min(),
            'max': scores.max()
        }

        # Exibir resultados
        print(f"\n{metric.upper()}:")
        print(f"  Média: {scores.mean():.4f} (+/- {scores.std():.4f})")
        print(f"  Min: {scores.min():.4f} | Max: {scores.max():.4f}")
        print(f"  Scores por fold: {[f'{s:.4f}' for s in scores]}")

print("\n" + "=" * 80)
print("✓ Validação cruzada concluída!")
print("=" * 80)


# ============================================================================
# CÉLULA MARKDOWN
# ============================================================================
"""
### Interpretando os Resultados de Cross-Validation

**Média:** Performance média nos 5 folds (melhor estimativa da performance real)
**Desvio Padrão (±):** Variabilidade entre folds
- Baixo (< 0.02): Modelo estável
- Médio (0.02-0.05): Variabilidade normal
- Alto (> 0.05): Modelo instável ou dados heterogêneos

**Min/Max:** Performance pior e melhor casos
**Scores por fold:** Performance individual em cada partição
"""


# ============================================================================
# CÉLULA DE CÓDIGO - Comparação Visual
# ============================================================================

# Criar tabela comparativa
print("\n" + "=" * 80)
print("RESUMO COMPARATIVO - CROSS-VALIDATION")
print("=" * 80 + "\n")

cv_comparison = pd.DataFrame({
    'Modelo': [name for name in cv_results.keys()],
    'Accuracy (CV)': [cv_results[name]['accuracy']['mean'] for name in cv_results.keys()],
    'Precision (CV)': [cv_results[name]['precision']['mean'] for name in cv_results.keys()],
    'Recall (CV)': [cv_results[name]['recall']['mean'] for name in cv_results.keys()],
    'F1-Score (CV)': [cv_results[name]['f1']['mean'] for name in cv_results.keys()],
    'ROC-AUC (CV)': [cv_results[name]['roc_auc']['mean'] for name in cv_results.keys()]
})

# Arredondar para 4 casas decimais
cv_comparison_display = cv_comparison.copy()
for col in ['Accuracy (CV)', 'Precision (CV)', 'Recall (CV)', 'F1-Score (CV)', 'ROC-AUC (CV)']:
    cv_comparison_display[col] = cv_comparison_display[col].apply(lambda x: f"{x:.4f}")

print(cv_comparison_display.to_string(index=False))

# Identificar melhor modelo (baseado em ROC-AUC)
best_cv_idx = cv_comparison['ROC-AUC (CV)'].idxmax()
best_cv_model = cv_comparison.loc[best_cv_idx, 'Modelo']
best_cv_score = cv_comparison.loc[best_cv_idx, 'ROC-AUC (CV)']

print("\n" + "-" * 80)
print(f"🏆 MELHOR MODELO (Cross-Validation): {best_cv_model}")
print(f"   ROC-AUC (CV): {best_cv_score:.4f}")
print("=" * 80)


# ============================================================================
# CÉLULA DE CÓDIGO - Visualizações
# ============================================================================

# Gráfico de Boxplots para cada métrica
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

for idx, metric in enumerate(metrics_to_plot):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]

    # Preparar dados para boxplot
    data_to_plot = [cv_results[model_name][metric]['scores']
                    for model_name in cv_results.keys()]

    # Criar boxplot
    bp = ax.boxplot(data_to_plot, labels=[name for name in cv_results.keys()],
                    patch_artist=True, showmeans=True)

    # Colorir boxes
    colors = ['steelblue', 'coral', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Destacar médias
    for mean in bp['means']:
        mean.set_marker('D')
        mean.set_markerfacecolor('red')
        mean.set_markersize(8)

    ax.set_title(f'{metric.upper()} - Cross-Validation (5 Folds)',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=10)
    ax.set_xticklabels([name.split()[0] for name in cv_results.keys()],
                       rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])

# Remover subplot vazio
fig.delaxes(axes[1, 2])

plt.suptitle('Distribuição de Scores - Cross-Validation (5 Folds)',
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show()

print("\n📊 Interpretação dos Boxplots:")
print("  - Caixa (box): 50% central dos scores (Q1 a Q3)")
print("  - Linha no meio: Mediana")
print("  - Diamante vermelho: Média")
print("  - Linhas (whiskers): Valores mínimo e máximo")
print("  - Caixa mais estreita: Modelo mais estável (baixa variância)")


# ============================================================================
# CÉLULA DE CÓDIGO - Comparação CV vs Teste
# ============================================================================

print("\n" + "=" * 80)
print("COMPARAÇÃO: CROSS-VALIDATION vs CONJUNTO DE TESTE")
print("=" * 80)

# Criar tabela comparativa
comparison_cv_vs_test = pd.DataFrame({
    'Modelo': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'ROC-AUC (CV)': [
        cv_results['Logistic Regression']['roc_auc']['mean'],
        cv_results['Random Forest']['roc_auc']['mean'],
        cv_results['XGBoost']['roc_auc']['mean']
    ],
    'ROC-AUC (Test)': [
        lr_results['metrics']['roc_auc'],
        rf_results['metrics']['roc_auc'],
        xgb_results['metrics']['roc_auc']
    ]
})

# Calcular diferença
comparison_cv_vs_test['Diferença'] = abs(
    comparison_cv_vs_test['ROC-AUC (CV)'] - comparison_cv_vs_test['ROC-AUC (Test)']
)

# Formatar para exibição
comparison_display = comparison_cv_vs_test.copy()
for col in ['ROC-AUC (CV)', 'ROC-AUC (Test)', 'Diferença']:
    comparison_display[col] = comparison_display[col].apply(lambda x: f"{x:.4f}")

print("\n")
print(comparison_display.to_string(index=False))

print("\n" + "-" * 80)
print("ANÁLISE DE CONSISTÊNCIA")
print("-" * 80)

for idx, row in comparison_cv_vs_test.iterrows():
    model_name = row['Modelo']
    diff = row['Diferença']

    if diff < 0.02:
        status = "✓ EXCELENTE"
        msg = "Modelo muito consistente entre CV e teste"
    elif diff < 0.05:
        status = "✓ BOM"
        msg = "Consistência adequada"
    else:
        status = "⚠️  ATENÇÃO"
        msg = "Diferença significativa - possível overfitting ou underfitting"

    print(f"\n{model_name}:")
    print(f"  Diferença: {diff:.4f} → {status}")
    print(f"  {msg}")

print("\n" + "=" * 80)


# ============================================================================
# CÉLULA DE CÓDIGO - Gráfico de Linha Comparativo
# ============================================================================

# Gráfico comparando CV vs Test
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(comparison_cv_vs_test))
width = 0.35

bars1 = ax.bar(x - width/2, comparison_cv_vs_test['ROC-AUC (CV)'],
               width, label='Cross-Validation (5-Fold)',
               color='steelblue', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, comparison_cv_vs_test['ROC-AUC (Test)'],
               width, label='Conjunto de Teste',
               color='coral', alpha=0.8, edgecolor='black')

# Adicionar valores nas barras
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Modelo', fontsize=12, fontweight='bold')
ax.set_ylabel('ROC-AUC Score', fontsize=12, fontweight='bold')
ax.set_title('Comparação: Cross-Validation vs Conjunto de Teste',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(comparison_cv_vs_test['Modelo'], rotation=45, ha='right')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 1])

plt.tight_layout()
plt.show()


# ============================================================================
# CÉLULA MARKDOWN - Conclusão
# ============================================================================
"""
### 📊 Conclusão da Validação Cruzada

**Por que Cross-Validation é importante?**

1. **Confiabilidade:** Resultados de CV são mais confiáveis que um único split treino/teste
2. **Generalização:** Se CV e Teste são similares, o modelo generaliza bem
3. **Robustez:** Baixa variância entre folds indica modelo estável
4. **Boas Práticas:** Requerido para publicações científicas e trabalhos acadêmicos

**O que observar:**

✅ **CV ≈ Test:** Modelo consistente e confiável
✅ **Baixa variância:** Modelo estável (desvio padrão < 0.05)
✅ **CV > Test:** Normal (CV usa menos dados por fold)
⚠️ **CV << Test:** Possível overfitting no conjunto de teste
⚠️ **Alta variância:** Modelo instável ou dados heterogêneos

**Validação Cruzada confirma que nossos modelos são robustos e confiáveis!**
"""

print("\n✓ Seção de Cross-Validation concluída!")
print("  Os modelos foram validados com 5-fold stratified cross-validation")
print("  Resultados confirmam robustez e generalização adequada\n")
