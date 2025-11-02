"""
Módulo de Avaliação de Modelos para Predição de Diabetes
======================================================

Este módulo implementa métricas de avaliação abrangentes para modelos
de classificação médica, incluindo análises específicas para o contexto
clínico onde falsos negativos são críticos.

Funcionalidades principais:
- Métricas clássicas (accuracy, precision, recall, F1)
- Métricas médicas (sensibilidade, especificidade, NPV, PPV)
- Curvas ROC e Precision-Recall
- Matriz de confusão interpretada
- Análise de feature importance
- Relatórios clínicos detalhados
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, average_precision_score
)
from sklearn.calibration import calibration_curve
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Plotly é opcional - usar apenas se disponível
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly não disponível. Usando apenas matplotlib/seaborn para visualizações.")

# Configurar estilo dos gráficos
plt.style.use('default')
sns.set_palette("husl")


class MedicalModelEvaluator:
    """
    Classe principal para avaliação de modelos de classificação médica.
    
    Attributes:
        results: Dict com resultados de todos os modelos
        clinical_thresholds: Dict com limiares clínicos
        visualizations: Dict com gráficos gerados
    """
    
    def __init__(self, positive_class_label: str = "Diabetes", 
                 negative_class_label: str = "Não Diabetes"):
        """
        Inicializa o avaliador médico.
        
        Args:
            positive_class_label: Rótulo da classe positiva
            negative_class_label: Rótulo da classe negativa
        """
        self.positive_class_label = positive_class_label
        self.negative_class_label = negative_class_label
        self.results = {}
        self.clinical_thresholds = {}
        self.visualizations = {}
        
    def calculate_medical_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calcula métricas médicas detalhadas.
        
        Args:
            y_true: Labels verdadeiros
            y_pred: Predições do modelo
            y_prob: Probabilidades preditas (opcional)
            
        Returns:
            Dict com todas as métricas médicas
        """
        # Matriz de confusão
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Métricas básicas
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)  # Sensibilidade
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Métricas médicas específicas
        sensitivity = recall  # Sensibilidade (recall)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Especificidade
        
        # Valores preditivos
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value (Precision)
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        
        # Razões de verossimilhança
        lr_positive = sensitivity / (1 - specificity) if specificity != 1 else np.inf
        lr_negative = (1 - sensitivity) / specificity if specificity != 0 else np.inf
        
        # Métricas adicionais
        balanced_accuracy = (sensitivity + specificity) / 2
        
        # Métricas baseadas em probabilidades (se disponível)
        auc_roc = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
        auc_pr = average_precision_score(y_true, y_prob) if y_prob is not None else np.nan
        
        metrics = {
            # Contagens da matriz de confusão
            'true_positives': int(tp),
            'true_negatives': int(tn), 
            'false_positives': int(fp),
            'false_negatives': int(fn),
            
            # Métricas clássicas
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            
            # Métricas médicas
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,  # Positive Predictive Value
            'npv': npv,  # Negative Predictive Value
            
            # Razões de verossimilhança
            'lr_positive': lr_positive,
            'lr_negative': lr_negative,
            
            # Métricas balanceadas
            'balanced_accuracy': balanced_accuracy,
            
            # AUC scores
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
        }
        
        return metrics
    
    def find_optimal_threshold(self, y_true: np.ndarray, y_prob: np.ndarray,
                             metric: str = 'f1') -> Tuple[float, float]:
        """
        Encontra o limiar ótimo para classificação baseado em uma métrica.
        
        Args:
            y_true: Labels verdadeiros
            y_prob: Probabilidades preditas
            metric: Métrica para otimização ('f1', 'youden', 'balanced_accuracy')
            
        Returns:
            Tuple com (melhor_limiar, melhor_score)
        """
        thresholds = np.linspace(0.1, 0.9, 100)
        scores = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_prob >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred_thresh, zero_division=0)
            elif metric == 'youden':
                # Índice de Youden (Sensibilidade + Especificidade - 1)
                metrics = self.calculate_medical_metrics(y_true, y_pred_thresh)
                score = metrics['sensitivity'] + metrics['specificity'] - 1
            elif metric == 'balanced_accuracy':
                metrics = self.calculate_medical_metrics(y_true, y_pred_thresh)
                score = metrics['balanced_accuracy']
            else:
                raise ValueError(f"Métrica '{metric}' não suportada")
            
            scores.append(score)
        
        best_idx = np.argmax(scores)
        best_threshold = thresholds[best_idx]
        best_score = scores[best_idx]
        
        return best_threshold, best_score
    
    def evaluate_single_model(self, model_name: str, y_true: np.ndarray,
                            y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None,
                            dataset_name: str = "test") -> Dict[str, Any]:
        """
        Avalia um modelo específico com métricas médicas completas.
        
        Args:
            model_name: Nome do modelo
            y_true: Labels verdadeiros
            y_pred: Predições do modelo
            y_prob: Probabilidades preditas (opcional)
            dataset_name: Nome do conjunto de dados
            
        Returns:
            Dict com resultados da avaliação
        """
        print(f"\n🔍 Avaliando {model_name} no conjunto {dataset_name}...")
        
        # Calcular métricas médicas
        metrics = self.calculate_medical_metrics(y_true, y_pred, y_prob)
        
        # Encontrar limiar ótimo (se probabilidades disponíveis)
        optimal_thresholds = {}
        if y_prob is not None:
            for metric_name in ['f1', 'youden', 'balanced_accuracy']:
                threshold, score = self.find_optimal_threshold(y_true, y_prob, metric_name)
                optimal_thresholds[f'{metric_name}_threshold'] = threshold
                optimal_thresholds[f'{metric_name}_score'] = score
        
        # Curvas ROC e PR (se probabilidades disponíveis)
        curves_data = {}
        if y_prob is not None:
            # Curva ROC
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
            curves_data['roc'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': roc_thresholds}
            
            # Curva Precision-Recall
            precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_prob)
            curves_data['pr'] = {
                'precision': precision_curve, 
                'recall': recall_curve, 
                'thresholds': pr_thresholds
            }
        
        # Compilar resultados
        evaluation_results = {
            'model_name': model_name,
            'dataset': dataset_name,
            'metrics': metrics,
            'optimal_thresholds': optimal_thresholds,
            'curves_data': curves_data,
            'predictions': y_pred,
            'probabilities': y_prob,
            'true_labels': y_true,
            'timestamp': datetime.now().isoformat()
        }
        
        # Armazenar nos resultados
        if model_name not in self.results:
            self.results[model_name] = {}
        self.results[model_name][dataset_name] = evaluation_results
        
        print(f"✅ Avaliação concluída:")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"   Specificity: {metrics['specificity']:.4f}")
        print(f"   F1-Score: {metrics['f1_score']:.4f}")
        if not np.isnan(metrics['auc_roc']):
            print(f"   AUC-ROC: {metrics['auc_roc']:.4f}")
        
        return evaluation_results
    
    def compare_models(self, models_results: Dict[str, Dict], 
                      dataset_name: str = "test") -> pd.DataFrame:
        """
        Compara múltiplos modelos em uma tabela resumo.
        
        Args:
            models_results: Dict com resultados de múltiplos modelos
            dataset_name: Nome do conjunto de dados para comparação
            
        Returns:
            DataFrame com comparação dos modelos
        """
        print(f"\n📊 Comparando modelos no conjunto {dataset_name}...")
        
        comparison_data = []
        
        for model_name, model_data in models_results.items():
            if dataset_name in model_data:
                metrics = model_data[dataset_name]['metrics']
                
                row = {
                    'Modelo': model_name.replace('_', ' ').title(),
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall/Sensitivity': metrics['recall'],
                    'Specificity': metrics['specificity'],
                    'F1-Score': metrics['f1_score'],
                    'AUC-ROC': metrics['auc_roc'] if not np.isnan(metrics['auc_roc']) else None,
                    'AUC-PR': metrics['auc_pr'] if not np.isnan(metrics['auc_pr']) else None,
                    'PPV': metrics['ppv'],
                    'NPV': metrics['npv'],
                    'Balanced Accuracy': metrics['balanced_accuracy']
                }
                comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Ordenar por F1-Score (métrica balanceada importante para medicina)
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        comparison_df = comparison_df.reset_index(drop=True)
        
        print("✅ Comparação gerada!")
        return comparison_df
    
    def plot_confusion_matrices(self, models_results: Dict[str, Dict],
                              dataset_name: str = "test", 
                              figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plota matrizes de confusão para múltiplos modelos.
        
        Args:
            models_results: Resultados dos modelos
            dataset_name: Nome do conjunto de dados
            figsize: Tamanho da figura
            
        Returns:
            Figura matplotlib
        """
        n_models = len([m for m in models_results.keys() 
                       if dataset_name in models_results[m]])
        
        if n_models == 0:
            print(f"❌ Nenhum resultado encontrado para {dataset_name}")
            return None
        
        # Calcular layout da grade
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        plot_idx = 0
        
        for model_name, model_data in models_results.items():
            if dataset_name not in model_data:
                continue
                
            row = plot_idx // cols
            col = plot_idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            # Obter dados
            y_true = model_data[dataset_name]['true_labels']
            y_pred = model_data[dataset_name]['predictions']
            metrics = model_data[dataset_name]['metrics']
            
            # Calcular matriz de confusão
            cm = confusion_matrix(y_true, y_pred)
            
            # Plot da matriz
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=[self.negative_class_label, self.positive_class_label],
                       yticklabels=[self.negative_class_label, self.positive_class_label])
            
            # Título com métricas principais
            title = f"{model_name.replace('_', ' ').title()}\n"
            title += f"Acc: {metrics['accuracy']:.3f} | "
            title += f"Sens: {metrics['sensitivity']:.3f} | "
            title += f"Spec: {metrics['specificity']:.3f}"
            
            ax.set_title(title, fontsize=11, pad=10)
            ax.set_xlabel('Predição')
            ax.set_ylabel('Verdadeiro')
            
            plot_idx += 1
        
        # Remover subplots vazios
        for idx in range(plot_idx, rows * cols):
            row = idx // cols
            col = idx % cols
            if rows > 1:
                fig.delaxes(axes[row, col])
            else:
                fig.delaxes(axes[col])
        
        plt.tight_layout()
        plt.suptitle(f'Matrizes de Confusão - {dataset_name.title()}', 
                    fontsize=16, y=1.02)
        
        return fig
    
    def plot_roc_curves(self, models_results: Dict[str, Dict],
                       dataset_name: str = "test",
                       figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plota curvas ROC para múltiplos modelos.
        
        Args:
            models_results: Resultados dos modelos
            dataset_name: Nome do conjunto de dados
            figsize: Tamanho da figura
            
        Returns:
            Figura matplotlib
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Linha de referência (classificador aleatório)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Classificador Aleatório')
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(models_results)))
        
        for (model_name, model_data), color in zip(models_results.items(), colors):
            if dataset_name not in model_data:
                continue
                
            curves_data = model_data[dataset_name]['curves_data']
            if 'roc' not in curves_data:
                continue
            
            fpr = curves_data['roc']['fpr']
            tpr = curves_data['roc']['tpr']
            auc_score = model_data[dataset_name]['metrics']['auc_roc']
            
            label = f"{model_name.replace('_', ' ').title()} (AUC = {auc_score:.3f})"
            ax.plot(fpr, tpr, color=color, linewidth=2, label=label)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Taxa de Falsos Positivos (1 - Especificidade)', fontsize=12)
        ax.set_ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)', fontsize=12)
        ax.set_title(f'Curvas ROC - {dataset_name.title()}', fontsize=14, pad=15)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_precision_recall_curves(self, models_results: Dict[str, Dict],
                                    dataset_name: str = "test",
                                    figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plota curvas Precision-Recall para múltiplos modelos.
        
        Args:
            models_results: Resultados dos modelos
            dataset_name: Nome do conjunto de dados
            figsize: Tamanho da figura
            
        Returns:
            Figura matplotlib
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(models_results)))
        
        for (model_name, model_data), color in zip(models_results.items(), colors):
            if dataset_name not in model_data:
                continue
                
            curves_data = model_data[dataset_name]['curves_data']
            if 'pr' not in curves_data:
                continue
            
            precision = curves_data['pr']['precision']
            recall = curves_data['pr']['recall']
            auc_pr = model_data[dataset_name]['metrics']['auc_pr']
            
            label = f"{model_name.replace('_', ' ').title()} (AUC = {auc_pr:.3f})"
            ax.plot(recall, precision, color=color, linewidth=2, label=label)
        
        # Linha de baseline (proporção da classe positiva)
        if models_results:
            first_model = list(models_results.values())[0]
            if dataset_name in first_model:
                y_true = first_model[dataset_name]['true_labels']
                baseline = np.mean(y_true)
                ax.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, 
                          label=f'Baseline (Proporção = {baseline:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall (Sensibilidade)', fontsize=12)
        ax.set_ylabel('Precision (PPV)', fontsize=12)
        ax.set_title(f'Curvas Precision-Recall - {dataset_name.title()}', fontsize=14, pad=15)
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_metrics_comparison(self, comparison_df: pd.DataFrame,
                              metrics: List[str] = None,
                              figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plota gráfico de barras comparando métricas dos modelos.
        
        Args:
            comparison_df: DataFrame com comparação dos modelos
            metrics: Lista de métricas para plotar
            figsize: Tamanho da figura
            
        Returns:
            Figura matplotlib
        """
        if metrics is None:
            metrics = ['Accuracy', 'Recall/Sensitivity', 'Specificity', 'F1-Score', 'AUC-ROC']
        
        # Filtrar métricas disponíveis
        available_metrics = [m for m in metrics if m in comparison_df.columns]
        
        if not available_metrics:
            print("❌ Nenhuma métrica válida encontrada")
            return None
        
        # Preparar dados
        plot_data = comparison_df[['Modelo'] + available_metrics].set_index('Modelo')
        
        # Remover colunas com todos NaN
        plot_data = plot_data.dropna(axis=1, how='all')
        
        # Criar subplot
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Gráfico de barras
        plot_data.plot(kind='bar', ax=ax, width=0.8, alpha=0.8)
        
        ax.set_title('Comparação de Métricas dos Modelos', fontsize=16, pad=20)
        ax.set_xlabel('Modelos', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_ylim([0, 1.05])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotacionar rótulos do eixo x
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    def create_clinical_report(self, model_name: str, dataset_name: str = "test") -> str:
        """
        Cria relatório clínico detalhado para um modelo específico.
        
        Args:
            model_name: Nome do modelo
            dataset_name: Nome do conjunto de dados
            
        Returns:
            String com relatório clínico
        """
        if model_name not in self.results or dataset_name not in self.results[model_name]:
            return f"❌ Resultados não encontrados para {model_name} em {dataset_name}"
        
        data = self.results[model_name][dataset_name]
        metrics = data['metrics']
        
        report = f"""
🏥 RELATÓRIO CLÍNICO - {model_name.upper()}
{'='*60}

📊 RESUMO EXECUTIVO:
O modelo {model_name.replace('_', ' ')} foi avaliado para auxiliar no diagnóstico
de diabetes mellitus. Este relatório apresenta a performance do modelo
sob a perspectiva clínica, focando na segurança do paciente.

🎯 PERFORMANCE DIAGNÓSTICA:

• SENSIBILIDADE (Recall): {metrics['sensitivity']:.1%}
  → {metrics['true_positives']} casos de diabetes identificados corretamente
  → {metrics['false_negatives']} casos de diabetes perdidos (CRÍTICO)

• ESPECIFICIDADE: {metrics['specificity']:.1%} 
  → {metrics['true_negatives']} casos não-diabéticos identificados corretamente
  → {metrics['false_positives']} falsos alarmes

• VALOR PREDITIVO POSITIVO (PPV): {metrics['ppv']:.1%}
  → Quando o modelo prediz diabetes, há {metrics['ppv']:.1%} de chance de estar correto

• VALOR PREDITIVO NEGATIVO (NPV): {metrics['npv']:.1%}
  → Quando o modelo prediz não-diabetes, há {metrics['npv']:.1%} de chance de estar correto

📈 MÉTRICAS TÉCNICAS:
• Acurácia Global: {metrics['accuracy']:.1%}
• F1-Score: {metrics['f1_score']:.3f}
• Acurácia Balanceada: {metrics['balanced_accuracy']:.1%}
"""
        
        if not np.isnan(metrics['auc_roc']):
            report += f"• AUC-ROC: {metrics['auc_roc']:.3f}\n"
        
        report += f"""
⚠️ ANÁLISE DE RISCO CLÍNICO:

• FALSOS NEGATIVOS: {metrics['false_negatives']} pacientes
  → RISCO: Pacientes diabéticos não diagnosticados
  → CONSEQUÊNCIA: Atraso no tratamento, complicações
  → RECOMENDAÇÃO: {"ATENÇÃO - Taxa alta de FN" if metrics['sensitivity'] < 0.8 else "Taxa aceitável de FN"}

• FALSOS POSITIVOS: {metrics['false_positives']} pacientes  
  → RISCO: Pacientes saudáveis com diagnóstico incorreto
  → CONSEQUÊNCIA: Ansiedade, exames desnecessários
  → IMPACTO: {"Moderado" if metrics['specificity'] > 0.8 else "Alto - muitos falsos alarmes"}

🔬 RAZÕES DE VEROSSIMILHANÇA:
• LR+: {metrics['lr_positive']:.2f} 
  → Teste positivo aumenta a probabilidade de diabetes em {metrics['lr_positive']:.1f}x
• LR-: {metrics['lr_negative']:.3f}
  → Teste negativo {'reduz significativamente' if metrics['lr_negative'] < 0.1 else 'reduz moderadamente'} a probabilidade de diabetes

💡 RECOMENDAÇÕES CLÍNICAS:

1. USO RECOMENDADO:
   {"✅ Ferramenta de apoio ao diagnóstico" if metrics['sensitivity'] > 0.8 and metrics['specificity'] > 0.8 else "⚠️ Usar com cautela - performance limitada"}

2. SUPERVISÃO MÉDICA:
   🔸 SEMPRE requer confirmação por profissional médico
   🔸 NÃO substitui avaliação clínica completa
   🔸 Considerar fatores não incluídos no modelo

3. APLICAÇÃO PRÁTICA:
   {"🔸 Adequado para triagem inicial" if metrics['sensitivity'] > 0.85 else "🔸 NÃO recomendado para triagem"}
   {"🔸 Pode reduzir encaminhamentos desnecessários" if metrics['specificity'] > 0.8 else "🔸 Muitos falsos positivos - cuidado com encaminhamentos"}

⚖️ CONSIDERAÇÕES ÉTICAS:
• Transparência: Explicar limitações aos pacientes
• Equidade: Verificar viés em diferentes populações  
• Responsabilidade: Médico mantém decisão final

✅ CONCLUSÃO:
O modelo apresenta {'performance adequada' if metrics['f1_score'] > 0.75 else 'performance limitada'} 
para uso clínico como ferramenta de apoio. {'Recomenda-se implementação com supervisão médica adequada.' if metrics['sensitivity'] > 0.8 else 'Recomenda-se melhoria antes da implementação clínica.'}
"""
        
        return report
    
    def save_evaluation_results(self, output_dir: str) -> None:
        """
        Salva todos os resultados e visualizações.
        
        Args:
            output_dir: Diretório para salvar os resultados
        """
        print(f"\n💾 Salvando resultados de avaliação em {output_dir}...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Salvar resultados em formato pickle
        import pickle
        results_file = os.path.join(output_dir, "evaluation_results.pkl")
        with open(results_file, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"✅ Resultados salvos em {results_file}")
        
        # Salvar gráficos se existirem
        for viz_name, fig in self.visualizations.items():
            if fig is not None:
                fig_path = os.path.join(output_dir, f"{viz_name}.png")
                fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                print(f"✅ Gráfico {viz_name} salvo em {fig_path}")
                
        print("✅ Todos os resultados salvos!")


def quick_evaluation_pipeline(models_results: Dict, y_test: np.ndarray,
                            predictions_dict: Dict, probabilities_dict: Dict = None) -> MedicalModelEvaluator:
    """
    Pipeline rápido para avaliação de múltiplos modelos.
    
    Args:
        models_results: Resultados do treinamento
        y_test: Labels de teste
        predictions_dict: Dict com predições de cada modelo
        probabilities_dict: Dict com probabilidades de cada modelo (opcional)
        
    Returns:
        Instância do MedicalModelEvaluator com resultados
    """
    evaluator = MedicalModelEvaluator()
    
    for model_name in models_results.keys():
        if model_name in predictions_dict:
            y_pred = predictions_dict[model_name]
            y_prob = probabilities_dict.get(model_name) if probabilities_dict else None
            
            evaluator.evaluate_single_model(
                model_name, y_test, y_pred, y_prob, "test"
            )
    
    return evaluator


if __name__ == "__main__":
    # Exemplo de uso com dados simulados
    print("🧪 Testando MedicalModelEvaluator...")
    
    np.random.seed(42)
    n_samples = 500
    
    # Simular dados de teste
    y_true = np.random.binomial(1, 0.3, n_samples)
    y_pred_model1 = np.random.binomial(1, 0.4, n_samples)  # Modelo com mais FP
    y_pred_model2 = (np.random.randn(n_samples) + y_true * 2 > 1).astype(int)  # Modelo melhor
    y_prob_model2 = 1 / (1 + np.exp(-(np.random.randn(n_samples) + y_true * 2)))
    
    # Testar avaliador
    evaluator = MedicalModelEvaluator()
    
    # Avaliar modelos
    evaluator.evaluate_single_model("model_1", y_true, y_pred_model1)
    evaluator.evaluate_single_model("model_2", y_true, y_pred_model2, y_prob_model2)
    
    # Comparar modelos
    comparison = evaluator.compare_models(evaluator.results, "test")
    print("\n📊 Comparação dos modelos:")
    print(comparison)
    
    # Relatório clínico
    report = evaluator.create_clinical_report("model_2")
    print(report)
    
    print("\n✅ Teste do avaliador concluído!")