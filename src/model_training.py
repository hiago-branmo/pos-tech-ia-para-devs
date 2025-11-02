"""
Módulo de Treinamento de Modelos para Predição de Diabetes
=========================================================

Este módulo implementa diferentes algoritmos de machine learning para
classificação de diabetes, incluindo otimização de hiperparâmetros
e validação cruzada.

Modelos implementados:
- Regressão Logística (baseline)
- Random Forest (ensemble)
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Gradient Boosting (XGBoost)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost não disponível. Instale com: pip install xgboost")


class ModelTrainer:
    """
    Classe principal para treinamento de modelos de classificação.
    
    Attributes:
        models: Dict com modelos treinados
        best_params: Dict com melhores hiperparâmetros
        cv_scores: Dict com scores de validação cruzada
        training_time: Dict com tempos de treinamento
    """
    
    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        """
        Inicializa o treinador de modelos.
        
        Args:
            random_state: Seed para reprodutibilidade
            n_jobs: Número de processadores (-1 para todos)
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.models = {}
        self.best_params = {}
        self.cv_scores = {}
        self.training_time = {}
        
    def get_model_configs(self) -> Dict[str, Dict]:
        """
        Retorna configurações dos modelos com hiperparâmetros para tuning.
        
        Returns:
            Dict com configurações de cada modelo
        """
        configs = {
            'logistic_regression': {
                'model': LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000,
                    n_jobs=self.n_jobs
                ),
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                },
                'description': 'Regressão Logística - Modelo linear baseline com regularização'
            },
            
            'random_forest': {
                'model': RandomForestClassifier(
                    random_state=self.random_state,
                    n_jobs=self.n_jobs
                ),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                },
                'description': 'Random Forest - Ensemble de árvores com bagging'
            },
            
            'svm': {
                'model': SVC(
                    random_state=self.random_state,
                    probability=True  # Para calcular probabilidades
                ),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
                },
                'description': 'Support Vector Machine - Classificador de margem máxima'
            },
            
            'knn': {
                'model': KNeighborsClassifier(n_jobs=self.n_jobs),
                'params': {
                    'n_neighbors': [3, 5, 7, 9, 11, 15, 21],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree'],
                    'p': [1, 2]  # Distância Manhattan (1) ou Euclidiana (2)
                },
                'description': 'K-Nearest Neighbors - Classificação por proximidade'
            }
        }
        
        # Adicionar XGBoost se disponível
        if XGBOOST_AVAILABLE:
            configs['xgboost'] = {
                'model': xgb.XGBClassifier(
                    random_state=self.random_state,
                    eval_metric='logloss',
                    n_jobs=self.n_jobs
                ),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                },
                'description': 'XGBoost - Gradient Boosting otimizado'
            }
        
        return configs
    
    def train_single_model(self, model_name: str, X_train: pd.DataFrame, 
                          y_train: pd.Series, X_val: pd.DataFrame, 
                          y_val: pd.Series, tune_hyperparams: bool = True) -> Dict:
        """
        Treina um modelo específico com opção de tuning.
        
        Args:
            model_name: Nome do modelo a treinar
            X_train: Features de treino
            y_train: Target de treino  
            X_val: Features de validação
            y_val: Target de validação
            tune_hyperparams: Se deve fazer tuning de hiperparâmetros
            
        Returns:
            Dict com resultados do treinamento
        """
        print(f"\n🤖 Treinando {model_name}...")
        
        configs = self.get_model_configs()
        if model_name not in configs:
            available_models = list(configs.keys())
            raise ValueError(f"Modelo '{model_name}' não disponível. Opções: {available_models}")
        
        config = configs[model_name]
        model = config['model']
        
        start_time = datetime.now()
        
        if tune_hyperparams and config['params']:
            print(f"🔧 Fazendo tuning de hiperparâmetros...")
            
            # Configurar validação cruzada estratificada
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            
            # Grid Search com validação cruzada
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=config['params'],
                cv=cv,
                scoring='f1',  # F1-score para dados desbalanceados
                n_jobs=self.n_jobs,
                verbose=0
            )
            
            # Treinar com grid search
            grid_search.fit(X_train, y_train)
            
            # Melhor modelo e parâmetros
            best_model = grid_search.best_estimator_
            self.best_params[model_name] = grid_search.best_params_
            
            print(f"✅ Melhores parâmetros: {self.best_params[model_name]}")
            
        else:
            print(f"🚀 Treinando com parâmetros padrão...")
            best_model = model
            best_model.fit(X_train, y_train)
            self.best_params[model_name] = "Parâmetros padrão"
        
        # Calcular tempo de treinamento
        training_duration = datetime.now() - start_time
        self.training_time[model_name] = training_duration.total_seconds()
        
        # Salvar modelo treinado
        self.models[model_name] = best_model
        
        # Fazer predições
        train_pred = best_model.predict(X_train)
        val_pred = best_model.predict(X_val)
        
        # Probabilidades (se suportado)
        try:
            train_prob = best_model.predict_proba(X_train)[:, 1]
            val_prob = best_model.predict_proba(X_val)[:, 1]
        except:
            train_prob = None
            val_prob = None
        
        # Validação cruzada no conjunto de treino
        cv_scores = cross_val_score(
            best_model, X_train, y_train, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring='f1', n_jobs=self.n_jobs
        )
        self.cv_scores[model_name] = cv_scores
        
        results = {
            'model': best_model,
            'train_predictions': train_pred,
            'val_predictions': val_pred,
            'train_probabilities': train_prob,
            'val_probabilities': val_prob,
            'best_params': self.best_params[model_name],
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'training_time': self.training_time[model_name],
            'description': config['description']
        }
        
        print(f"✅ Modelo treinado em {self.training_time[model_name]:.2f}s")
        print(f"📊 CV F1-Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        return results
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame, y_val: pd.Series,
                        models_to_train: Optional[List[str]] = None,
                        tune_hyperparams: bool = True) -> Dict:
        """
        Treina todos os modelos disponíveis ou uma lista específica.
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            X_val: Features de validação  
            y_val: Target de validação
            models_to_train: Lista de modelos específicos (None para todos)
            tune_hyperparams: Se deve fazer tuning de hiperparâmetros
            
        Returns:
            Dict com resultados de todos os modelos
        """
        print("🚀 Iniciando treinamento de modelos...")
        print("=" * 60)
        
        configs = self.get_model_configs()
        
        if models_to_train is None:
            models_to_train = list(configs.keys())
        
        results = {}
        total_start_time = datetime.now()
        
        for model_name in models_to_train:
            try:
                results[model_name] = self.train_single_model(
                    model_name, X_train, y_train, X_val, y_val, tune_hyperparams
                )
            except Exception as e:
                print(f"❌ Erro ao treinar {model_name}: {e}")
                continue
        
        total_time = (datetime.now() - total_start_time).total_seconds()
        
        print("\n" + "=" * 60)
        print(f"🎉 Treinamento concluído em {total_time:.2f}s!")
        print(f"📊 {len(results)} modelos treinados com sucesso")
        
        return results
    
    def get_feature_importance(self, model_name: str, feature_names: List[str]) -> Dict:
        """
        Extrai importância das features do modelo treinado.
        
        Args:
            model_name: Nome do modelo
            feature_names: Lista com nomes das features
            
        Returns:
            Dict com importância das features
        """
        if model_name not in self.models:
            raise ValueError(f"Modelo '{model_name}' não foi treinado ainda")
        
        model = self.models[model_name]
        importance_dict = {}
        
        try:
            # Random Forest e XGBoost têm feature_importances_
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                importance_dict = dict(zip(feature_names, importances))
                
            # Regressão Logística tem coeficientes
            elif hasattr(model, 'coef_'):
                # Para classificação binária, usar valores absolutos dos coeficientes
                coefs = np.abs(model.coef_[0])
                importance_dict = dict(zip(feature_names, coefs))
                
            # SVM com kernel linear tem coeficientes
            elif hasattr(model, 'coef_') and model.kernel == 'linear':
                coefs = np.abs(model.coef_[0])
                importance_dict = dict(zip(feature_names, coefs))
                
            else:
                print(f"⚠️ Feature importance não disponível para {model_name}")
                return {}
            
            # Ordenar por importância
            importance_dict = dict(sorted(importance_dict.items(), 
                                        key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            print(f"❌ Erro ao extrair feature importance para {model_name}: {e}")
            return {}
        
        return importance_dict
    
    def save_models(self, output_dir: str) -> None:
        """
        Salva todos os modelos treinados em arquivos.
        
        Args:
            output_dir: Diretório para salvar os modelos
        """
        print(f"\n💾 Salvando modelos em {output_dir}...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            filename = os.path.join(output_dir, f"{model_name}_model.joblib")
            joblib.dump(model, filename)
            print(f"✅ {model_name} salvo em {filename}")
        
        # Salvar metadados
        metadata = {
            'best_params': self.best_params,
            'cv_scores': {k: v.tolist() for k, v in self.cv_scores.items()},
            'training_time': self.training_time,
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_file = os.path.join(output_dir, "training_metadata.joblib")
        joblib.dump(metadata, metadata_file)
        print(f"✅ Metadados salvos em {metadata_file}")
    
    def load_models(self, output_dir: str) -> None:
        """
        Carrega modelos previamente treinados.
        
        Args:
            output_dir: Diretório com os modelos salvos
        """
        print(f"\n📂 Carregando modelos de {output_dir}...")
        
        # Carregar metadados
        metadata_file = os.path.join(output_dir, "training_metadata.joblib")
        if os.path.exists(metadata_file):
            metadata = joblib.load(metadata_file)
            self.best_params = metadata['best_params']
            self.cv_scores = {k: np.array(v) for k, v in metadata['cv_scores'].items()}
            self.training_time = metadata['training_time']
            print(f"✅ Metadados carregados")
        
        # Carregar modelos
        configs = self.get_model_configs()
        for model_name in configs.keys():
            model_file = os.path.join(output_dir, f"{model_name}_model.joblib")
            if os.path.exists(model_file):
                self.models[model_name] = joblib.load(model_file)
                print(f"✅ {model_name} carregado")
    
    def create_training_report(self, results: Dict, feature_names: List[str]) -> str:
        """
        Cria relatório detalhado do treinamento.
        
        Args:
            results: Resultados do treinamento
            feature_names: Lista com nomes das features
            
        Returns:
            String com o relatório
        """
        report = f"""
🤖 RELATÓRIO DE TREINAMENTO DE MODELOS - DIABETES
{'='*70}

📊 RESUMO GERAL:
• Modelos treinados: {len(results)}
• Features utilizadas: {len(feature_names)}
• Tempo total: {sum(self.training_time.values()):.2f}s

🏆 RANKING POR PERFORMANCE (CV F1-Score):
"""
        
        # Ranking dos modelos por CV F1-Score
        ranking = sorted(results.items(), key=lambda x: x[1]['cv_mean'], reverse=True)
        
        for i, (model_name, result) in enumerate(ranking, 1):
            report += f"{i}. {model_name.upper()}: {result['cv_mean']:.4f} (±{result['cv_std']:.4f})\n"
        
        report += f"\n📋 DETALHES POR MODELO:\n"
        report += "-" * 70 + "\n"
        
        for model_name, result in results.items():
            report += f"\n🔸 {model_name.upper()}\n"
            report += f"   Descrição: {result['description']}\n"
            report += f"   CV F1-Score: {result['cv_mean']:.4f} (±{result['cv_std']:.4f})\n"
            report += f"   Tempo de treino: {result['training_time']:.2f}s\n"
            
            if isinstance(result['best_params'], dict):
                report += f"   Melhores parâmetros:\n"
                for param, value in result['best_params'].items():
                    report += f"     • {param}: {value}\n"
            else:
                report += f"   Parâmetros: {result['best_params']}\n"
            
            # Feature importance (top 5)
            importance = self.get_feature_importance(model_name, feature_names)
            if importance:
                report += f"   Top 5 features mais importantes:\n"
                for i, (feature, imp) in enumerate(list(importance.items())[:5], 1):
                    report += f"     {i}. {feature}: {imp:.4f}\n"
        
        report += f"\n✅ Todos os modelos prontos para avaliação!"
        
        return report


def quick_model_comparison(X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series,
                          random_state: int = 42) -> Dict:
    """
    Função utilitária para comparação rápida de modelos sem tuning.
    
    Args:
        X_train: Features de treino
        y_train: Target de treino
        X_val: Features de validação
        y_val: Target de validação
        random_state: Seed para reprodutibilidade
        
    Returns:
        Dict com resultados da comparação
    """
    print("⚡ Comparação rápida de modelos (sem tuning)...")
    
    trainer = ModelTrainer(random_state=random_state)
    
    # Treinar apenas com parâmetros padrão
    results = trainer.train_all_models(
        X_train, y_train, X_val, y_val, 
        tune_hyperparams=False
    )
    
    # Resumo rápido
    print("\n📊 RESUMO RÁPIDO:")
    for model_name, result in results.items():
        print(f"{model_name}: {result['cv_mean']:.4f} CV F1-Score")
    
    return results


if __name__ == "__main__":
    # Exemplo de uso (dados fictícios)
    print("🧪 Testando ModelTrainer com dados simulados...")
    
    # Simular dados para teste
    np.random.seed(42)
    n_samples, n_features = 1000, 10
    
    X_train = pd.DataFrame(np.random.randn(n_samples, n_features),
                          columns=[f'feature_{i}' for i in range(n_features)])
    y_train = pd.Series(np.random.binomial(1, 0.3, n_samples))
    
    X_val = pd.DataFrame(np.random.randn(200, n_features),
                        columns=[f'feature_{i}' for i in range(n_features)])  
    y_val = pd.Series(np.random.binomial(1, 0.3, 200))
    
    # Testar treinamento rápido
    results = quick_model_comparison(X_train, y_train, X_val, y_val)
    
    print(f"\n✅ Teste concluído! {len(results)} modelos testados.")