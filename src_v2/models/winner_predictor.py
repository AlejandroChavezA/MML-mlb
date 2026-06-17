import warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class WinnerPredictor:
    MODEL_NAMES = ["random_forest", "gradient_boosting", "logistic_regression"]

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models: Dict[str, object] = {}
        self.feature_cols: List[str] = []
        self.performance: Dict[str, Dict] = {}
        self.best_model: str = "gradient_boosting"

    def train(self, features_df: pd.DataFrame, targets: pd.Series,
               X_test: pd.DataFrame = None, y_test: pd.Series = None) -> bool:
        print("\n" + "=" * 50)
        print("  ENTRENANDO WINNER PREDICTOR (3 MODELOS CALIBRADOS)")
        print("=" * 50)

        self.feature_cols = features_df.columns.tolist()

        if X_test is not None and y_test is not None:
            X_train, y_train = features_df, targets
            print(f"  Train: {len(X_train)}, Test externo: {len(X_test)}")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                features_df, targets, test_size=0.2, random_state=42
            )
            print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

        cal_configs = {
            "random_forest": CalibratedClassifierCV(
                estimator=RandomForestClassifier(
                    n_estimators=200, max_depth=10, min_samples_split=20,
                    min_samples_leaf=10, random_state=42, class_weight="balanced", n_jobs=-1,
                ),
                method="sigmoid", cv=5,
            ),
            "gradient_boosting": CalibratedClassifierCV(
                estimator=GradientBoostingClassifier(
                    n_estimators=80, max_depth=4, learning_rate=0.05,
                    min_samples_split=20, min_samples_leaf=10,
                    random_state=42, subsample=0.7, max_features="sqrt",
                ),
                method="sigmoid", cv=5,
            ),
            "logistic_regression": CalibratedClassifierCV(
                estimator=Pipeline([
                    ("scaler", StandardScaler()),
                    ("lr", LogisticRegression(
                        random_state=42, class_weight="balanced",
                        max_iter=2000, solver="lbfgs",
                    )),
                ]),
                method="sigmoid", cv=5,
            ),
        }

        for name in self.MODEL_NAMES:
            model = cal_configs[name]
            print(f"\n  >> {name} (calibrated)")

            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            train_proba = model.predict_proba(X_train)
            test_proba = model.predict_proba(X_test)

            train_acc = accuracy_score(y_train, train_pred)
            test_acc = accuracy_score(y_test, test_pred)

            try:
                train_auc = roc_auc_score(y_train, train_proba[:, 1])
                test_auc = roc_auc_score(y_test, test_proba[:, 1])
            except Exception:
                train_auc = test_auc = 0.5

            cv = cross_val_score(model, X_train, y_train, cv=5)

            print(f"    Train Acc: {train_acc:.3f}  Test Acc: {test_acc:.3f}")
            print(f"    Train AUC: {train_auc:.3f}  Test AUC:  {test_auc:.3f}")
            print(f"    CV:        {cv.mean():.3f} +- {cv.std():.3f}")

            self.models[name] = model
            self.performance[name] = {
                "train_acc": train_acc,
                "test_acc": test_acc,
                "train_auc": train_auc,
                "test_auc": test_auc,
                "cv_mean": cv.mean(),
                "cv_std": cv.std(),
            }

        self.best_model = max(
            self.MODEL_NAMES, key=lambda n: self.performance[n]["test_acc"]
        )
        print(f"\n  Mejor modelo: {self.best_model} ({self.performance[self.best_model]['test_acc']:.3f})")
        self._save()
        return True

    def train_final(self, features_df: pd.DataFrame, targets: pd.Series) -> bool:
        """Entrenar todos los modelos con el 100% de los datos y guardar."""
        print("\n" + "=" * 50)
        print("  ENTRENANDO MODELO FINAL (100% DE DATOS)")
        print("=" * 50)

        self.feature_cols = features_df.columns.tolist()

        cal_configs = {
            "random_forest": CalibratedClassifierCV(
                estimator=RandomForestClassifier(
                    n_estimators=200, max_depth=10, min_samples_split=20,
                    min_samples_leaf=10, random_state=42, class_weight="balanced", n_jobs=-1,
                ),
                method="sigmoid", cv=5,
            ),
            "gradient_boosting": CalibratedClassifierCV(
                estimator=GradientBoostingClassifier(
                    n_estimators=80, max_depth=4, learning_rate=0.05,
                    min_samples_split=20, min_samples_leaf=10,
                    random_state=42, subsample=0.7, max_features="sqrt",
                ),
                method="sigmoid", cv=5,
            ),
            "logistic_regression": CalibratedClassifierCV(
                estimator=Pipeline([
                    ("scaler", StandardScaler()),
                    ("lr", LogisticRegression(
                        random_state=42, class_weight="balanced",
                        max_iter=2000, solver="lbfgs",
                    )),
                ]),
                method="sigmoid", cv=5,
            ),
        }

        for name in self.MODEL_NAMES:
            model = cal_configs[name]
            print(f"  >> {name}")
            model.fit(features_df, targets)
            self.models[name] = model

        self.best_model = max(
            self.MODEL_NAMES,
            key=lambda n: self.performance.get(n, {}).get("test_acc", 0)
        )
        self._save()
        print(f"  Modelo final guardado ({len(features_df)} muestras, 100%)")
        return True

    def _save(self):
        self.models_dir.mkdir(parents=True, exist_ok=True)
        for name in self.MODEL_NAMES:
            with open(self.models_dir / f"winner_{name}.pkl", "wb") as f:
                pickle.dump(self.models[name], f)
        with open(self.models_dir / "winner_features.pkl", "wb") as f:
            pickle.dump(self.feature_cols, f)
        with open(self.models_dir / "winner_performance.pkl", "wb") as f:
            pickle.dump({"performance": self.performance, "best": self.best_model}, f)
        print(f"\n  Guardado en {self.models_dir}")

    def load(self) -> bool:
        try:
            for name in self.MODEL_NAMES:
                path = self.models_dir / f"winner_{name}.pkl"
                if path.exists():
                    with open(path, "rb") as f:
                        self.models[name] = pickle.load(f)
            with open(self.models_dir / "winner_features.pkl", "rb") as f:
                self.feature_cols = pickle.load(f)
            perf_path = self.models_dir / "winner_performance.pkl"
            if perf_path.exists():
                with open(perf_path, "rb") as f:
                    data = pickle.load(f)
                    self.performance = data.get("performance", {})
                    self.best_model = data.get("best", "gradient_boosting")
            print(f"  Modelos cargados: {list(self.models.keys())}, best={self.best_model}")
            return True
        except Exception as e:
            print(f"  Error cargando modelos: {e}")
            return False

    def predict(self, home_team: str, away_team: str,
                match_date, feature_engineer, model_name: str = None) -> Dict:
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        if not self.models:
            return {"error": "Modelos no cargados"}
        if feature_engineer is None:
            return {"error": "Feature engineer no proveido"}

        if model_name is None:
            model_name = self.best_model
        if model_name not in self.models:
            return {"error": f"Modelo {model_name} no disponible. Usa: {self.MODEL_NAMES}"}

        try:
            if hasattr(match_date, "tzinfo") and match_date.tzinfo:
                match_date = match_date.replace(tzinfo=None)

            features = feature_engineer.create_match_features(
                home_team, away_team, match_date
            )
            df = pd.DataFrame([features])
            numeric = df[self.feature_cols].fillna(0)

            model = self.models[model_name]
            pred = model.predict(numeric)[0]
            probs = model.predict_proba(numeric)[0]

            result_map = {0: "VISITANTE", 1: "LOCAL"}
            return {
                "home": home_team,
                "away": away_team,
                "date": match_date,
                "model": model_name,
                "predicted": result_map[pred],
                "code": int(pred),
                "confidence": float(max(probs)),
                "probabilities": {
                    "VISITANTE": float(probs[0]),
                    "LOCAL": float(probs[1]),
                },
            }
        except Exception as e:
            return {"error": str(e)}

    def get_performance(self) -> Dict:
        return self.performance

    def get_best_model(self) -> str:
        return self.best_model

    def print_comparison(self):
        print("\n  COMPARATIVA WINNER PREDICTOR")
        print("  " + "-" * 65)
        print(f"  {'Modelo':<20} {'Train Acc':<10} {'Test Acc':<10} {'Train AUC':<10} {'Test AUC':<10}")
        print("  " + "-" * 65)
        for name in self.MODEL_NAMES:
            p = self.performance.get(name, {})
            train_a = p.get("train_acc", 0)
            test_a = p.get("test_acc", 0)
            train_auc = p.get("train_auc", 0)
            test_auc = p.get("test_auc", 0)
            marker = " <<<" if name == self.best_model else ""
            print(f"  {name:<20} {train_a:<10.3f} {test_a:<10.3f} {train_auc:<10.3f} {test_auc:<10.3f}{marker}")
        print("  " + "-" * 65)

    def get_feature_importance(self, model_name: str = None,
                               feature_names: List[str] = None) -> List[Tuple]:
        if model_name is None:
            model_name = self.best_model
        model = self.models.get(model_name)
        if model is None:
            return []
        imp = None
        if hasattr(model, "calibrated_classifiers_"):
            est = model.calibrated_classifiers_[0].estimator
            if hasattr(est, "feature_importances_"):
                imp = est.feature_importances_
        if imp is None and hasattr(model, "estimator") and hasattr(model.estimator, "feature_importances_"):
            imp = model.estimator.feature_importances_
        if imp is None and hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        if imp is None:
            return []
        if feature_names is None:
            feature_names = self.feature_cols
        return sorted(zip(feature_names, imp), key=lambda x: x[1], reverse=True)


def get_winner_predictor(models_dir: str = "models") -> WinnerPredictor:
    return WinnerPredictor(models_dir)
