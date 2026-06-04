import warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path
from typing import Dict
from math import exp, factorial


class RunsPredictor:
    MODEL_NAMES = ["random_forest", "gradient_boosting", "logistic_regression"]

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models: Dict[str, object] = {}
        self.feature_cols = []
        self.performance: Dict[str, Dict] = {}
        self.best_model: str = "random_forest"

    def train(self, features_df: pd.DataFrame, targets: pd.Series,
               X_test: pd.DataFrame = None, y_test: pd.Series = None) -> bool:
        print("\n" + "=" * 50)
        print("  ENTRENANDO RUNS PREDICTOR (3 MODELOS)")
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

        print(f"  Avg runs: {y_train.mean():.2f}")

        # binary O/U target for LR classifier
        y_ou_train = (y_train > 8.5).astype(int)
        y_ou_test = (y_test > 8.5).astype(int)

        configs = {
            "random_forest": RandomForestRegressor(
                n_estimators=100, max_depth=10, min_samples_split=10,
                min_samples_leaf=5, random_state=42, n_jobs=-1,
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.05,
                min_samples_split=10, min_samples_leaf=5,
                random_state=42, subsample=0.8,
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
            model = configs[name]
            print(f"\n  >> {name}")

            if name == "logistic_regression":
                model.fit(X_train, y_ou_train)
                train_pred_ou = model.predict(X_train)
                test_pred_ou = model.predict(X_test)
                train_proba = model.predict_proba(X_train)
                test_proba = model.predict_proba(X_test)

                train_ou_acc = (train_pred_ou == y_ou_train).mean()
                test_ou_acc = (test_pred_ou == y_ou_test).mean()

                self.models[name] = model
                self.performance[name] = {
                    "train_ou_acc": train_ou_acc,
                    "test_ou_acc": test_ou_acc,
                    "type": "classifier",
                }

                print(f"    Train O/U acc: {train_ou_acc:.3f}")
                print(f"    Test O/U acc:  {test_ou_acc:.3f}")
            else:
                model.fit(X_train, y_train)
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)

                train_mae = mean_absolute_error(y_train, train_pred)
                test_mae = mean_absolute_error(y_test, test_pred)

                train_ou_acc = ((train_pred > 8.5) == (y_train > 8.5)).mean()
                test_ou_acc = ((test_pred > 8.5) == (y_test > 8.5)).mean()

                self.models[name] = model
                self.performance[name] = {
                    "train_mae": train_mae,
                    "test_mae": test_mae,
                    "train_ou_acc": train_ou_acc,
                    "test_ou_acc": test_ou_acc,
                    "type": "regressor",
                }

                print(f"    Train MAE: {train_mae:.2f}  Test MAE: {test_mae:.2f}")
                print(f"    Train O/U: {train_ou_acc:.3f}  Test O/U: {test_ou_acc:.3f}")

        # pick best by test O/U accuracy
        self.best_model = max(
            self.MODEL_NAMES, key=lambda n: self.performance[n]["test_ou_acc"]
        )
        print(f"\n  Mejor modelo: {self.best_model} ({self.performance[self.best_model]['test_ou_acc']:.3f})")
        self._save()
        return True

    def train_final(self, features_df: pd.DataFrame, targets: pd.Series) -> bool:
        """Entrenar todos los modelos con el 100% de los datos y guardar."""
        print("\n" + "=" * 50)
        print("  ENTRENANDO MODELO FINAL RUNS (100% DE DATOS)")
        print("=" * 50)

        self.feature_cols = features_df.columns.tolist()

        configs = {
            "random_forest": RandomForestRegressor(
                n_estimators=100, max_depth=10, min_samples_split=10,
                min_samples_leaf=5, random_state=42, n_jobs=-1,
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.05,
                min_samples_split=10, min_samples_leaf=5,
                random_state=42, subsample=0.8,
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

        y_ou = (targets > 8.5).astype(int)

        for name in self.MODEL_NAMES:
            model = configs[name]
            print(f"  >> {name}")
            if name == "logistic_regression":
                model.fit(features_df, y_ou)
            else:
                model.fit(features_df, targets)
            self.models[name] = model

        self.best_model = max(
            self.MODEL_NAMES,
            key=lambda n: self.performance.get(n, {}).get("test_ou_acc", 0)
        )
        self._save()
        print(f"  Modelo final runs guardado ({len(features_df)} muestras, 100%)")
        return True

    def _save(self):
        self.models_dir.mkdir(parents=True, exist_ok=True)
        for name in self.MODEL_NAMES:
            with open(self.models_dir / f"runs_{name}.pkl", "wb") as f:
                pickle.dump(self.models[name], f)
        with open(self.models_dir / "runs_features.pkl", "wb") as f:
            pickle.dump(self.feature_cols, f)
        with open(self.models_dir / "runs_performance.pkl", "wb") as f:
            pickle.dump({"performance": self.performance, "best": self.best_model}, f)
        print(f"\n  Guardado en {self.models_dir}")

    def load(self) -> bool:
        try:
            for name in self.MODEL_NAMES:
                path = self.models_dir / f"runs_{name}.pkl"
                if path.exists():
                    with open(path, "rb") as f:
                        self.models[name] = pickle.load(f)
            with open(self.models_dir / "runs_features.pkl", "rb") as f:
                self.feature_cols = pickle.load(f)
            perf_path = self.models_dir / "runs_performance.pkl"
            if perf_path.exists():
                with open(perf_path, "rb") as f:
                    data = pickle.load(f)
                    self.performance = data.get("performance", {})
                    self.best_model = data.get("best", "random_forest")
            print(f"  Runs modelos cargados: {list(self.models.keys())}, best={self.best_model}")
            return True
        except Exception as e:
            print(f"  Error cargando runs models: {e}")
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
            return {"error": f"Modelo {model_name} no disponible"}

        try:
            if hasattr(match_date, "tzinfo") and match_date.tzinfo:
                match_date = match_date.replace(tzinfo=None)

            features = feature_engineer.create_match_features(
                home_team, away_team, match_date
            )
            df = pd.DataFrame([features])
            numeric = df[self.feature_cols].fillna(0)

            model = self.models[model_name]
            perf = self.performance.get(model_name, {})
            is_classifier = perf.get("type") == "classifier"

            threshold = 8.5

            if is_classifier:
                pred = model.predict(numeric)[0]
                proba = model.predict_proba(numeric)[0]
                over_prob = float(proba[1])
                expected_runs = None
            else:
                expected_runs = float(model.predict(numeric)[0])
                over_prob = self._poisson_over(expected_runs, threshold)

            prediction = "OVER" if over_prob > 0.5 else "UNDER"
            confidence = abs(over_prob - 0.5) * 2

            result = {
                "home": home_team,
                "away": away_team,
                "date": match_date,
                "model": model_name,
                "expected_runs": expected_runs,
                "markets": {
                    "over_8.5": {
                        "over_prob": over_prob,
                        "under_prob": 1 - over_prob,
                        "prediction": prediction,
                        "confidence": confidence,
                    }
                },
            }
            return result
        except Exception as e:
            return {"error": str(e)}

    def _poisson_over(self, lamb: float, threshold: float) -> float:
        lamb = max(3.0, lamb)
        k = int(threshold)
        prob_under = sum(
            exp(-lamb) * (lamb ** i) / factorial(i)
            for i in range(k)
        )
        return min(0.99, max(0.01, 1 - prob_under))

    def get_performance(self) -> Dict:
        return self.performance

    def get_best_model(self) -> str:
        return self.best_model

    def print_comparison(self):
        print("\n  COMPARATIVA RUNS PREDICTOR")
        print("  " + "-" * 68)
        print(f"  {'Modelo':<20} {'Tipo':<12} {'Test O/U':<10} {'Test MAE':<10}")
        print("  " + "-" * 68)
        for name in self.MODEL_NAMES:
            p = self.performance.get(name, {})
            ptype = p.get("type", "?")
            ou = p.get("test_ou_acc", 0)
            mae = p.get("test_mae", 0)
            mae_str = f"{mae:.2f}" if ptype == "regressor" else "N/A"
            marker = " <<<" if name == self.best_model else ""
            print(f"  {name:<20} {ptype:<12} {ou:<10.3f} {mae_str:<10}{marker}")
        print("  " + "-" * 68)


def get_runs_predictor(models_dir: str = "models") -> RunsPredictor:
    return RunsPredictor(models_dir)
