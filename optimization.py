from  lightgbm import LGBMRegressor
import optuna
from Configs import (
    FeatureConfig,
    ModelConfig,
)
from main import evaluate_model

def run_optuna(n_trials: int):
    # Set startup trials as 5 because out total trials is lower.
    sampler = optuna.samplers.TPESampler(n_startup_trials=5, seed=42)
    # Create a study
    study = optuna.create_study(direction="maximize", sampler=sampler)
    # Start the optimization run
    study.optimize(lgbm_objective_step2, n_trials=n_trials, show_progress_bar=True)

    # Show best results
    trial = study.best_trial

    print('Number of finished trials: ', len(study.trials))
    print('Best trial:')
    print('   Trial id:', trial.number)
    print('   Score:', trial.value)
    print('Params:')

    for key, value in trial.params.items():
        print('   {}: {}'.format(key, value))

def lgbm_objective_step1(trial):

    feature_config = FeatureConfig(
        date='time_id',
        target='responder_6',
        weight='weight',
        continuous_features=[f"feature_{idx:02d}" for idx in range(79) if idx not in [9, 10, 11]] + \
            [f"responder_{idx}_lag_1" for idx in range(9)] + \
            ['time_id_Elapsed',
             'time_id_Period_968_sin_1','time_id_Period_968_sin_2','time_id_Period_968_sin_3','time_id_Period_968_sin_4','time_id_Period_968_sin_5',
             'time_id_Period_968_cos_1','time_id_Period_968_cos_2','time_id_Period_968_cos_3','time_id_Period_968_cos_4','time_id_Period_968_cos_5'],
        categorical_features=['symbol_id'] + [f"feature_{idx:02d}" for idx in [9, 10, 11]] + [f"time_id_Period_{period}" for period in [332, 725, 968]],
        boolean_features=[],
        index_cols=["time_id"],
        exogenous_features=[f"feature_{idx:02d}" for idx in range(79)]
    )

    lgb_params={
        "boosting_type": "gbdt",
        "objective":     trial.suggest_categorical("objective", ["regression", "regression_l1", "huber"]),
        "random_state":  2025,
        #"max_depth":     10,
        "learning_rate": 0.10,
        "n_estimators":  trial.suggest_int("n_estimators", 5, 200),
        "colsample_bytree": trial.suggest_float ("colsample_bytree", 0.3, 1.0),
        "colsample_bynode": 0.6,
        "lambda_l1":     trial.suggest_float("lambda_l1", 0, 10),
        "lambda_l2":     trial.suggest_float("lambda_l2", 0, 10),
        "extra_trees":   True,
        "num_leaves":    trial.suggest_int("num_leaves", 10, 100),
        "max_bin":       255,
        #'device':'gpu',
        "n_jobs":        -1,
        #"verbose":       1,
    }
    model_config = ModelConfig(
        model=LGBMRegressor(**lgb_params, verbose=-1), # try MAE or Huber loss to penalize outliers less!!
        name="LightGBM",
        normalize=False, # LGBM is not affected by normalization
        fill_missing=False, # # LGBM handles missing values
    )

    _, metrics, _= evaluate_model(feature_config, model_config, train_dates=[1580, 1600], test_dates=[1600, 1640], use_weights=True)

    return metrics['R2']


def lgbm_objective_step2(trial):

    feature_config = FeatureConfig(
        date='time_id',
        target='responder_6',
        weight='weight',
        continuous_features=[f"feature_{idx:02d}" for idx in range(79) if idx not in [9, 10, 11]] + \
            [f"responder_{idx}_lag_1" for idx in range(9)] + \
            ['time_id_Elapsed',
             'time_id_Period_968_sin_1','time_id_Period_968_sin_2','time_id_Period_968_sin_3','time_id_Period_968_sin_4','time_id_Period_968_sin_5',
             'time_id_Period_968_cos_1','time_id_Period_968_cos_2','time_id_Period_968_cos_3','time_id_Period_968_cos_4','time_id_Period_968_cos_5'],
        categorical_features=['symbol_id'] + [f"feature_{idx:02d}" for idx in [9, 10, 11]] + [f"time_id_Period_{period}" for period in [332, 725, 968]],
        boolean_features=[],
        index_cols=["time_id"],
        exogenous_features=[f"feature_{idx:02d}" for idx in range(79)]
    )

    lgb_params={
        "boosting_type": "gbdt",
        "objective":     'mean_squared_error',
        "random_state":  2025,
        "max_depth":     7,
        "learning_rate": 0.05,
        "n_estimators":  trial.suggest_int("n_estimators", 5, 500),
        "colsample_bytree": 0.6,
        "colsample_bynode": 0.6,
        "reg_alpha":     0.2,
        "reg_lambda":    5,
        "extra_trees":  True,
        "num_leaves":    64,
        "max_bin":       255,
        #'device':'gpu',
        "n_jobs":        -1,
    }
    model_config = ModelConfig(
        model=LGBMRegressor(**lgb_params, verbose=-1), # try MAE or Huber loss to penalize outliers less!!
        name="LightGBM",
        normalize=False, # LGBM is not affected by normalization
        fill_missing=False, # # LGBM handles missing values
    )

    date_id_min = trial.suggest_int("date_id_min", 1350, 1575)
    _, metrics, _= evaluate_model(feature_config, model_config, train_dates=[date_id_min, 1580], test_dates=[1580, 1630], use_weights=True)

    return metrics['R2']