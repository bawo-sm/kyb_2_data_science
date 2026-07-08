import os, logging, pprint, typing
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report


BEST_PARAMS_FILEPATH = ""
BEST_MODEL_FILEPATH = ""
DATA_FILEPATH = ""
FINE_TUNING: typing.Literal["random", "grid"] = "random"
N_RANDOM_TRIES = 2


def load_train_data():
    """Sztuczne dane do sprawdzenia pipelineu"""
    df = pd.DataFrame(
        dict(
            x1=np.random.randint(0, 100, 100),
            x2=np.random.randint(0, 100, 100),
            y=np.random.randint(0, 2, 100)
        )
    )
    return df[["x1", "x2"]].to_numpy(), df["y"].to_numpy()


def load_test_data():
    """Sztuczne dane do sprawdzenia pipelineu"""
    df = pd.DataFrame(
        dict(
            x1=np.random.randint(0, 100, 100),
            x2=np.random.randint(0, 100, 100),
            y=np.random.randint(0, 2, 100)
        )
    )
    return df[["x1", "x2"]].to_numpy(), df["y"].to_numpy()


def train_on_default_params(
        x_train: list[float], 
        x_test: list[float],
        y_train: list[float], 
        y_test: list[float]
):
    model = LGBMClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return report, model.get_params()


def get_single_metric(report: dict) -> float:
    return report["accuracy"]


def get_random_params():
    return dict(
        boosting_type=np.random.choice(["gbdt", "dart"]),
        num_leaves=np.random.randint(5, 200, 1)[0],
        max_depth=-1 if np.random.randint(0, 100, 1)[0] >= 80 else np.random.randint(5, 500, 1)[0],
        learning_rate=np.random.randint(1, 1000, 1)[0] / 1000,
        n_estimators=np.random.randint(5, 500, 1)[0],
        reg_alpha=np.random.randint(1, 10000, 1)[0] / 10000,
        reg_lambda=np.random.randint(1, 10000, 1)[0] / 10000,
    )


def get_params_grid():
    """
    Zwraca listę zestawów parametrów wg zdefiniwoanej siatki. 
    Wartości parametrów można uszeregowac, np. od najmniejszej złozoności modelu, do coraz większych. 
    Warto dodawać różen learning_rate, reg_alpha oraz reg_lambda. 
    """
    max_depth = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 130]
    num_leaves = [x*3 for x in max_depth]
    n_estimators = [50, 100, 200, 250, 50, 100, 200, 250, 50, 100, 200, 250]
    learning_rate = [0.1, 0.01, 0.001, 0.1, 0.01, 0.001, 0.1, 0.01, 0.001, 0.1, 0.01, 0.001]
    
    assert len(max_depth) == len(num_leaves)
    assert len(max_depth) == len(n_estimators)
    assert len(max_depth) == len(learning_rate)

    params_list = []
    for i in range(len(max_depth)):
        params_list += [
            dict(
                max_depth=max_depth[i],
                num_leaves=num_leaves[i],
                n_estimators=n_estimators[i],
                learning_rate=learning_rate[i],
                reg_alpha=0,
                reg_lambda=0
            ),
            dict(
                max_depth=max_depth[i],
                num_leaves=num_leaves[i],
                n_estimators=n_estimators[i],
                learning_rate=learning_rate[i],
                reg_alpha=0.001,
                reg_lambda=0.001
            ),
            dict(
                max_depth=max_depth[i],
                num_leaves=num_leaves[i],
                n_estimators=n_estimators[i],
                learning_rate=learning_rate[i],
                reg_alpha=0.01,
                reg_lambda=0.01
            )
        ]
        
    return params_list


def fine_tuning_loop(
        x_train: list[float], 
        x_test: list[float],
        y_train: list[float], 
        y_test: list[float],
        params_list: list[dict]
):    
    """
    Pętla do trenowania modelu na róznych zestawach parametów.
    Można zaimplementować MLFlow lub zapiswyac metryki w inny sposób (np. sql, json)
    """

    best_metric = 0
    best_params = None

    for i, params in enumerate(params_list):
        logger.info(f">>> Random loop no {i}")
        model = LGBMClassifier(**params)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        current_metric = get_single_metric(classification_report(y_test, y_pred, output_dict=True))
        current_loop = dict(
            metric=current_metric,
            params=model.get_params()
        )

        if current_metric > best_metric:
            best_metric = current_metric
            best_params = model.get_params()
            # with open(BEST_PARAMS_PATH, "w") as file:
            #     json.dump(current_loop, file, indent=4)
        else:
            pass
            # with open(CURRENT_PARAMS_PATH, "w") as file:
            #     json.dump(current_loop, file, indent=4)

    return best_metric, best_params 


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("LGBM Trainer")

    x_train, y_train = load_train_data()
    x_test, y_test = load_train_data()
    default_report, default_params = train_on_default_params(x_train, x_test, y_train, y_test)
    
    logger.info(">>> Default report:")
    pprint.pp(default_report)
    logger.info(">>> Default params:")
    pprint.pp(default_params)

    match FINE_TUNING:
        case "random":
            params_list = [get_random_params() for i in range(N_RANDOM_TRIES)]
        case "grid":
            params_list = get_params_grid()
        case _:
            raise ValueError

    best_metric, best_params = fine_tuning_loop(x_train, x_test, y_train, y_test, params_list)