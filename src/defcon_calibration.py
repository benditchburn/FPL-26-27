import numpy as np
import pandas as pd

from scipy.optimize import minimize

from src.defcon_validation import validate_defcon


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def build_defcon_oof(
    hist: pd.DataFrame,
    shrink_k=4,
) -> pd.DataFrame:

    pieces = []

    folds = [
        (12, 18),
        (18, 25),
        (25, 38),
    ]

    for train_end, test_end in folds:

        v = validate_defcon(
            hist,
            train_end_gw=train_end,
            shrink_k=shrink_k,
        )

        test = v["test"].copy()

        test = test[
            (test["GW"] > train_end)
            & (test["GW"] <= test_end)
        ].copy()

        test["Train Through"] = train_end

        pieces.append(test)

    return pd.concat(
        pieces,
        ignore_index=True,
    )


def fit_defender_calibrator(
    oof: pd.DataFrame,
):
    """
    Fit:
        calibrated_p =
        sigmoid(a + b * logit(raw_p))

    using out-of-fold defender observations.
    """

    d = oof[
        oof["FPL Pos"] == "DEF"
    ].copy()

    p = d["Predicted P"].to_numpy()
    y = d["DC Hit"].to_numpy()

    x = _logit(p)

    def objective(params):
        a, b = params

        pred = _sigmoid(
            a + b * x
        )

        # minimise Brier score directly
        return np.mean(
            (pred - y) ** 2
        )

    result = minimize(
        objective,
        x0=np.array([0.0, 1.0]),
        method="Nelder-Mead",
    )

    if not result.success:
        raise RuntimeError(
            result.message
        )

    a, b = result.x

    raw_brier = np.mean(
        (p - y) ** 2
    )

    calibrated = _sigmoid(
        a + b * x
    )

    calibrated_brier = np.mean(
        (calibrated - y) ** 2
    )

    return {
        "a": a,
        "b": b,
        "raw_brier": raw_brier,
        "calibrated_brier": calibrated_brier,
        "n": len(d),
    }


def calibrate_defender_prob(
    p,
    calibrator,
):
    p = np.asarray(p, dtype=float)

    return _sigmoid(
        calibrator["a"]
        + calibrator["b"]
        * _logit(p)
    )