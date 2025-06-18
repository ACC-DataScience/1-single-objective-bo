import os
import warnings
from utils import set_seeds, measure_yield_strength
import numpy as np
from ax.service.ax_client import AxClient, ObjectiveProperties
import matplotlib.pyplot as plt
from ax.modelbridge.cross_validation import cross_validate, compute_diagnostics
from ax.core.observation import ObservationFeatures
import pytest


@pytest.fixture(scope="session")
def get_namespace():
    script_fname = "sobo_assignment.py"
    with open(script_fname) as f:
        script_content = f.read()
    namespace = {}
    exec(script_content, namespace)
    return namespace


def test_task_a(get_namespace):

    running_ax_client = get_namespace["ax_client"]
    user_op_params = running_ax_client.experiment.parameters

    # assert that len op_params is 4
    assert len(user_op_params) == 4, "Expected 4 parameters, got {}".format(
        len(user_op_params)
    )

    # assert that op_params contains ['time', 'temperature', 'v_prct', 'process']
    assert all(
        [
            param in ["time", "temperature", "v_prct", "process"]
            for param in user_op_params
        ]
    ), "Expected parameters named ['time', 'temperature', 'v_prct', 'process'], got {}".format(
        user_op_params.keys()
    )

    # assert that the ax_client budget is 25
    assert (
        len(running_ax_client.get_trials_data_frame()) == 25
    ), "Expected optimization budget of 25 trials, got {}".format(
        len(running_ax_client.get_trials_data_frame())
    )


def test_task_b(get_namespace):

    user_optimal_params = get_namespace["optimal_params"]
    user_optimal_ys = get_namespace["optimal_yield_strength"]

    true_optimal_params = {
        "time": 15.85,
        "temperature": 784.92,
        "v_prct": 2.72,
        "process": "CR",
    }

    # assert that the time parameter is close
    assert np.isclose(
        user_optimal_params["time"], true_optimal_params["time"], atol=3
    ), f"Expected optimal time of 15.85, got {user_optimal_params['time']}"

    # assert that the temperature parameter is close
    assert np.isclose(
        user_optimal_params["temperature"], true_optimal_params["temperature"], atol=10
    ), f"Expected optimal temperature of 784.92, got {user_optimal_params['temperature']}"

    # assert that the v_prct parameter is close
    assert np.isclose(
        user_optimal_params["v_prct"], true_optimal_params["v_prct"], atol=0.5
    ), f"Expected optimal v_prct of 2.72, got {user_optimal_params['v_prct']}"

    # assert that the process parameter is close
    assert (
        user_optimal_params["process"] == true_optimal_params["process"]
    ), f"Expected optimal process of 'CR', got {user_optimal_params['process']}"

    # assert that user_optimal_yield_strength is greater than 950.0
    assert (
        user_optimal_ys > 950.0
    ), "Expected optimal yield strength > 950.0, got {}".format(user_optimal_ys)


def test_task_c(get_namespace):

    user_gp_improvement = get_namespace["gp_improvement"]

    # assert that gp_improvement is greater than 450.0
    assert user_gp_improvement > 450.0, "Expected improvement > 450.0, got {}".format(
        user_gp_improvement
    )


def test_task_d(get_namespace):
    user_most_important = get_namespace["most_important"]

    # assert that the letter "v" is in the most_important string
    assert (
        "v" in user_most_important
    ), "Expected 'v_prct' as most important feature, got {}".format(user_most_important)


def test_task_e(get_namespace):

    user_rmse = get_namespace["rmse"]
    user_corr_coeff = get_namespace["corr_coeff"]

    # assert that the rmse is less than 55
    assert user_rmse < 55, "Expected RMSE < 55, got {}".format(user_rmse)

    # assert that the correlation coefficient is greater than 0.95
    assert (
        user_corr_coeff > 0.95
    ), "Expected correlation coefficient > 0.95, got {}".format(user_corr_coeff)


def test_task_f(get_namespace):

    user_max_deviation = get_namespace["max_deviation"]

    # assert that the user max deviation is between 20 and 35
    assert (
        20 < user_max_deviation < 35
    ), "Expected deviation between 20 and 35, got {}".format(user_max_deviation)
