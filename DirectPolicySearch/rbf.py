import numpy as np


def rbf_test(xt, c1, r1, w1):
    at = [6] * 3  # 3=num_realizations
    return at


def rbf_cubic(labels_perunit_sorted, xt, c_lm, r_lm, w_md, u):
    rule = w_md * (abs(xt - c_lm) / r_lm) ** 3

    return rule


def rbf_gaussian(labels_perunit_sorted, xt, c_lm, r_lm, w_md, u):
    rule = w_md * np.exp(-(np.power(((xt - c_lm) / r_lm), 2)))
    rule = rule * len(labels_perunit_sorted[u])
    at = np.clip(rule, 0, (len(labels_perunit_sorted[u]) - 0.0001))

    return at


def rbf_linear(labels_perunit_sorted, xt, c_lm, r_lm, w_md, u):
    rule = w_md * (c_lm * xt + r_lm)
    rule = rule * (len(labels_perunit_sorted[u]) + 3)
    return rule


