import numpy as np
import pandas as pd


epsilon = 0.1

def median(R, xs):
    n = len(xs)
    ns = np.clip(np.round(xs), 1, R).astype(int)
    scores = -np.abs(np.sum(np.sign(np.subtract.outer(np.arange(1, R+1), ns)), axis=1))
    noise = np.random.exponential(2 / epsilon, size=R)
    return np.argmax(scores + noise) + 1


def evaluate(xs, R):
    n = len(xs)
    errors = []
    for _ in range(10):
        m = median(R, xs)
        rank = np.sum(xs < m) + 1
        errors.append(abs(rank - n/2))
    return errors


df = pd.DataFrame(columns=["distribution", "R", "n", "i", "error"])

for i in range(50):
    print("===================================== {} =====================================".format(i))
    for n in [50, 100, 500, 2000, 10000]:
        for R in [100, 1000, 10000]:
            for distribution, sample in [("gaussian", np.random.normal(R/4, np.sqrt(R**2/4), n)),
                                         ("poisson", np.random.poisson(R, n))]:
                for e in evaluate(sample, R):
                    df = df.append({"i":i, "n": n, "R": R,
                        "distribution": distribution, "error": e}, ignore_index=True)
        R = 1000
        for k in [10, 100, 200]:
            sample = R/2 + (2*np.random.randint(2, size=n)-1) * k
            for e in evaluate(sample, R):
                df = df.append({"i":i, "n": n, "R": R,
                    "distribution": "bimodal (k={})".format(k), "error": e}, ignore_index=True)

    cols = ["distribution", "R", "n"]
    g = df.groupby(cols)["error"]
    avg_error = g.mean()
    std = g.agg(np.std)
    avg_std = df.groupby(cols + ["i"]).agg(np.std).groupby(cols)["error"].mean()

    results = pd.concat([avg_error.rename("average error"),
                         std.rename("standard deviation"),
                         avg_std.rename("avg std per sample")], axis=1)
    print(results)
    results.to_pickle("./results.pkl")
    results.to_csv("./results.csv")
    df.to_pickle("./all.pkl")


