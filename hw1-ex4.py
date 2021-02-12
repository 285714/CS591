import numpy as np
import matplotlib.pyplot as plt


def generate(n):
    x = np.random.randint(2, size=n)
    a = np.cumsum(x) + np.random.randint(2, size=n)
    w = np.mod(x + np.floor_divide(np.random.randint(3, size=n), 2), 2)
    return x, a, w


def guess(a, w=None, lookahead=10):
    n = len(a)
    x_ = np.zeros(n)
    x_sum = 0
    for i in range(n):
        d = a[i] - (a[i-1] if i >= 0 else 0)
        g = 0 if d <= 0 else 1
        if w is not None:
            g = w[i] if d in [0,1] else g
        a_ = a[i:i+lookahead]
        z = np.arange(0, min(lookahead, n - i))
        if not(np.all(x_sum + g <= a_) and np.all(x_sum + g + z >= a_-1)):
            g = 1 - g
        x_[i] = g
        x_sum += g
    return x_


ns = [100, 500, 1000, 5000, 50000]
rs = 100
res1 = np.zeros((len(ns), rs))
res2 = np.zeros((len(ns), rs))

for l, n in enumerate(ns):
    for r in range(rs):
        x, a, w = generate(n)
        x1 = guess(a)
        x2 = guess(a, w)
        m1 = np.mean(x1 == x)
        m2 = np.mean(x2 == x)
        res1[l, r] = m1
        res2[l, r] = m2
        print("{}/{} {}/{}:\t(1) {}\t(2) {}".format(l+1, len(ns), r+1, rs, m1, m2))

plt.xticks(range(len(ns)), ns)
plt.errorbar(
        -0.1 + np.array(range(len(ns))),
        np.mean(res1, axis=1),
        yerr=np.std(res1, axis=1),
        fmt='o',
        label="a")
plt.errorbar(0.1 + np.array(range(len(ns))),
        np.mean(res2, axis=1),
        yerr=np.std(res2, axis=1),
        fmt='o',
        label="b")
plt.legend()
plt.title("Recovery Rates (each repeated {} times)".format(rs))

plt.savefig('results.pdf')
plt.show()


