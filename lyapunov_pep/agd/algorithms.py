import numpy as np


def accelerated_gradient_descent(iter_K, eta) :

    dimG = iter_K + 2  # [x0, g0, ..., gn]
    dimF = iter_K + 1  # [f0, ..., fn]
    eyeG = np.eye(dimG)
    eyeF = np.eye(dimF)

    x = eyeG[0, :]  # x0
    g = eyeG[1, :]  # g0
    f = eyeF[0, :]  # f0

    y = x

    xs = 0.0 * x
    gs = 0.0 * g
    fs = 0.0 * f

    repX, repG, repF = [x], [g], [f]

    # Gradient descent algorithm
    for i in range(1, iter_K+1):
        y_prev = y

        # y = x - t * g(x)
        # x = y + (k-1)/(k+2)*(y - y_prev)

        y = x - eta * g
        x = y + (i - 1) / (i + 2) * (y - y_prev) # xi

        g = eyeG[i+1, :]    # gi
        f = eyeF[i, :]      # fi

        repX.append(x)
        repG.append(g)
        repF.append(f)
    
    repX.append(xs)
    repG.append(gs)
    repF.append(fs)

    assert len(repX) == len(repG) == len(repF) == iter_K + 2, "constraint on same number of points"
    repX = np.array(repX)
    repG = np.array(repG)
    repF = np.array(repF)

    return repX, repG, repF, dimG, dimF

if __name__ == "__main__":
    repX, _, _, _, _ = accelerated_gradient_descent(10, 0.5)
    print(repX[1])
    print(repX[-1])

