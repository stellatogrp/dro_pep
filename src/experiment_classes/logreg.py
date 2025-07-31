from sklearn.datasets import load_breast_cancer


class LogReg(object):

    def __init__(self):
        full_X, full_y = load_breast_cancer(return_X_y=True)
        print(full_X)
        print(full_y)
        print(full_X.shape, full_y.shape)


def main():
    lr = LogReg()


if __name__ == '__main__':
    main()
