import sympy as sym

sym.init_printing()


def main():
    L = sym.symbols('L')
    A1 = sym.Matrix([[0, 0, 0],
                     [0, 1/ (2 * L), 0],
                     [0, 0, 1/(2 * L)]])
    b1 = sym.Matrix([[0], [-1], [1]])

    A2 = sym.Matrix([[0, -1/2, 0],
                     [-1/2, 1/ (2 * L), 0],
                     [0, 0, 0]])
    b2 = sym.Matrix([[-1], [1], [0]])

    A3 = sym.Matrix([[0, 0, -1/2],
                     [0, 0, 1/ (2 * L)],
                     [-1/2, 1/ (2 * L), 1/ (2 * L)]])
    b3 = sym.Matrix([[-1], [0], [1]])

    A4 = sym.Matrix([[-L, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0]])
    b4 = sym.Matrix([[0], [0], [0]])
    
    A5 = 3/(8 * L) * sym.Matrix([[4 * L ** 2 / 9, 2 * L / 3, 2 * L / 3],
                                 [2 * L / 3, 1, 1],
                                 [2 * L / 3, 1, 1]])
    b5 = sym.Matrix([[0], [0], [0]])

    A6 = 1/(8 * L) * sym.Matrix([[0, 0, 0],
                                 [0, 1, -1],
                                 [0, -1, 1]])
    b6 = sym.Matrix([[0], [0], [0]])

    # sym.pprint(sym.Matrix.vstack(A1.vec(), b1))

    D = -sym.Matrix.vstack(A1.vec(), b1)

    for A, b in zip([A2, A3, A4], [b2, b3, b4]):
        D = sym.Matrix.hstack(D, -sym.Matrix.vstack(A.vec(), b))

    sym.pprint(D)

    for A, b in zip([A5, A6], [b5, b6]):
        D = sym.Matrix.hstack(D, sym.Matrix.vstack(A.vec(), b))

    sym.pprint(D)

    DTD = D.T @ D
    sym.pprint(DTD)

    DTDinv = sym.simplify(DTD.inv())
    sym.pprint(DTDinv)

    DTDinv_DT = DTDinv @ D.T
    sym.pprint(DTDinv_DT)


if __name__ == '__main__':
    main()
