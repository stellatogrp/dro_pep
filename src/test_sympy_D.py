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

    print('D:')
    sym.pprint(D)

    print('DTD:')
    DTD = D.T @ D
    sym.pprint(sym.simplify(DTD))

    sym.pprint(DTD.evalf(subs={L:1}))

    sym.pprint(DTD.evalf(subs={L:1}).det())

    # sym.pprint(DTD.evalf(subs={L:1}).det())
    # sym.pprint(DTD.det())

    # test = DTD.evalf(subs={L:100})
    # sym.pprint(test.det())

    print('DTD_inv:')
    DTDinv = sym.simplify(DTD.inv())
    sym.pprint(DTDinv)

    sym.pprint(DTDinv.evalf(subs={L:1}))

    print('-----')

    DTDinv_DT = DTDinv @ D.T
    print('DTDinv_DT:')
    sym.pprint(DTDinv_DT)

    test = DTDinv_DT.evalf(subs={L:1000})
    sym.pprint(test)

    # sym.pprint(sym.limit(DTDinv_DT[5, 8], L, sym.oo))

    print('D_DTDinv_DT:')
    sym.pprint((D @ DTDinv_DT).evalf(n=5, subs={L:1}))


if __name__ == '__main__':
    main()
