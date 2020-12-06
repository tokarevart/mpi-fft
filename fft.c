#include <math.h>

typedef struct {
    double re;
    double im;
} Complex;

Complex new_compl(double re, double im) {
    Complex res = { re, im };
    return res;
}

Complex conj(Complex c) {
    Complex res = { c.re, -c.im };
    return res;
}

Complex add_compl(Complex lhs, Complex rhs) {
    Complex res = { lhs.re + rhs.re, lhs.im + rhs.im };
    return res;
}

Complex sub_compl(Complex lhs, Complex rhs) {
    Complex res = { lhs.re - rhs.re, lhs.im - rhs.im };
    return res;
}

Complex scale_compl(Complex c, double k) {
    Complex res = { c.re * k, c.im * k };
    return res;
}

Complex mul_compl(Complex lhs, Complex rhs) {
    Complex res = { 
        lhs.re * rhs.re - lhs.im * rhs.im, 
        lhs.re * rhs.im + lhs.im * rhs.re 
    };
    return res;
}

Complex expi(double x) {
    Complex res = { cos(x), sin(x) };
    return res;
}

Complex dft_expi(double top, double bottom) {
    const double twopi = 6.2831853071795864769;
    return expi(twopi * top / bottom);
}

Complex prod_expi(const Complex* cvec, int q, int l, double nfactor, int expsign) {
    Complex res = new_compl(0.0, 0.0);
    int signedl = l * expsign;
    for (int i = 0; i < q; ++i) {
        Complex epsiprod = mul_compl(cvec[i], dft_expi(signedl * i, q));
        res = add_compl(res, epsiprod);
    }
    return scale_compl(res, nfactor);
}

Complex forward_prod_expi(const Complex* cvec, int q, int l) {
    return prod_expi(cvec, q, l, 1.0, -1);
}

Complex inverse_prod_expi(const Complex* cvec, int q, int l) {
    return prod_expi(cvec, q, l, 1.0 / (q * q), 1);
}