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

Complex epsi(double x) {
    Complex res = { cos(x), sin(x) };
    return res;
}