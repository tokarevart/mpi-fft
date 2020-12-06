#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

typedef struct {
    double re;
    double im;
} Complex;

Complex conj_compl(Complex c) {
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
    Complex res = { 0.0, 0.0 };
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

bool is_power_of_two(int n) {
    bool res = true;
    while (n > 1) {
        if (n % 2 == 1) {
            res = false;
            break;
        }
        n /= 2;
    }
    return res;
}

double random() {
    return rand() / (double)RAND_MAX;
}

Complex* random_tr_cmat(int q) {
    int n = q * q;
    if (!is_power_of_two(n)) {
        printf("n=%d is not power of two\n", n);
        return NULL;
    }

    Complex* res = calloc(n, sizeof(Complex));
    for (int i = 0; i < n; ++i) {
        res[i] = (Complex){ random(), random() };
    }
    return res;
}

typedef struct {
    int beg;
    int end;
    int size;
} IntBlock;

// constraint: block_idx < num_blocks <= total
IntBlock partition(int total, int num_blocks, int block_idx) {
    num_blocks = min_int(num_blocks, total);
    int block_maxsize = (total - 1) / num_blocks + 1;
    int block_beg = block_idx * block_maxsize;
    int block_end = min_int(block_beg + block_maxsize, total);
    IntBlock res = { block_beg, block_end, block_end - block_beg };
    return res;
}