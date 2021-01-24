#include <memory.h>
#include <math.h>

typedef struct {
    double re;
    double im;
} Complex;

static inline double abs2_compl(Complex c) {
    return c.re * c.re + c.im * c.im;
}

static inline double abs_compl(Complex c) {
    return sqrt(abs2_compl(c));
}

static inline double arg(Complex c) {
    return atan2(c.im, c.re);
}

static inline Complex conj_compl(Complex c) {
    return (Complex){ c.re, -c.im };
}

static inline Complex add_compl(Complex lhs, Complex rhs) {
    return (Complex){ lhs.re + rhs.re, lhs.im + rhs.im };
}

static inline Complex sub_compl(Complex lhs, Complex rhs) {
    return (Complex){ lhs.re - rhs.re, lhs.im - rhs.im };
}

static inline Complex scale_compl(Complex c, double k) {
    return (Complex){ c.re * k, c.im * k };
}

static inline Complex mul_compl(Complex lhs, Complex rhs) {
    return (Complex){ 
        lhs.re * rhs.re - lhs.im * rhs.im, 
        lhs.re * rhs.im + lhs.im * rhs.re 
    };
}

static inline Complex div_compl(Complex lhs, Complex rhs) {
    double t2 = 1.0 / abs2_compl(rhs);
    double t1 = t2 * rhs.re; 
    t2 *= rhs.im;
    return (Complex){ 
        lhs.im * t2 + lhs.re * t1, 
        lhs.im * t1 - lhs.re * t2 
    };
}

static inline Complex expi(double x) {
    return (Complex){ cos(x), sin(x) };
}

static inline void swap_compl(Complex* lhs, Complex* rhs) {
    Complex tmp = *lhs;
    *lhs = *rhs;
    *rhs = tmp;
}



static inline Complex* add_cvec(const Complex* lhs, const Complex* rhs, int n) {
    Complex* res = calloc(n, sizeof(Complex));
    for (int i = 0; i < n; ++i) {
        res[i] = add_compl(lhs[i], rhs[i]);
    }
    return res;
}

static inline void add_assign_cvec(Complex* lhs, const Complex* rhs, int n) {
    for (int i = 0; i < n; ++i) {
        lhs[i] = add_compl(lhs[i], rhs[i]);
    }
}

static inline Complex* sub_cvec(const Complex* lhs, const Complex* rhs, int n) {
    Complex* res = calloc(n, sizeof(Complex));
    for (int i = 0; i < n; ++i) {
        res[i] = sub_compl(lhs[i], rhs[i]);
    }
    return res;
}

static inline void sub_assign_cvec(Complex* lhs, const Complex* rhs, int n) {
    for (int i = 0; i < n; ++i) {
        lhs[i] = sub_compl(lhs[i], rhs[i]);
    }
}

static inline Complex* scale_cvec(const Complex* cvec, int n, double k) {
    Complex* res = calloc(n, sizeof(Complex));
    for (int i = 0; i < n; ++i) {
        res[i] = scale_compl(cvec[i], k);
    }
    return res;
}

static inline void scale_assign_cvec(Complex* cvec, int n, double k) {
    for (int i = 0; i < n; ++i) {
        cvec[i] = scale_compl(cvec[i], k);
    }
}

static inline Complex* mul_cvec(const Complex* lhs, const Complex* rhs, int n) {
    Complex* res = calloc(n, sizeof(Complex));
    for (int i = 0; i < n; ++i) {
        res[i] = mul_compl(lhs[i], rhs[i]);
    }
    return res;
}

static inline void mul_assign_cvec(Complex* lhs, const Complex* rhs, int n) {
    for (int i = 0; i < n; ++i) {
        lhs[i] = mul_compl(lhs[i], rhs[i]);
    }
}

static inline Complex sum_cvec(const Complex* cvec, int n) {
    Complex acc = cvec[0];
    for (int i = 1; i < n; ++i) {
        acc = add_compl(acc, cvec[i]);
    }
    return acc;
}

static inline double norm2_cvec(const Complex* cvec, int n) {
    double acc = 0.0;
    for (int i = 0; i < n; ++i) {
        acc += abs2_compl(cvec[i]);
    }
    return acc;
}

static inline double norm_cvec(const Complex* cvec, int n) {
    return sqrt(norm2_cvec(cvec, n));
}

static void transpose_part_cmat(Complex* dest, int dest_ncols, const Complex* cmat, int nrows, int ncols) {
    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < ncols; ++j) {
            dest[j * dest_ncols + i] = cmat[j + i * ncols];
        }
    }
}

static Complex* transpose_cmat(const Complex* cmat, int nrows, int ncols) {
    Complex* res = calloc(ncols * nrows, sizeof(Complex));
    transpose_part_cmat(res, nrows, cmat, nrows, ncols);
    return res;
}

static void transpose_assign_cmat(Complex* cmat, int nrows, int ncols) {
    Complex* res = transpose_cmat(cmat, nrows, ncols);
    memcpy(cmat, res, nrows * ncols * sizeof(Complex));
    free(res);
}

static inline void transpose_sqr_cmat(Complex* cmat, int q) {
    for (int i = 0; i < q; ++i) {
        for (int j = i + 1; j < q; ++j) {
            swap_compl(cmat + j * q + i, cmat + j + i * q);
        }
    }
}
