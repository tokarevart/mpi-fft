#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <stdbool.h>
#include <math.h>
// #include <mpi.h>
#include "mpi.h" // only for intellisense

typedef struct {
    double re;
    double im;
} Complex;

Complex conj_compl(Complex c) {
    // Complex res = { c.re, -c.im };
    return (Complex){ c.re, -c.im };
}

Complex add_compl(Complex lhs, Complex rhs) {
    // Complex res = { lhs.re + rhs.re, lhs.im + rhs.im };
    return (Complex){ lhs.re + rhs.re, lhs.im + rhs.im };
}

Complex sub_compl(Complex lhs, Complex rhs) {
    // Complex res = { lhs.re - rhs.re, lhs.im - rhs.im };
    return (Complex){ lhs.re - rhs.re, lhs.im - rhs.im };
}

Complex scale_compl(Complex c, double k) {
    // Complex res = { c.re * k, c.im * k };
    return (Complex){ c.re * k, c.im * k };
}

Complex mul_compl(Complex lhs, Complex rhs) {
    // Complex res = { 
    //     lhs.re * rhs.re - lhs.im * rhs.im, 
    //     lhs.re * rhs.im + lhs.im * rhs.re 
    // };
    return (Complex) { 
        lhs.re * rhs.re - lhs.im * rhs.im, 
        lhs.re * rhs.im + lhs.im * rhs.re 
    };
}

Complex expi(double x) {
    // Complex res = { cos(x), sin(x) };
    return (Complex){ cos(x), sin(x) };
}

Complex dft_expi(double top, double bottom) {
    const double twopi = 6.2831853071795864769;
    return expi(twopi * top / bottom);
}

Complex dft_prod(const Complex* cvec, int q, int l, double nfactor, int expsign) {
    Complex res = { 0.0, 0.0 };
    int signedl = l * expsign;
    for (int i = 0; i < q; ++i) {
        Complex epsiprod = mul_compl(cvec[i], dft_expi(signedl * i, q));
        res = add_compl(res, epsiprod);
    }
    return scale_compl(res, nfactor);
}

Complex forward_dft_prod(const Complex* cvec, int q, int l) {
    return dft_prod(cvec, q, l, 1.0, -1);
}

Complex inverse_dft_prod(const Complex* cvec, int q, int l) {
    return dft_prod(cvec, q, l, 1.0 / (q * q), 1);
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
    int size;
} IntBlock;

// constraint: block_idx < num_blocks <= total
IntBlock partition(int total, int num_blocks, int block_idx) {
    num_blocks = min_int(num_blocks, total);
    int block_maxsize = (total - 1) / num_blocks + 1;
    int block_beg = block_idx * block_maxsize;
    int block_end = min_int(block_beg + block_maxsize, total);
    // IntBlock res = { block_beg, block_end, block_end - block_beg };
    return (IntBlock){ block_beg, block_end - block_beg };
}

// in:  transposed matrix of x
// out: transposed matrix of F(x)
// transposed means column-major
// 0-th process is root
Complex* fft(const Complex* trcmat, int q, int crank, int csize) {
    IntBlock block = partition(q, csize, crank);
    Complex* flatlines = calloc(block.size * q, sizeof(Complex));

    if (crank == 0) {
        memcpy(
            flatlines, 
            trcmat + block.beg * q, 
            block.size * q * sizeof(Complex)
        );

        for (int i = 1; i < csize; ++i) {
            IntBlock block_i = partition(q, csize, i);
            MPI_Send(
                trcmat + block_i.beg * q, 
                block_i.size * q * sizeof(Complex), 
                MPI_BYTE, i, 0, MPI_COMM_WORLD
            );
        }

    } else {
        MPI_Recv(
            flatlines, 
            block.size * q * sizeof(Complex), 
            MPI_BYTE, 0, 0, MPI_COMM_WORLD
        );
    }

    //
}

int main(int argc, char** argv) {
    int q = 4;

    int crank, csize;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &crank);
    MPI_Comm_size(MPI_COMM_WORLD, &csize);
    if (crank >= q) {
        MPI_Finalize();
        return 0;
    }
    csize = min_int(csize, q);
    srand(crank);

    Complex* tr_x = NULL;
    if (crank == 0) {
        tr_x = random_tr_cmat(q);
    }
    
    Complex* tr_fx = fft(tr_x, q, crank, csize);

    MPI_Finalize();
    return 0;
}