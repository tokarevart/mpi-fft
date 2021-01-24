#include <memory.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include "complex.h"
#include "intops.h"
#include <mpi.h>


static inline Complex dft_expi(int top, int bottom) {
    if (top % bottom == 0) {
        return (Complex){ 1.0, 0.0 };
    }
    const double twopi = 6.2831853071795864769;
    return expi(twopi * (double)top / bottom);
}

static Complex* mpi_transpose_cmat(
    const Complex* lines, int q, int crank, int csize
) {
    IntBlock block = partition(q, csize, crank);
    Complex* tr_lines = transpose_cmat(lines, block.size, q);

    int* counts = calloc(csize, sizeof(int));
    int blsize_szcompl = block.size * sizeof(Complex);
    for (int i = 0; i < csize; ++i) {
        IntBlock block_i = partition(q, csize, i);
        counts[i] = block_i.size * blsize_szcompl;
    }

    int* displs = calloc(csize, sizeof(int));
    displs[0] = 0;
    for (int i = 1; i < csize; ++i) {
        displs[i] = displs[i - 1] + counts[i - 1];
    }

    Complex* buf = calloc(block.size * q, sizeof(Complex));
    MPI_Alltoallv(
        tr_lines, counts, displs, MPI_BYTE, 
        buf, counts, displs, MPI_BYTE, 
        MPI_COMM_WORLD
    );
    free(tr_lines);

    Complex* res = calloc(block.size * q, sizeof(Complex));
    Complex* res_iter = res;
    for (int i = 0; i < block.size; ++i) {
        for (int j = 0; j < csize; ++j) {
            int countj_div_blsize = counts[j] / (block.size * sizeof(Complex));
            int part = displs[j] / sizeof(Complex) + i * countj_div_blsize;
            memcpy(res_iter, buf + part, countj_div_blsize * sizeof(Complex));
            res_iter += countj_div_blsize;
        }
    }
    free(buf);
    free(counts);
    free(displs);
    return res;
}

// for debug of mpi_transpose_cmat
static Complex* mpi_transpose_root_cmat(
    const Complex* cmat, int q, int crank, int csize, int root
) {
    IntBlock block = partition(q, csize, crank);
    Complex* lines = calloc(block.size * q, sizeof(Complex));

    int* counts;
    int* displs;
    if (crank == root) {
        counts = calloc(csize, sizeof(int));
        for (int i = 0; i < csize; ++i) {
            IntBlock block_i = partition(q, csize, i);
            counts[i] = block_i.size * q * sizeof(Complex);
        }

        displs = calloc(csize, sizeof(int));
        displs[0] = 0;
        for (int i = 1; i < csize; ++i) {
            displs[i] = displs[i - 1] + counts[i - 1];
        }
    }

    MPI_Scatterv(
        cmat, counts, displs, MPI_BYTE, 
        lines, block.size * q * sizeof(Complex), MPI_BYTE, 
        root, MPI_COMM_WORLD
    );

    Complex* res_part = mpi_transpose_cmat(lines, q, crank, csize);

    Complex* res = NULL;
    if (crank == root) {
        res = calloc(q * q, sizeof(Complex));
    }
    MPI_Gatherv(
        res_part, block.size * q * sizeof(Complex), MPI_BYTE, 
        res, counts, displs, MPI_BYTE, 
        root, MPI_COMM_WORLD
    );
    free(res_part);
    if (crank == root) {
        free(counts);
        free(displs);
    }
    return res;
}

static void generic_dft(
    Complex* out, const Complex* cvec, int n, 
    double nfactor, int expsign
) {
    for (int l = 0; l < n; ++l) {
        Complex acc = { 0.0, 0.0 };
        for (int k = 0; k < n; ++k) {
            Complex expiprod = mul_compl(
                cvec[k], 
                dft_expi(expsign * k * l, n)
            );
            acc = add_compl(acc, expiprod);
        }
        out[l] = scale_compl(acc, nfactor);
    }
}

static Complex* dft(const Complex* cvec, int n) {
    Complex* res = calloc(n, sizeof(Complex));
    generic_dft(res, cvec, n, 1.0, -1);
    return res;
}

static Complex* inverse_dft(const Complex* cvec, int n) {
    Complex* res = calloc(n, sizeof(Complex));
    generic_dft(res, cvec, n, 1.0 / n, 1);
    return res;
}

static void generic_fft_rec(
    Complex* out, const Complex* cvec, int n, 
    int rec, int expsign
) {
    const int maxrec = 1000;
    if (n == 2) {
        out[0] = add_compl(cvec[0], cvec[1]);
        out[1] = sub_compl(cvec[0], cvec[1]);

    } else if (rec >= maxrec) {
        generic_dft(out, cvec, n, 1.0, expsign);

    } else {
        Complex* evens = calloc(n / 2, sizeof(Complex));
        Complex* odds = calloc(n / 2, sizeof(Complex));
        for (int i = 0; i < n / 2; ++i) {
            evens[i] = cvec[i * 2];
            odds[i] = cvec[i * 2 + 1];
        }
        generic_fft_rec(out, evens, n / 2, rec + 1, expsign);
        generic_fft_rec(out + n / 2, odds, n / 2, rec + 1, expsign);
        free(evens);
        free(odds);
        for (int k = 0; k < n / 2; ++k) {
            Complex tmp = out[k];
            Complex expx = mul_compl(dft_expi(
                expsign * k, n), out[k + n / 2]
            );
            out[k] = add_compl(tmp, expx);
            out[k + n / 2] = sub_compl(tmp, expx);
        }
    }
}

static void generic_fft(
    Complex* out, const Complex* cvec, int n, 
    double nfactor, int expsign
) {
    generic_fft_rec(out, cvec, n, 0, expsign);
    for (int i = 0; i < n; ++i) {
        out[i] = scale_compl(out[i], nfactor);
    }
}

static Complex* fft(const Complex* cvec, int n) {
    Complex* res = calloc(n, sizeof(Complex));
    generic_fft(res, cvec, n, 1.0, -1);
    return res;
}

static Complex* inverse_fft(const Complex* cvec, int n) {
    Complex* res = calloc(n, sizeof(Complex));
    generic_fft(res, cvec, n, 1.0 / n, 1);
    return res;
}

// in:  column-major matrix of x
// out: column-major matrix of F(x)
static Complex* mpi_generic_fft_colmajor_q(
    const Complex* tr_cmat, int q, double nfactor, int expsign, 
    int crank, int csize, int root
) {
    IntBlock block = partition(q, csize, crank);
    Complex* phi = calloc(block.size * q, sizeof(Complex));

    int* counts;
    int* displs;
    if (crank == root) {
        counts = calloc(csize, sizeof(int));
        for (int i = 0; i < csize; ++i) {
            IntBlock block_i = partition(q, csize, i);
            counts[i] = block_i.size * q * sizeof(Complex);
        }

        displs = calloc(csize, sizeof(int));
        displs[0] = 0;
        for (int i = 1; i < csize; ++i) {
            displs[i] = displs[i - 1] + counts[i - 1];
        }
    }

    MPI_Scatterv(
        tr_cmat, counts, displs, MPI_BYTE, 
        phi, block.size * q * sizeof(Complex), MPI_BYTE, 
        root, MPI_COMM_WORLD
    );

    Complex* nu = calloc(block.size * q, sizeof(Complex));
    for (int s = 0; s < block.size; ++s) {
        Complex* nu_line = nu + s * q;
        Complex* phi_line = phi + s * q;
        generic_fft(nu_line, phi_line, q, sqrt(nfactor), expsign);
    }
    free(phi);

    for (int s = 0; s < block.size; ++s) {
        Complex* nu_line = nu + s * q;
        for (int l = 0; l < q; ++l) {
            nu_line[l] = mul_compl(
                nu_line[l], dft_expi(expsign * (block.beg + s) * l, q * q)
            );
        }
    }

    Complex* tr_nu = mpi_transpose_cmat(nu, q, crank, csize);
    free(nu);

    Complex* res_part = calloc(block.size * q, sizeof(Complex));
    for (int l = 0; l < block.size; ++l) {
        Complex* res_line = res_part + l * q;
        Complex* tr_nu_line = tr_nu + l * q;
        generic_fft(res_line, tr_nu_line, q, sqrt(nfactor), expsign);
    }
    free(tr_nu);

    Complex* res = NULL;
    if (crank == root) {
        res = calloc(q * q, sizeof(Complex));
    }
    MPI_Gatherv(
        res_part, block.size * q * sizeof(Complex), MPI_BYTE, 
        res, counts, displs, MPI_BYTE, 
        root, MPI_COMM_WORLD
    );
    free(res_part);
    if (crank == root) {
        free(counts);
        free(displs);
    }
    return res;
}

static Complex* mpi_fft_colmajor_q(
    const Complex* tr_cmat, int q, int crank, int csize, int root
) {
    return mpi_generic_fft_colmajor_q(tr_cmat, q, 1.0, -1, crank, csize, root);
}

static Complex* mpi_inverse_fft_colmajor_q(
    const Complex* tr_cmat, int q, int crank, int csize, int root
) {
    return mpi_generic_fft_colmajor_q(
        tr_cmat, q, 1.0 / (q * q), 1, crank, csize, root
    );
}

static Complex* mpi_generic_fft(
    const Complex* cvec, int n, double nfactor, int expsign, 
    int crank, int csize, int root
) {
    int q = sqrt_int(n);
    Complex* tr_cmat = NULL;
    if (crank == root) {
        tr_cmat = transpose_cmat(cvec, q, q);
    }
    Complex* res = mpi_generic_fft_colmajor_q(
        tr_cmat, q, nfactor, expsign, crank, csize, root
    );
    if (crank == root) {
        transpose_sqr_cmat(res, q);
        free(tr_cmat);
    }
    return res;
}

static Complex* mpi_fft(
    const Complex* cvec, int n, int crank, int csize, int root
) {
    return mpi_generic_fft(cvec, n, 1.0, -1, crank, csize, root);
}

static Complex* mpi_inverse_fft(
    const Complex* cvec, int n, int crank, int csize, int root
) {
    return mpi_generic_fft(cvec, n, 1.0 / n, 1, crank, csize, root);
}
