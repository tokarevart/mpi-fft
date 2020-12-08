#include <memory.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include "complex.h"
#include "intops.h"
#include <mpi.h>


Complex dft_expi(double top, double bottom) {
    const double twopi = 6.2831853071795864769;
    return expi(twopi * top / bottom);
}

Complex generic_dft_prod(const Complex* cvec, int q, int l, double nfactor, int expsign) {
    Complex res = { 0.0, 0.0 };
    int signed_l = l * expsign;
    for (int i = 0; i < q; ++i) {
        Complex expiprod = mul_compl(cvec[i], dft_expi(signed_l * i, q));
        res = add_compl(res, expiprod);
    }
    return scale_compl(res, nfactor);
}

Complex* mpi_transpose_cmat(const Complex* lines, int q, int crank, int csize) {
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
Complex* mpi_transpose_root_cmat(const Complex* cmat, int q, int crank, int csize, int root) {
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

// in:  column-major matrix of x
// out: column-major matrix of F(x)
Complex* mpi_generic_fft_colmajor_q(
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
        for (int l = 0; l < q; ++l) {
            Complex acc = { 0.0, 0.0 };
            for (int k = 0; k < q; ++k) {
                Complex expiprod = mul_compl(
                    phi_line[k], 
                    dft_expi(expsign * k * l, q)
                );
                acc = add_compl(acc, expiprod);
            }
            nu_line[l] = acc;
        }
    }
    free(phi);

    Complex* tr_nu = mpi_transpose_cmat(nu, q, crank, csize);
    free(nu);

    Complex* res_part = calloc(block.size * q, sizeof(Complex));
    for (int l = block.beg; l < block.beg + block.size; ++l) {
        Complex* res_line = res_part + (l - block.beg) * q;
        Complex* tr_nu_line = tr_nu + (l - block.beg) * q;
        for (int t = 0; t < q; ++t) {
            Complex acc = { 0.0, 0.0 };
            int signed_qt_l = expsign * (q * t + l);
            for (int s = 0; s < q; ++s) {
                Complex expiprod = mul_compl(
                    tr_nu_line[s], 
                    dft_expi(s * signed_qt_l, q * q)
                );
                acc = add_compl(acc, expiprod);
            }
            res_line[t] = scale_compl(acc, nfactor);
        }
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

Complex* mpi_fft_colmajor_q(const Complex* tr_cmat, int q, int crank, int csize, int root) {
    return mpi_generic_fft_colmajor_q(tr_cmat, q, 1.0, -1, crank, csize, root);
}

Complex* mpi_inverse_fft_colmajor_q(const Complex* tr_cmat, int q, int crank, int csize, int root) {
    return mpi_generic_fft_colmajor_q(tr_cmat, q, 1.0 / (q * q), 1, crank, csize, root);
}

Complex* mpi_generic_fft(
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

Complex* mpi_fft(const Complex* cvec, int n, int crank, int csize, int root) {
    return mpi_generic_fft(cvec, n, 1.0, -1, crank, csize, root);
}

Complex* mpi_inverse_fft(const Complex* cvec, int n, int crank, int csize, int root) {
    return mpi_generic_fft(cvec, n, 1.0 / n, 1, crank, csize, root);
}

// in:  column-major matrix of x
// out: column-major matrix of F(x)
Complex* generic_fft_colmajor_q(const Complex* tr_cmat, int q, double nfactor, int expsign) {
    Complex* nu = calloc(q * q, sizeof(Complex));
    for (int s = 0; s < q; ++s) {
        Complex* nu_line = nu + s * q;
        const Complex* phi_line = tr_cmat + s * q;
        for (int l = 0; l < q; ++l) {
            Complex acc = { 0.0, 0.0 };
            for (int k = 0; k < q; ++k) {
                Complex expiprod = mul_compl(
                    phi_line[k], 
                    dft_expi(expsign * k * l, q)
                );
                acc = add_compl(acc, expiprod);
            }
            nu_line[l] = acc;
        }
    }

    transpose_assign_cmat(nu, q, q);
    Complex* tr_nu = nu;

    Complex* res = calloc(q * q, sizeof(Complex));
    for (int l = 0; l < q; ++l) {
        Complex* res_line = res + l * q;
        Complex* tr_nu_line = tr_nu + l * q;
        for (int t = 0; t < q; ++t) {
            Complex acc = { 0.0, 0.0 };
            int signed_qt_l = expsign * (q * t + l);
            for (int s = 0; s < q; ++s) {
                Complex expiprod = mul_compl(
                    tr_nu_line[s], 
                    dft_expi(s * signed_qt_l, q * q)
                );
                acc = add_compl(acc, expiprod);
            }
            res_line[t] = scale_compl(acc, nfactor);
        }
    }
    free(tr_nu);
    return res;
}

Complex* fft_colmajor_q(const Complex* tr_cmat, int q) {
    return generic_fft_colmajor_q(tr_cmat, q, 1.0, -1);
}

Complex* inverse_fft_colmajor_q(const Complex* tr_cmat, int q) {
    return generic_fft_colmajor_q(tr_cmat, q, 1.0 / (q * q), 1);
}

Complex* generic_fft(const Complex* cvec, int n, double nfactor, int expsign) {
    int q = sqrt_int(n);
    Complex* tr_cmat = transpose_cmat(cvec, q, q);
    Complex* res = generic_fft_colmajor_q(tr_cmat, q, nfactor, expsign);
    transpose_sqr_cmat(res, q);
    free(tr_cmat);
    return res;
}

Complex* fft(const Complex* cvec, int n) {
    return generic_fft(cvec, n, 1.0, -1);
}

Complex* inverse_fft(const Complex* cvec, int n) {
    return generic_fft(cvec, n, 1.0 / n, 1);
}

Complex* generic_dft(const Complex* cvec, int n, double nfactor, int expsign) {
    Complex* res = calloc(n, sizeof(Complex));
    for (int l = 0; l < n; ++l) {
        Complex acc = { 0.0, 0.0 };
        for (int k = 0; k < n; ++k) {
            Complex expiprod = mul_compl(
                cvec[k], 
                dft_expi(expsign * k * l, n)
            );
            acc = add_compl(acc, expiprod);
        }
        res[l] = scale_compl(acc, nfactor);
    }
    return res;
}

Complex* dft(const Complex* cvec, int n) {
    return generic_dft(cvec, n, 1.0, -1);
}

Complex* inverse_dft(const Complex* cvec, int n) {
    return generic_dft(cvec, n, 1.0 / n, 1);
}
