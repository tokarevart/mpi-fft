#include <memory.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include "complex.h"
#include <mpi.h>


int min_int(int left, int right) {
    if (left < right) {
        return left;
    } else {
        return right;
    }
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
    return (IntBlock){ block_beg, block_end - block_beg };
}

Complex dft_expi(double top, double bottom) {
    const double twopi = 6.2831853071795864769;
    return expi(twopi * top / bottom);
}

Complex generic_dft_prod(const Complex* cvec, int q, int l, double nfactor, int expsign) {
    Complex res = { 0.0, 0.0 };
    int signed_l = l * expsign;
    for (int i = 0; i < q; ++i) {
        Complex epsiprod = mul_compl(cvec[i], dft_expi(signed_l * i, q));
        res = add_compl(res, epsiprod);
    }
    return scale_compl(res, nfactor);
}

void generic_dft_line_prod(Complex* out, const Complex* cvec, int q, double nfactor, int expsign) {
    for (int l = 0; l < q; ++l) {
        out[l] = generic_dft_prod(cvec, q, l, nfactor, expsign);
    }
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
    int cur_pos = 0;
    for (int i = 0; i < block.size; ++i) {
        for (int j = 0; j < csize; ++j) {
            int countj_div_blsize = counts[j] / (block.size * sizeof(Complex));
            int part = displs[j] / sizeof(Complex) + countj_div_blsize * i;
            for (int k = 0; k < countj_div_blsize; ++k) {
                res[cur_pos++] = buf[part + k];
            }
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

void mpi_debug_cmat(char* fname_prefix, char* cmat_name, const Complex* cvec, int nrows, int ncols, int crank, const char* mode) {
    char* res = malloc(32 * nrows * ncols + 64);
    char* tmp = malloc(128);
    sprintf(res, "proc %d %s\n", crank, cmat_name);
    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < ncols; ++j) {
            sprintf(tmp, "(%f, %f) ", cvec[i * ncols + j].re, cvec[i * ncols + j].im);
            strcat(res, tmp);
        }
        strcat(res, "\n");
    }
    strcat(res, "\n");
    char* filename = malloc(128);
    strcpy(filename, fname_prefix);
    sprintf(tmp, "proc-%d-debug.out", crank);
    strcat(filename, tmp);
    FILE* file = fopen(filename, mode);
    fputs(res, file);
    fclose(file);
    free(filename);
    free(tmp);
    free(res);
}

void debug_cmat(char* fname_prefix, char* cmat_name, const Complex* cvec, int nrows, int ncols, const char* mode) {
    char* res = malloc(32 * nrows * ncols + 64);
    char* tmp = malloc(128);
    sprintf(res, "%s\n", cmat_name);
    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < ncols; ++j) {
            sprintf(tmp, "(%f, %f) ", cvec[i * ncols + j].re, cvec[i * ncols + j].im);
            strcat(res, tmp);
        }
        strcat(res, "\n");
    }
    strcat(res, "\n");
    char* filename = malloc(128);
    strcpy(filename, fname_prefix);
    sprintf(tmp, "-debug.out");
    strcat(filename, tmp);
    FILE* file = fopen(filename, mode);
    fputs(res, file);
    fclose(file);
    free(filename);
    free(tmp);
    free(res);
}

// in:  column-major matrix of x
// out: column-major matrix of F(x)
Complex* mpi_generic_fft(const Complex* cmat, int q, double nfactor, int expsign, int crank, int csize, int root, char* fname_prefix) {
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
        cmat, counts, displs, MPI_BYTE, 
        phi, block.size * q * sizeof(Complex), MPI_BYTE, 
        root, MPI_COMM_WORLD
    );

    mpi_debug_cmat(fname_prefix, "phi", phi, block.size, q, crank, "w");

    Complex* nu = calloc(block.size * q, sizeof(Complex));
    for (int s = 0; s < block.size; ++s) {
        Complex* nu_line = nu + s * q;
        Complex* phi_line = phi + s * q;
        for (int l = 0; l < q; ++l) {
            Complex acc = { 0.0, 0.0 };
            for (int k = 0; k < q; ++k) {
                Complex epsiprod = mul_compl(
                    phi_line[k], 
                    dft_expi(expsign * k * l, q)
                );
                acc = add_compl(acc, epsiprod);
            }
            nu_line[l] = acc;
        }
    }
    free(phi);

    mpi_debug_cmat(fname_prefix, "nu", nu, block.size, q, crank, "a");

    Complex* tr_nu = mpi_transpose_cmat(nu, q, crank, csize);
    free(nu);

    mpi_debug_cmat(fname_prefix, "tr_nu", tr_nu, block.size, q, crank, "a");

    Complex* res_part = calloc(block.size * q, sizeof(Complex));
    for (int l = block.beg; l < block.beg + block.size; ++l) {
        Complex* res_line = res_part + (l - block.beg) * q;
        Complex* tr_nu_line = tr_nu + (l - block.beg) * q;
        for (int t = 0; t < q; ++t) {
            Complex acc = { 0.0, 0.0 };
            for (int s = 0; s < q; ++s) {
                Complex epsiprod = mul_compl(
                    tr_nu_line[s], 
                    dft_expi(expsign * s * (q * t + l), q * q)
                );
                acc = add_compl(acc, epsiprod);
            }
            res_line[t] = scale_compl(acc, nfactor);
        }
    }
    free(tr_nu);

    mpi_debug_cmat(fname_prefix, "res_part", res_part, block.size, q, crank, "a");

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
        mpi_debug_cmat(fname_prefix, "res", res, q, q, crank, "a");
        free(counts);
        free(displs);
    }
    return res;
}

Complex* mpi_fft(const Complex* cmat, int q, int crank, int csize, int root, char* fname_prefix) {
    return mpi_generic_fft(cmat, q, 1.0, -1, crank, csize, root, fname_prefix);
}

Complex* mpi_inverse_fft(const Complex* cmat, int q, int crank, int csize, int root, char* fname_prefix) {
    return mpi_generic_fft(cmat, q, 1.0 / (q * q), 1, crank, csize, root, fname_prefix);
}

// in:  column-major matrix of x
// out: column-major matrix of F(x)
Complex* generic_fft(const Complex* cmat, int q, double nfactor, int expsign, char* fname_prefix) {
    Complex* nu = calloc(q * q, sizeof(Complex));
    for (int s = 0; s < q; ++s) {
        Complex* nu_line = nu + s * q;
        Complex* phi_line = cmat + s * q;
        for (int l = 0; l < q; ++l) {
            Complex acc = { 0.0, 0.0 };
            for (int k = 0; k < q; ++k) {
                Complex epsiprod = mul_compl(
                    phi_line[k], 
                    dft_expi(expsign * k * l, q)
                );
                acc = add_compl(acc, epsiprod);
            }
            nu_line[l] = acc;
        }
    }

    debug_cmat(fname_prefix, "nu", nu, q, q, "w");

    transpose_assign_cmat(nu, q, q);
    Complex* tr_nu = nu;

    debug_cmat(fname_prefix, "tr_nu", tr_nu, q, q, "a");

    Complex* res = calloc(q * q, sizeof(Complex));
    for (int l = 0; l < q; ++l) {
        Complex* res_line = res + l * q;
        Complex* tr_nu_line = tr_nu + l * q;
        for (int t = 0; t < q; ++t) {
            Complex acc = { 0.0, 0.0 };
            for (int s = 0; s < q; ++s) {
                Complex epsiprod = mul_compl(
                    tr_nu_line[s], 
                    dft_expi(expsign * s * (q * t + l), q * q)
                );
                acc = add_compl(acc, epsiprod);
            }
            res_line[t] = scale_compl(acc, nfactor);
        }
    }
    debug_cmat(fname_prefix, "res", res, q, q, "a");
    free(tr_nu);
    return res;
}

Complex* fft(const Complex* cmat, int q, char* fname_prefix) {
    return generic_fft(cmat, q, 1.0, -1, fname_prefix);
}

Complex* inverse_fft(const Complex* cmat, int q, char* fname_prefix) {
    return generic_fft(cmat, q, 1.0 / (q * q), 1, fname_prefix);
}

// in:  column-major matrix of x
// out: column-major matrix of F(x)
Complex* generic_dft(const Complex* cmat, int q, double nfactor, int expsign) {
    int n = q * q;
    Complex* res = calloc(n, sizeof(Complex));
    for (int l = 0; l < n; ++l) {
        Complex acc = { 0.0, 0.0 };
        for (int k = 0; k < n; ++k) {
            Complex epsiprod = mul_compl(
                cmat[k], 
                dft_expi(expsign * k * l, n)
            );
            acc = add_compl(acc, epsiprod);
        }
        res[l] = scale_compl(acc, nfactor);
    }
    return res;
}

Complex* dft(const Complex* cmat, int q) {
    return generic_dft(cmat, q, 1.0, -1);
}

Complex* inverse_dft(const Complex* cmat, int q) {
    return generic_dft(cmat, q, 1.0 / (q * q), 1);
}
