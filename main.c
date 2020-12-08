#include <stdlib.h>
#include <stdio.h>
#include "fft.h"

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
        res[i] = (Complex){ random(), 0.0 };
    }
    return res;
}

void print_cmat(const Complex* cvec, int nrows, int ncols) {
    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < ncols; ++j) {
            printf("(%f, %f) ", cvec[i * ncols + j].re, cvec[i * ncols + j].im);
        }
        printf("\n");
    }
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
    srand(crank + 1);

    Complex* tr_x = NULL;
    if (crank == 0) {
        tr_x = random_tr_cmat(q);
    }
    
    Complex* mpi_tr_fx = mpi_fft(tr_x, q, crank, csize, 0);
    Complex* mpi_tr_ifx = mpi_inverse_fft(mpi_tr_fx, q, crank, csize, 0);
    Complex* mpi_x = mpi_transpose_root_cmat(tr_x, q, crank, csize, 0);
    
    if (crank == 0) {
        printf("tr_x\n");
        print_cmat(tr_x, q, q);
        printf("\n");

        Complex* x = transpose_cmat(tr_x, q, q);
        printf("x\n");
        print_cmat(x, q, q);
        printf("\n");

        printf("mpi_x\n");
        print_cmat(mpi_x, q, q);
        printf("\n");
    
        Complex* fft_tr_fx = fft(tr_x, q);
        printf("fft_tr_fx\n");
        print_cmat(fft_tr_fx, q, q);
        printf("\n");

        Complex* dft_fx = dft(x, q);
        printf("dft_fx\n");
        print_cmat(dft_fx, q, q);
        printf("\n");
        
        printf("mpi_tr_fx\n");
        print_cmat(mpi_tr_fx, q, q);
        printf("\n");

        Complex* dft_ifx = inverse_dft(dft_fx, q);
        printf("dft_ifx\n");
        print_cmat(dft_ifx, q, q);
        printf("\n");

        Complex* fft_tr_ifx = inverse_fft(fft_tr_fx, q);
        printf("fft_tr_ifx\n");
        print_cmat(fft_tr_ifx, q, q);
        printf("\n");
        
        printf("mpi_tr_ifx\n");
        print_cmat(mpi_tr_ifx, q, q);
        printf("\n");

        Complex* dft_diff = sub_cvec(dft_ifx, x, q * q);
        printf("dft_diff\n");
        print_cmat(dft_diff, q, q);
        printf("\n");

        Complex* fft_diff = sub_cvec(fft_tr_ifx, tr_x, q * q);
        printf("fft_diff\n");
        print_cmat(fft_diff, q, q);
        printf("\n");

        Complex* mpi_diff = sub_cvec(mpi_tr_ifx, tr_x, q * q);
        printf("mpi_diff\n");
        print_cmat(mpi_diff, q, q);
        printf("\n");

        printf("norm2 x        = %f\n", norm2_cvec(tr_x, q * q));
        printf("norm2 dft_fx   = %f\n", norm2_cvec(dft_fx, q * q));
        printf("norm2 fft_fx   = %f\n", norm2_cvec(fft_tr_fx, q * q));
        printf("norm2 mpi_fx   = %f\n", norm2_cvec(mpi_tr_fx, q * q));
        printf("norm2 dft_ifx  = %f\n", norm2_cvec(dft_ifx, q * q));
        printf("norm2 fft_ifx  = %f\n", norm2_cvec(fft_tr_ifx, q * q));
        printf("norm2 mpi_ifx  = %f\n", norm2_cvec(mpi_tr_ifx, q * q));
        printf("norm2 dft_diff = %f\n", norm2_cvec(dft_diff, q * q));
        printf("norm2 fft_diff = %f\n", norm2_cvec(fft_diff, q * q));
        printf("norm2 mpi_diff = %f\n", norm2_cvec(mpi_diff, q * q));

        free(x);
        free(mpi_x);
        free(dft_ifx);
        free(fft_tr_ifx);
        free(fft_tr_fx);
        free(dft_fx);
        free(dft_diff);
        free(fft_diff);
        free(mpi_diff);
        free(tr_x);
        free(mpi_tr_fx);
        free(mpi_tr_ifx);
    }

    MPI_Finalize();
    return 0;
}