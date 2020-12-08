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

Complex* random_cmat(int q) {
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
    int n = q * q;

    int crank, csize;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &crank);
    MPI_Comm_size(MPI_COMM_WORLD, &csize);
    if (crank >= q) {
        MPI_Finalize();
        return 0;
    }
    srand(crank + 1);

    Complex* x = NULL;
    if (crank == 0) {
        x = random_cmat(q);
        // for (int i = 0; i <= 16; ++i) {
        //     printf("sqrt(%d)=%d\n", i, sqrt_int(i));
        // }
    }
    
    Complex* mpi_fx = mpi_fft(x, n, crank, csize, 0);
    Complex* mpi_ifx = mpi_inverse_fft(mpi_fx, n, crank, csize, 0);
    
    if (crank == 0) {
        printf("x\n");
        print_cmat(x, q, q);
        printf("\n");
    
        Complex* fft_fx = fft(x, n);
        printf("fft_fx\n");
        print_cmat(fft_fx, q, q);
        printf("\n");

        Complex* dft_fx = dft(x, n);
        printf("dft_fx\n");
        print_cmat(dft_fx, q, q);
        printf("\n");
        
        printf("mpi_fx\n");
        print_cmat(mpi_fx, q, q);
        printf("\n");

        Complex* dft_ifx = inverse_dft(dft_fx, n);
        printf("dft_ifx\n");
        print_cmat(dft_ifx, q, q);
        printf("\n");

        Complex* fft_ifx = inverse_fft(fft_fx, n);
        printf("fft_ifx\n");
        print_cmat(fft_ifx, q, q);
        printf("\n");
        
        printf("mpi_ifx\n");
        print_cmat(mpi_ifx, q, q);
        printf("\n");

        Complex* dft_diff = sub_cvec(dft_ifx, x, q * q);
        printf("dft_diff\n");
        print_cmat(dft_diff, q, q);
        printf("\n");

        Complex* fft_diff = sub_cvec(fft_ifx, x, q * q);
        printf("fft_diff\n");
        print_cmat(fft_diff, q, q);
        printf("\n");

        Complex* mpi_diff = sub_cvec(mpi_ifx, x, q * q);
        printf("mpi_diff\n");
        print_cmat(mpi_diff, q, q);
        printf("\n");

        printf("norm2 x        = %f\n", norm2_cvec(x, q * q));
        printf("norm2 dft_fx   = %f\n", norm2_cvec(dft_fx, q * q));
        printf("norm2 fft_fx   = %f\n", norm2_cvec(fft_fx, q * q));
        printf("norm2 mpi_fx   = %f\n", norm2_cvec(mpi_fx, q * q));
        printf("norm2 dft_ifx  = %f\n", norm2_cvec(dft_ifx, q * q));
        printf("norm2 fft_ifx  = %f\n", norm2_cvec(fft_ifx, q * q));
        printf("norm2 mpi_ifx  = %f\n", norm2_cvec(mpi_ifx, q * q));
        printf("norm2 dft_diff = %f\n", norm2_cvec(dft_diff, q * q));
        printf("norm2 fft_diff = %f\n", norm2_cvec(fft_diff, q * q));
        printf("norm2 mpi_diff = %f\n", norm2_cvec(mpi_diff, q * q));

        free(x);
        free(dft_ifx);
        free(fft_ifx);
        free(fft_fx);
        free(dft_fx);
        free(dft_diff);
        free(fft_diff);
        free(mpi_diff);
        free(mpi_fx);
        free(mpi_ifx);
    }

    MPI_Finalize();
    return 0;
}