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
        res[i] = (Complex){ random(), random() };
    }
    return res;
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
    
    Complex* tr_fx = mpi_fft(tr_x, q, crank, csize, 0);
    Complex* tr_iffx = mpi_inverse_fft(tr_fx, q, crank, csize, 0);

    if (crank == 0) {
        Complex* diff = sub_cvec(tr_iffx, tr_x, q * q);
        printf("norm=%f", norm_cvec(diff, q * q));

        free(tr_x);
        free(tr_fx);
        free(tr_iffx);
    }

    MPI_Finalize();
    return 0;
}