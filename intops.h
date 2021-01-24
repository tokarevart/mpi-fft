#include <stdbool.h>

static int sqrt_int(int x) {
    for (int i = 0; i < x / 2; ++i) {
        if (i * i == x) {
            return i;
        }
    }
    return -1;
} 

static inline int min_int(int left, int right) {
    if (left < right) {
        return left;
    } else {
        return right;
    }
}

static int power_of(int n, int base) {
    int res = 0;
    while (n > 1) {
        if (n % base != 0) {
            res = -1;
            break;
        }
        n /= base;
        ++res;
    }
    return res;
}

static inline bool is_power_of(int n, int base) {
    return power_of(n, base) != -1;
}

typedef struct {
    int beg;
    int size;
} IntBlock;

// constraint: block_idx < num_blocks <= total
static inline IntBlock partition(int total, int num_blocks, int block_idx) {
    int block_maxsize = (total - 1) / num_blocks + 1;
    int block_beg = block_idx * block_maxsize;
    int block_end = min_int(block_beg + block_maxsize, total);
    return (IntBlock){ block_beg, block_end - block_beg };
}