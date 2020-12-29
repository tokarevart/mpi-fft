#include <stdbool.h>

static int sqrt_int(int x) {
    if (x == 0 || x == 1) {
        return x;
    }
  
    int beg = 1;
    int end = x / 2;
    int res;
    while (beg <= end) {
        int mid = (beg + end) / 2;
        int mid2 = mid * mid;
  
        if (mid2 == x) {
            return mid;
        }
  
        if (mid2 < x) { 
            beg = mid + 1;
            res = mid;
        } else {
            end = mid - 1;
        }
    }
    return res;
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

static bool is_power_of(int n, int base) {
    return power_of(n, base) != -1;
}

int* binarr_from_int(int size, int num) {
    int* res = calloc(size, sizeof(int));
    for (int i = 0; i < size; ++i) {
        res[i] = num % 2;
        num /= 2;
    }
    return res;
}

void binarr_inc(int* binarr, int size) {
    for (int i = 0; i < size; ++i) {
        if (binarr[i] == 1) {
            binarr[i] = 0;
        } else {
            binarr[i] = 1;
            return;
        }
    }
}

typedef struct {
    int beg;
    int size;
} IntBlock;

// constraint: block_idx < num_blocks <= total
static inline IntBlock partition(int total, int num_blocks, int block_idx) {
    num_blocks = min_int(num_blocks, total);
    int block_maxsize = (total - 1) / num_blocks + 1;
    int block_beg = block_idx * block_maxsize;
    int block_end = min_int(block_beg + block_maxsize, total);
    return (IntBlock){ block_beg, block_end - block_beg };
}