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