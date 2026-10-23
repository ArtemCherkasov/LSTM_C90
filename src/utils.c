#include "utils.h"

/* ==================== Memory Management ==================== */

void *utils_safe_malloc(size_t n) {
    void *p = malloc(n);
    if (!p) {
        fprintf(stderr, "malloc failed (%zu bytes)\n", n);
        exit(EXIT_FAILURE);
    }
    return p;
}

void *utils_safe_calloc(size_t n, size_t s) {
    void *p = calloc(n, s);
    if (!p) {
        fprintf(stderr, "calloc failed (%zu x %zu bytes)\n", n, s);
        exit(EXIT_FAILURE);
    }
    return p;
}

void utils_safe_free(void *ptr) {
    free(ptr);
}

/* ==================== Matrix Operations ==================== */

Matrix *matrix_alloc(int rows, int cols) {
    Matrix *m = (Matrix *)utils_safe_malloc(sizeof(Matrix));
    m->rows = rows;
    m->cols = cols;
    m->data = (double *)utils_safe_calloc((size_t)(rows * cols), sizeof(double));
    return m;
}

void matrix_free(Matrix *m) {
    if (m) {
        free(m->data);
        free(m);
    }
}

void matrix_set(Matrix *m, int r, int c, double val) {
    m->data[r * m->cols + c] = val;
}

double matrix_get(Matrix *m, int r, int c) {
    return m->data[r * m->cols + c];
}

void matrix_fill(Matrix *m, double val) {
    int i, n = m->rows * m->cols;
    for (i = 0; i < n; i++) {
        m->data[i] = val;
    }
}

void matrix_copy(Matrix *dest, Matrix *src) {
    int i, n = dest->rows * dest->cols;
    for (i = 0; i < n; i++) {
        dest->data[i] = src->data[i];
    }
}

void matrix_add(Matrix *dest, Matrix *src) {
    int i, n = dest->rows * dest->cols;
    for (i = 0; i < n; i++) {
        dest->data[i] += src->data[i];
    }
}

void matrix_sub(Matrix *dest, Matrix *src) {
    int i, n = dest->rows * dest->cols;
    for (i = 0; i < n; i++) {
        dest->data[i] -= src->data[i];
    }
}

void matrix_scale(Matrix *m, double val) {
    int i, n = m->rows * m->cols;
    for (i = 0; i < n; i++) {
        m->data[i] *= val;
    }
}

void matrix_elem_mul(Matrix *dest, Matrix *a, Matrix *b) {
    int i, n = a->rows * a->cols;
    for (i = 0; i < n; i++) {
        dest->data[i] = a->data[i] * b->data[i];
    }
}

void matrix_transpose(Matrix *dest, Matrix *src) {
    int r, c;
    for (r = 0; r < src->rows; r++) {
        for (c = 0; c < src->cols; c++) {
            matrix_set(dest, c, r, matrix_get(src, r, c));
        }
    }
}

void matrix_mul(Matrix *dest, Matrix *a, Matrix *b) {
    int i, j, k;
    matrix_fill(dest, 0.0);
    for (i = 0; i < dest->rows; i++) {
        for (j = 0; j < dest->cols; j++) {
            double sum = 0.0;
            for (k = 0; k < a->cols; k++) {
                sum += matrix_get(a, i, k) * matrix_get(b, k, j);
            }
            matrix_set(dest, i, j, sum);
        }
    }
}

double matrix_dot_product(Matrix *a, Matrix *b) {
    double sum = 0.0;
    int i, n = a->rows * a->cols;
    for (i = 0; i < n; i++) {
        sum += a->data[i] * b->data[i];
    }
    return sum;
}

/* ==================== Math Functions ==================== */

double utils_sigmoid(double x) {
    /* Numerically stable sigmoid */
    if (x >= 0) {
        double ez = exp(-x);
        return 1.0 / (1.0 + ez);
    } else {
        double ez = exp(x);
        return ez / (1.0 + ez);
    }
}

double utils_tanh_act(double x) {
    /* Numerically stable tanh — prevents Inf/Inf = NaN for |x| > 430 */
    if (x > 20.0) return 1.0;
    if (x < -20.0) return -1.0;
    double e2x = exp(2.0 * x);
    return (e2x - 1.0) / (e2x + 1.0);
}

double utils_relu(double x) {
    return (x > 0.0) ? x : 0.0;
}

double utils_binary_cross_entropy(double prediction, double target) {
    double p = prediction;
    if (p < EPSILON) p = EPSILON;
    if (p > 1.0 - EPSILON) p = 1.0 - EPSILON;
    return -(target * log(p) + (1.0 - target) * log(1.0 - p));
}

/* ==================== Random Number Generation ==================== */

static double _rand_acc = 0.0;
static int _rand_ready = 0;

void utils_srand(unsigned int seed) {
    srand(seed);
    _rand_acc = 0.0;
    _rand_ready = 0;
}

double utils_rand(void) {
    return (double)rand() / (double)RAND_MAX;
}

double utils_gaussian_random(double mean, double stddev) {
    double u1, u2, r, z;
    do {
        u1 = utils_rand();
        u2 = utils_rand();
        /* Avoid log(0) */
        if (u1 < EPSILON) u1 = EPSILON;
        r = sqrt(-2.0 * log(u1));
        z = r * cos(2.0 * PI * u2);
    } while (z > GAUSSIAN_CLIP || z < -GAUSSIAN_CLIP);
    return mean + stddev * z;
}

/* ==================== Gradient Clipping ==================== */

double utils_clip(double val, double min_val, double max_val) {
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
}

/* ==================== String Helpers ==================== */

void utils_trim(char *str) {
    char *start;
    char *end;
    
    /* Skip leading whitespace */
    start = str;
    while (*start == ' ' || *start == '\t' || *start == '\r' || *start == '\n') start++;
    
    /* Empty string after trimming */
    if (*start == 0) {
        *str = 0;
        return;
    }
    
    /* Find end and trim trailing whitespace */
    end = start + strlen(start) - 1;
    while (end > start && (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n')) end--;
    
    /* Copy trimmed string back to original position */
    if (start != str) {
        memmove(str, start, (size_t)(end - start + 2));
    } else {
        *(end + 1) = 0;
    }
}

char *utils_strdup(const char *s) {
    size_t len = strlen(s) + 1;
    char *dup = (char *)utils_safe_malloc(len);
    memcpy(dup, s, len);
    return dup;
}
