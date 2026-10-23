#ifndef UTILS_H
#define UTILS_H

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

/* --- Constants --- */
#define PI 3.14159265358979323846
#define CLIP_MIN -5.0
#define CLIP_MAX 5.0
#define EPSILON 1e-15
#define GAUSSIAN_CLIP 10.0

/* --- Memory Management --- */
void *utils_safe_malloc(size_t n);
void *utils_safe_calloc(size_t n, size_t s);
void utils_safe_free(void *ptr);

/* --- Matrix Allocation --- */
typedef struct {
    double *data;
    int rows;
    int cols;
} Matrix;

Matrix *matrix_alloc(int rows, int cols);
void matrix_free(Matrix *m);
void matrix_set(Matrix *m, int r, int c, double val);
double matrix_get(Matrix *m, int r, int c);
void matrix_fill(Matrix *m, double val);
void matrix_copy(Matrix *dest, Matrix *src);
void matrix_add(Matrix *dest, Matrix *src);
void matrix_sub(Matrix *dest, Matrix *src);
void matrix_scale(Matrix *m, double val);
void matrix_elem_mul(Matrix *dest, Matrix *a, Matrix *b);
void matrix_transpose(Matrix *dest, Matrix *src);
void matrix_mul(Matrix *dest, Matrix *a, Matrix *b);
double matrix_dot_product(Matrix *a, Matrix *b);

/* --- Math Functions --- */
double utils_sigmoid(double x);
double utils_tanh_act(double x);
double utils_relu(double x);
double utils_binary_cross_entropy(double prediction, double target);

/* --- Random Number Generation --- */
void utils_srand(unsigned int seed);
double utils_rand(void);
double utils_gaussian_random(double mean, double stddev);

/* --- Gradient Clipping --- */
double utils_clip(double val, double min_val, double max_val);

/* --- String Helpers --- */
void utils_trim(char *str);
char *utils_strdup(const char *s);

#endif /* UTILS_H */
