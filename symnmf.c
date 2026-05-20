#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* Prints a matrix in the required output format:
   values are printed with 4 decimal digits and separated by commas. */
void print_matrix(double **mat, int rows, int cols) {
    int i, j;

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {

            if (j == cols - 1) {
                printf("%.4f", mat[i][j]);
            } else {
                printf("%.4f,", mat[i][j]);
            }

        }
        printf("\n");
    }
}

/* Allocates a rows x cols matrix initialized to zeros.
   Returns NULL if memory allocation fails. */
double **allocate_matrix(int rows, int cols) {
    int i, j;
    double **mat = (double **)malloc(rows * sizeof(double *));
    if (mat == NULL) {
        return NULL;
    }

    for (i = 0; i < rows; i++) {
        mat[i] = (double *)calloc(cols, sizeof(double));
        if (mat[i] == NULL) {
            for (j = 0; j < i; j++) {
                free(mat[j]);
            }
            free(mat);
            return NULL;
        }
    }

    return mat;
}

/* Frees a dynamically allocated matrix with the given number of rows. */
void free_matrix(double **mat, int rows) {
    int i;
    if (mat == NULL) {
        return;
    }

    for (i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
}

/* Computes C = A * B^T.
   This is used for calculating H * H^T without explicitly transposing H. */
void multiply_matrices_transpose(double **A, double **B, double **C, int n, int k, int m) {

    int i, j, l;

    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            C[i][j] = 0.0;
            for (l = 0; l < k; l++) {
                /* Instead of multiplying row of A with column of B^T,
                   we multiply row of A with row of B. */
                C[i][j] += A[i][l] * B[j][l];
            }
        }
    }
}

/* Computes standard matrix multiplication: C = A * B.
   A is m x p, B is p x n, and C is m x n. */
void multiply_matrices(double **A, double **B, double **C, int m, int p, int n) {
    
    int i, j, k;
    
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            /* Initialize the cell to 0 */
            C[i][j] = 0.0; 
            for (k = 0; k < p; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

/* Computes the squared Euclidean distance between two d-dimensional points. */
double squared_distance(double *p1, double *p2, int d) {

    double sum = 0.0;
    int i;

    for (i = 0; i < d; i++) {
        sum += (p1[i] - p2[i]) * (p1[i] - p2[i]);
    }
    return sum;
}

/* Computes the similarity matrix A.
   For i != j: A[i][j] = exp(-||xi - xj||^2 / 2), and A[i][i] = 0. */
void sym(double **X, double **A, int n, int d) {

    int i, j;
    double dist, val;   

    for (i = 0; i < n; i++) {

        A[i][i] = 0.0; /* Diagonal elements are 0 */

        for (j = i + 1; j < n; j++) {

            dist = squared_distance(X[i], X[j], d);
            val = exp(-dist / 2.0);
            A[i][j] = val; /* Upper triangle */
            A[j][i] = val; /* Lower triangle (Symmetry) */
        }
    }
}

/* Computes the normalized similarity matrix W = D^(-1/2) * A * D^(-1/2),
   using the diagonal values of the degree matrix. */
void norm(double **A, double *D_diag, double **W, int n) {

    int i, j;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (D_diag[i] > 0 && D_diag[j] > 0) {
                W[i][j] = A[i][j] / sqrt(D_diag[i] * D_diag[j]);
            } else {
                W[i][j] = 0.0;
            }
        }
    }
}

/* Computes the diagonal values of the degree matrix.
   Each diagonal entry is the sum of the corresponding row in A. */
void ddg(double **A, double *D_diag, int n){
    int i, j;
    double curr_sum;

    for(i = 0; i < n; i ++){
        curr_sum = 0.0;
        for(j = 0; j < n; j ++){
            curr_sum += A[i][j];
        }
        D_diag[i] = curr_sum;
    }
}

/* Computes the squared Frobenius norm of the difference between two matrices.
   Used as the convergence criterion for the SymNMF update process. */
double frobenius_norm_squared_diff(double **A, double **B, int n, int k) {
    int i, j;
    double sum = 0.0, diff;

    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            diff = A[i][j] - B[i][j];
            sum += diff * diff;
        }
    }

    return sum;
}

/* Updates the H matrix according to the SymNMF update rule.
   The process stops when convergence is reached or max_iter iterations are completed. */
void update_H(double **W, double **H, int n, int k, int max_iter, double epsilon) {
    int iter, i, j;
    double beta = 0.5;
    double **WH, **HHT, **HHTH, **H_new;
    double denominator, change;

    WH = allocate_matrix(n, k); /* WH = W * H */
    HHT = allocate_matrix(n, n); /* HHT = H * H^T */
    HHTH = allocate_matrix(n, k); /* HHTH = (H * H^T) * H */
    H_new = allocate_matrix(n, k);

    if (WH == NULL || HHT == NULL || HHTH == NULL || H_new == NULL) {
        printf("An Error Has Occurred\n");
        return;
    }

    for (iter = 0; iter < max_iter; iter++) {

        /* WH = W * H */
        multiply_matrices(W, H, WH, n, n, k);

        /* HHT = H * H^T */
        multiply_matrices_transpose(H, H, HHT, n, k, n);

        /* HHTH = (H * H^T) * H */
        multiply_matrices(HHT, H, HHTH, n, n, k);

        /* Compute new H */
        for (i = 0; i < n; i++) {
            for (j = 0; j < k; j++) {
                denominator = HHTH[i][j];

                if (denominator != 0.0) {
                    H_new[i][j] = H[i][j] * (1.0 - beta + beta * (WH[i][j] / denominator));
                } else {
                    H_new[i][j] = H[i][j];
                }
            }
        }

        /* Check convergence */
        change = frobenius_norm_squared_diff(H_new, H, n, k);
        
        /* Copy H_new into H */
        for (i = 0; i < n; i++) {
            for (j = 0; j < k; j++) {
                H[i][j] = H_new[i][j];
            }
        }

        /* Stop if the change in H is smaller than epsilon. */
        if (change < epsilon) {
            break;
        }
    }

    free_matrix(WH, n);
    free_matrix(HHT, n);
    free_matrix(HHTH, n);
    free_matrix(H_new, n);
}

/* Reads the input data points from a comma-separated text file.
   The function first counts the number of rows and columns, then allocates
   and fills the data matrix X. */
double **read_points_file(const char *filename, int *n, int *d) {

    FILE *fp;
    char line[1024];
    int rows = 0, cols = 0;
    int i, j;
    double **X;

    fp = fopen(filename, "r");
    if (fp == NULL) {
        return NULL;
    }

    /* count rows and columns */
    while (fgets(line, sizeof(line), fp)) {

        if (rows == 0) {
            cols = 1;
            for (i = 0; line[i] != '\0'; i++) {
                if (line[i] == ',')
                    cols++;
            }
        }

        rows++;
    }

    *n = rows;
    *d = cols;

    rewind(fp);

    X = allocate_matrix(rows, cols);
    if (X == NULL) {
        fclose(fp);
        return NULL;
    }

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            if (j == 0) {
                if (fscanf(fp, "%lf", &X[i][j]) != 1) {
                    fclose(fp);
                    free_matrix(X, rows);
                    return NULL;
                }
            } else {
                if (fscanf(fp, ",%lf", &X[i][j]) != 1) {
                    fclose(fp);
                    free_matrix(X, rows);
                    return NULL;
                }
            }
        }
    }

    fclose(fp);

    return X;
}

int main(int argc, char *argv[]) {

    char *goal;
    char *file_name;

    double **X;
    double **A;
    double **W;
    double *D_diag;

    int n, d;
    int i, j;

    if (argc != 3) {
        printf("An Error Has Occurred\n");
        return 1;
    }

    goal = argv[1];
    file_name = argv[2];

    /* read input points */
    X = read_points_file(file_name, &n, &d);
    if (X == NULL) {
        printf("An Error Has Occurred\n");
        return 1;
    }

    /* allocate matrices */
    A = allocate_matrix(n, n);
    D_diag = (double *)calloc(n, sizeof(double));

    if (A == NULL || D_diag == NULL) {
        printf("An Error Has Occurred\n");
        free_matrix(X, n);
        free_matrix(A, n);
        free(D_diag);
        return 1;
    }

    /* compute similarity matrix */
    sym(X, A, n, d);

    if (strcmp(goal, "sym") == 0) {

        print_matrix(A, n, n);
    }

    else if (strcmp(goal, "ddg") == 0) {
        ddg(A, D_diag, n);

        for (i = 0; i < n; i++) {

            for (j = 0; j < n; j++) {

                if (i == j)
                    printf("%.4f", D_diag[i]);
                else
                    printf("%.4f", 0.0);

                if (j != n - 1)
                    printf(",");
            }
            printf("\n");
        }
    }

    else if (strcmp(goal, "norm") == 0) {

        W = allocate_matrix(n, n);
        if (W == NULL) {
            printf("An Error Has Occurred\n");
            free_matrix(X, n);
            free_matrix(A, n);
            free(D_diag);
            return 1;
        }

        ddg(A, D_diag, n);
        norm(A, D_diag, W, n);

        print_matrix(W, n, n);
        free_matrix(W, n);
    }

    else {
        printf("An Error Has Occurred\n");
        free_matrix(X, n);
        free_matrix(A, n);
        free(D_diag);
        return 1;
    }

    free_matrix(X, n);
    free_matrix(A, n);
    free(D_diag);

    return 0;
}