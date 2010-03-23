// Define complex number
typedef struct {
    float re;
    float im;
} Cplfloat;

void cadd(int dim1, int dim2, Cplfloat a[dim1][dim2], Cplfloat b[dim2][dim1]) {
    int i, j;
    for (i=0; i<dim1; i++) {
        for (j=0; j<dim2; j++) {
            b[i][j].re += a[i][j].re;
            b[i][j].im += a[i][j].im;
        }
    }
}

