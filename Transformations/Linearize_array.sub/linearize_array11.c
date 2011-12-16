typedef struct { int re,im; } complex;
int pof(int n, complex IN[3][n]) {
    complex (*a)[n] = (complex(*)[n])&IN[2];
    return (*a)[3].im;
}

