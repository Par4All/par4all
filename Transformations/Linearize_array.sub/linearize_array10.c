typedef float complex[2];
int pof(int n, complex IN[3][n]) {
    complex (*a)[n] = (complex(*)[n])&IN[2];
    return (*a)[3][1];
}

