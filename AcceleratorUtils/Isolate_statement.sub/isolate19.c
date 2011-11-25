typedef struct {
    double im,re;
} dcomplex;
void sain(dcomplex *t) {
    t[0].re=0;
    t[1].im=0;
}
void pain() {
    dcomplex *t=malloc(sizeof(*t)*18);
holy:sain(t);
}
