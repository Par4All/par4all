typedef struct {
    double *im,*re;
} mcomplex;
void sain(mcomplex *t) {
    t->re[0]=0;
    t->im[1]=0;
}
void pain() {
    mcomplex t= { malloc(sizeof(double)*18), malloc(sizeof(double)*1) };
holy:sain(&t);
}
