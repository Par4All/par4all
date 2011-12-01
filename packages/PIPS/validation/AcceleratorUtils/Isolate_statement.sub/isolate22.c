typedef struct {
    double im,re;
} complex_t;
void sain(complex_t t[10]) {
    t[0].re=0;
    t[1].im=0;
}
void pain() {
    complex_t t[10];
holy:sain(t);
}
