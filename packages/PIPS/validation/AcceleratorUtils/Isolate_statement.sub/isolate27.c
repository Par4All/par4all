typedef struct { struct { float im, re; } data[300] ; } type;
int pain( type* rdata) {
    int i;
holy:for(i=0;i<100;i++)
        rdata[0].data[i].im=1.f;
}

int main() {
    type R ;
    R.data[20].im=0;
    pain(&R);
    return (int)R.data[20].im - 1;
}
