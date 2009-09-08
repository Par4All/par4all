int K=12;
void foo(int bar[256][256]) {
    int i,j;
    for(i=0;i<256;i++)
kernel:        for(j=0;j<256;j++)
            bar[i][j]=K*i*j;
}
