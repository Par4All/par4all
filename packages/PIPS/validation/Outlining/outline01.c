void foo(int bar[256][256]) {
    int i,j;
    for(i=0;i<256;i++)
        for(j=0;j<256;j++)
kernel:            bar[i][j]=i*j;
}
