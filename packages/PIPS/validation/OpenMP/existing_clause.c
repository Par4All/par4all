void dotp(int n, int a[n]) {
    int i,r=0;
#pragma   omp parallel for private(i) reduction(+:r)
    for(i=0;i<n;i++)
        r+=a[i];
}
