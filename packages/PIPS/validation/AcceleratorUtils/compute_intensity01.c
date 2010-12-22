void code(int n, int a[n]) {
    int i,j;
    // low computation intensity
    for(i=0;i<n;i++)
        a[i]=1;
    for(i=0;i<n;i++)
        for(j=0;j<n;j++)
            a[i]+=j;
}
