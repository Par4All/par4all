typedef int el;
el imine(int n) {
    int a[n*n];
    for(int i=n*n;i>=0;i--) {
        a[i]=i;
    }
    return a;
}
