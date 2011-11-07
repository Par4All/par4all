int main() {
    int a[18],i,j;

    for(j=0;j<5;j++)
    {
        for(i=j*4;i<MIN((j+1)*4,18);i++)
            /* here, preconditions should be i >= j*4 , i<=(j+1)*4 , i <= 18 */
            a[i]=0;
        /* this loop should only write element that do not exist */
        for(i=MIN((j+1)*4,18);i<(j+1)*4;i++)
            a[i]=0;
    }

    return 0;
}
