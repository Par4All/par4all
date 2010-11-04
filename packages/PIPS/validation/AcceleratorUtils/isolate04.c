#include <stdio.h>
void lonesome_cowboy()
{
  /* Use n & m to trace down dimensions in the generated code... */
  int i,j,n=12,m=24;
    int a[n][m];
    for(i=0;i<n;i++)
        for(j=0;j<m;j++)
            a[i][j]=0;
    for(i=1;i<n-1;i++)
isolate:
        for(j=1;j<m-1;j++)
            a[i][j]=1;
    /* compute trace */
    for(i=0,j=0;i<n;i++)
        j+=a[i][i];
    printf("%d",j);
}
main()
{
    lonesome_cowboy();
    return 0;
}
