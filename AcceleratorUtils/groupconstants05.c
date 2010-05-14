#include <stdio.h>
int main()
{
    int a[2][4]={ {0,1,2,3},{4,5,6,7}} ,b=3,i;
ou_est_charlie:
    for(i=0;i<4;i++)
        a[0][i]=1;
    printf("%d",a[0][1]);
    return 0;
}
