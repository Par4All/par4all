#include <stdio.h>
void lonesome_cowboy()
{
    int i,a[12],j=0;
    for(i=0;i<12;i++)a[i]=1;
isolate:
    for(i=1;i<11;i++)
        j+=a[i];
    printf("%d",j);
}
main()
{
    lonesome_cowboy();
    return 0;
}
