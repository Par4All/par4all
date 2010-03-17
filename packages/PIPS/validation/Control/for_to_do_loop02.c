#include <stdio.h>
main()
{
    int i,j,n,BlockEnd=3;
    for(i=1;i<16; i<<=2)
        for ( j=i, n=0; n < BlockEnd; j++, n++ )
            printf("%d",j);
}
