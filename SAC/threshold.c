#include <stdio.h>
void threshold(float data[128],float val)
{
    int i;
    for(i=0;i<128;++i)
        if(data[i]>val)
            data[i]=val;
}

main()
{
    float data[128];
    int i;
    for(i=0;i<128;i++) data[i]=i/10;
    threshold(data,10.);
    for(i=0;i<128;i++)
        printf("%f",data[i]);
}
