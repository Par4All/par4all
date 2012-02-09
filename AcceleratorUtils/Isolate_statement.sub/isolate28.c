void pain( int rdata[100]) {
    int i;
holy:for(i=0;i<100;i++)
        rdata[0]+=rdata[i];
}
#include <stdio.h>
int main() {
    int rdata[100];
    for(int i=0;i<100;i++) rdata[i]=i;
    pain(rdata);
    printf("%d\n",rdata[0]);
    return 0;
}


