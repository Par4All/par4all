#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    int j;
    for(int i=1;i<argc;i++) {
        int n = atoi(argv[i]);
        int m = 0;
rof:        for(j=0;j<n;j++) {
            m+=j;
        }
        printf("%d\n",m);
    }
    return 0;
}
