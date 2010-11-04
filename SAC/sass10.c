#include <stdio.h>
main(int argc, char ** argv) {
    int a=1;
    if(argc>1) {
        a=2;printf("%d",a);
    }
    else {
        a=3;printf("%d",a);
    }
    return 0;
}
