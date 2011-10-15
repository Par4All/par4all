#include <sys/time.h>
#include <stdlib.h> // for NULL :p
int main() {
    struct timeval tv;
    gettimeofday(&tv,NULL);
    gettimeofday(&tv,NULL);
    return 0;
}
