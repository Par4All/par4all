#include <sys/time.h>

int main() {
    int n=100;
    int a[n];
    struct timeval tic, toc;
    for(int i=0;i<n;i++) {
        gettimeofday(&tic,0);
        for(int j=0;j<n;j++){
            a[j]=j;
        }
        gettimeofday(&toc,0);
    }
}
