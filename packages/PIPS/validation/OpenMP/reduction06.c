#include <math.h>

int test(int bSize, int matOut[bSize][bSize]) {
    int out=0., norm;
    for(int i = 0; i <= bSize-1; i += 1) {
        norm = 0;
        for(int j = 0; j <= i-1; j += 1)
            norm += matOut[i][j]*matOut[i][j]+matOut[i][j]*matOut[i][j];
    }
    out+=norm;
}
