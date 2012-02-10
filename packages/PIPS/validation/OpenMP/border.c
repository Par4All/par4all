#include <stddef.h>
#include <stdlib.h>
#include <limits.h>
#ifndef __PIPS__
#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)>(b))?(b):(a))
#endif

void img_erode(size_t w, size_t n, size_t m, char out[n][m], char in[n][m]) {
    for(size_t i= w; i<n-w ; i++)
        for(size_t j= w; j<m-w ; j++) {
            out[i][j] = CHAR_MIN ;
            for( size_t k = -w ; k <= w ;k++)
                for( size_t l = -w ; l <= w ;l++)
                    out[i][j] = MIN(out[i][j],in[i+k][j+k]);
        }
}
void img_dilate(size_t w, size_t n, size_t m, char out[n][m], char in[n][m]) {
    for(size_t i= w; i<n-w ; i++)
        for(size_t j= w; j<m-w ; j++) {
            out[i][j] = CHAR_MAX ;
            for( size_t k = -w ; k <= w ;k++)
                for( size_t l = -w ; l <= w ;l++)
                    out[i][j] = MAX(out[i][j],in[i+k][j+k]);
        }
}
void img_diff(size_t n, size_t m, char out[n][m], char self[n][m], char other[n][m]) {
    for(size_t i= 0; i<n ; i++)
        for(size_t j= 0; j<m ; j++)
            out[i][j] = self[i][j] - other[i][j] ;
}


void img_border(size_t w, size_t n, size_t m, char out[n][m], char in[n][m]) {

    char (*tmp0)[n][m] = malloc(sizeof(char)*n*m);
    img_erode(w, n, m, *tmp0, in);

    char (*tmp1)[n][m] = malloc(sizeof(char)*n*m);
    img_dilate(w, n, m, *tmp1, in);

    img_diff(n, m, out, *tmp1, *tmp0);
    free(tmp0);
    free(tmp1);
}
