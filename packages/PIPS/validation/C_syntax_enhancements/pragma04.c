#define N 100

void pragma04( int a[N] ) {

    int i, j, k;

#pragma scop

    k = 1; for(i = 0; i <= 99; i += 1) {

        a[i] = k;

    } for(i = 0; i <= 99; i += 1) {

        k = 2 * k;

    }

#pragma endscop

}

