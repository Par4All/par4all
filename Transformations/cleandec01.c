#include<stdio.h>
void c_clean_declarations01(int x)
{
    /* Hide the formal parameter */
    //int x; << compilation of this failed with gcc
    int y;

    if(y) {
        /* This third x is useless, but not its initialization */
        int x = y++;
        int z;

        y = 1;
    }
    else {
        int x = 0; /* This fourth x is really useless */
        int z = y--;

        y = -1;
    }
    fprintf(stderr, "x=%dn", x);
} 
