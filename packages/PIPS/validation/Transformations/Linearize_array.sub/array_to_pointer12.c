#include <stdio.h>

int duck(int riri[10], int fifi[2][3], int size, int loulou[1][size][6])
{
    int* zaza = fifi[1];
    return riri[2] = zaza[1] +loulou[0][1][3];
}

int main()
{
    int riri[10] = { 0,1,2,3,4,5,6,7,8,9};
    int fifi[2][3] = {
        { 10,11,12},
        {13,14,15}
    };
    int size=2;
    int loulou[1][size][6];
    int i,j,k=16;
    for(i=0;i<size;i++)
        for(j=0;j<6;j++)
            loulou[0][i][j]=k++;
    printf("%d\n",duck(riri,fifi,size,loulou));
    return 0;
}
