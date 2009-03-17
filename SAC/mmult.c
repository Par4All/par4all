void Matrix_Mult(int a1[4], int a2[4], int a3[4])
{
    int i = 0;
    int j = 0;
    int k = 0;
    for(i = 0; i < 2; i++) 
        for( j = 0; j < 2; j++)
            for( k = 0; k < 2; k++) 
                 a3[i*2+j] = a3[i*2+j] +  a1[i*2+k] * a2[k*2+j];
}
