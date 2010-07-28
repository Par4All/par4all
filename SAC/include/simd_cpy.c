void SIMD_CPY(float a[4],float b[4])
{
    a[0]=b[0];
    a[1]=b[1];
    a[2]=b[2];
    a[3]=b[3];
}
void
SIMD_LOAD_GENERIC_V4SF (float VEC[4], float X0, float X1, float X2, float X3)
{
    VEC[0] = X0;
    VEC[1] = X1;
    VEC[2] = X2;
    VEC[3] = X3;
}
void
SIMD_STORE_V4SF (float VEC[4], float BASE[4])
{  
    BASE[0] = VEC[0];
    BASE[1] = VEC[1];
    BASE[2] = VEC[2];
    BASE[3] = VEC[3];
}
void
SIMD_STORE_GENERIC_V4SF (float VEC[4], float X1[1], float X2[1],
        float X3[1], float X4[1])
{

    X1 [0]= VEC[0];
    X2 [0]= VEC[1];
    X3 [0]= VEC[2];
    X4 [0]= VEC[3];
}
