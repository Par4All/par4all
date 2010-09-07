void SIMD_ADDD(short a[8],short b[8],short c[8])
{
    a[0]=b[0]+c[0];
    a[1]=b[1]+c[1];
    a[2]=b[2]+c[2];
    a[3]=b[3]+c[3];
    a[4]=b[4]+c[4];
    a[5]=b[5]+c[5];
    a[6]=b[6]+c[6];
    a[7]=b[7]+c[7];
}
void SIMD_MULD(short a[8],short b[8],short c[8])
{
    a[0]=b[0]*c[0];
    a[1]=b[1]*c[1];
    a[2]=b[2]*c[2];
    a[3]=b[3]*c[3];
    a[4]=b[4]*c[4];
    a[5]=b[5]*c[5];
    a[6]=b[6]*c[6];
    a[7]=b[7]*c[7];
}
void SIMD_LOAD_GENERIC_V8HI(short a[8], short a0[1],short a1[1],short a2[1],short a3[1],short a4[1],short a5[1],short a6[1],short a7[1])
{
    a[0]=a0[0];
    a[1]=a1[0];
    a[2]=a2[0];
    a[3]=a3[0];
    a[4]=a4[0];
    a[5]=a5[0];
    a[6]=a6[0];
    a[7]=a7[0];
}
void SIMD_STORE_GENERIC_V8HI(short a[8], short a0[1],short a1[1],short a2[1],short a3[1],short a4[1],short a5[1],short a6[1],short a7[1])
{
    a0[0]=a[0];
    a1[0]=a[1];
    a2[0]=a[2];
    a3[0]=a[3];
    a4[0]=a[4];
    a5[0]=a[5];
    a6[0]=a[6];
    a7[0]=a[7];
}
