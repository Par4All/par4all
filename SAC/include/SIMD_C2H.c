void
SIMD_LOAD_V2HI_TO_V2SI(int VEC[2], short BASE[2])
{
    VEC[0] = BASE[0];
    VEC[1] = BASE[1];
}
void
SIMD_STORE_V2HI_TO_V2SI(short VEC[2], int BASE[2])
{
    VEC[0] = BASE[0];
    VEC[1] = BASE[1];
}
void
SIMD_LOAD_V4QI_TO_V4HI (short VEC[4], char BASE[4])
{
    VEC[0] = BASE[0];
    VEC[1] = BASE[1];
    VEC[2] = BASE[2];
    VEC[3] = BASE[3];
}
void
SIMD_LOAD_GENERIC_V2HI (short VEC[4], short X1, short X2)
{

    VEC[0] = X1;
    VEC[1] = X2;
}

void
SIMD_LOAD_GENERIC_V4HI (short VEC[4], short X1, short X2, short X3, short X4)
{

    VEC[0] = X1;
    VEC[1] = X2;
    VEC[2] = X3;
    VEC[3] = X4;
}
void
SIMD_STORE_GENERIC_V2HI (short VEC[4], short X1[1], short X2[1])
{

    X1 [0]= VEC[0];
    X2 [0]= VEC[1];
}
void
SIMD_STORE_V2HI (short VEC[2], short BASE[2])
{

    BASE [0]= VEC[0];
    BASE [1]= VEC[1];
}
void
SIMD_STORE_GENERIC_V4HI (short VEC[4], short X1[1],
        short X2[1], short X3[1], short X4[1])
{

    X1 [0]= VEC[0];
    X2 [0]= VEC[1];
    X3 [0]= VEC[2];
    X4 [0]= VEC[3];
}
void
SIMD_STORE_V4HI (short VEC[4], short BASE[4])
{  BASE[0] = VEC[0];
    BASE[1] = VEC[1];
    BASE[2] = VEC[2];
    BASE[3] = VEC[3];
}

void
SIMD_LOAD_V2HI (short VEC[2], short BASE[2])
{
    VEC[0] = BASE[0];
    VEC[1] = BASE[1];
}

void
SIMD_LOAD_V4HI (short VEC[4], short BASE[4])
{  VEC[0] = BASE[0];
    VEC[1] = BASE[1];
    VEC[2] = BASE[2];
    VEC[3] = BASE[3];
}

void
SIMD_LOAD_GENERIC_V2SI (int VEC[2], int X1, int X2)
{

    VEC[0] = X1;
    VEC[1] = X2;
}
void
SIMD_STORE_GENERIC_V2SI (int VEC[2], int X1[1], int X2[1])
{

    X1 [0]= VEC[0];
    X2 [0]= VEC[1];
}
void
SIMD_SETD (int DEST[1], short SRC[2])
{
    DEST[0] = SRC[0] | (SRC[1] << 4);
}

void
SIMD_SETW (int DEST[1], char SRC[4])
{
    DEST[0] = SRC[0] | (SRC[1] << 2) | (SRC[2] << 4) | (SRC[3] << 6) ;
}

