void
SIMD_LOAD_CONSTANT_V4SI (int VEC[4], int X0, int X1, int X2, int X3)
{

    VEC[0] = X0;
    VEC[1] = X1;
    VEC[2] = X2;
    VEC[3] = X3;
}
void
SIMD_LOAD_V4SI (short VEC[4], short BASE[4])
{  VEC[0] = BASE[0];
    VEC[1] = BASE[1];
    VEC[2] = BASE[2];
    VEC[3] = BASE[3];
}
void
SIMD_STORE_GENERIC_V4SI (int VEC[4], int X1[1], int X2[1], int X3[1], int X4[1])
{

    X1 [0]= VEC[0];
    X2 [0]= VEC[1];
    X3 [0]= VEC[2];
    X4 [0]= VEC[3];
}
void
SIMD_LOAD_GENERIC_V4SI (short VEC[4], short X1, short X2, short X3, short X4)
{

    VEC[0] = X1;
    VEC[1] = X2;
    VEC[2] = X3;
    VEC[3] = X4;
}

void
SIMD_STORE_V4SI (short VEC[4], short BASE[4])
{  BASE[0] = VEC[0];
    BASE[1] = VEC[1];
    BASE[2] = VEC[2];
    BASE[3] = VEC[3];
}

void
SIMD_ADDW (short DEST[4], short SRC1[4], short SRC2[4])
{
    DEST[0] = SRC1[0] + SRC2[0];
    DEST[1] = SRC1[1] + SRC2[1];
    DEST[2] = SRC1[2] + SRC2[2];
    DEST[3] = SRC1[3] + SRC2[3];
}

void
SIMD_SUBW (short DEST[4], short SRC1[4], short SRC2[4])
{
    DEST[0] = SRC1[0] - SRC2[0];
    DEST[1] = SRC1[1] - SRC2[1];
    DEST[2] = SRC1[2] - SRC2[2];
    DEST[3] = SRC1[3] - SRC2[3];
}

void
SIMD_MULW (short DEST[4], short SRC1[4], short SRC2[4])
{
    DEST[0] = SRC1[0] * SRC2[0];
    DEST[1] = SRC1[1] * SRC2[1];
    DEST[2] = SRC1[2] * SRC2[2];
    DEST[3] = SRC1[3] * SRC2[3];
}
void
SIMD_GTW (int DEST[4], short SRC1[4], short SRC2[4])
{
    DEST[0] = SRC1[0] > SRC2[0];
    DEST[1] = SRC1[1] > SRC2[1];
    DEST[2] = SRC1[2] > SRC2[2];
    DEST[3] = SRC1[3] > SRC2[3];
}
