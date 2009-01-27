#define LOGICAL int
#define DMAX(A,B) (A)>(B)?(A):(B)

int
PHI (LOGICAL L, int X1, int X2)
{
    return L ? X1 : X2;
}

void
SIMD_LOAD_V4SF (float VEC[4], float BASE[4])
{
    VEC[0] = BASE[0];
    VEC[1] = BASE[1];
    VEC[2] = BASE[2];
    VEC[3] = BASE[3];
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
SIMD_LOAD_CONSTANT_V4SF (float VEC[4], float X0, float X1, float X2, float X3)
{

    VEC[0] = X0;
    VEC[1] = X1;
    VEC[2] = X2;
    VEC[3] = X3;
}

void
SIMD_SAVE_V4SF (float VEC[4], float BASE[4])
{  BASE[0] = VEC[0];
    BASE[1] = VEC[1];
    BASE[2] = VEC[2];
    BASE[3] = VEC[3];
}

void
SIMD_SAVE_GENERIC_V4SF (float VEC[4], float X1[1], float X2[1],
        float X3[1], float X4[1])
{

    X1 [0]= VEC[0];
    X2 [0]= VEC[1];
    X3 [0]= VEC[2];
    X4 [0]= VEC[3];
}

void
SIMD_CMPGTPS (LOGICAL DEST[4], float SRC1[4], float SRC2[4])
{
    DEST[0] = SRC1[0] > SRC2[0];
    DEST[1] = SRC1[1] > SRC2[1];
    DEST[2] = SRC1[2] > SRC2[2];
    DEST[3] = SRC1[3] > SRC2[3];
}

void
SIMD_PHIPS (float DEST[4], LOGICAL COND[4], float SRC1[4], float SRC2[4])
{

    if (COND[0])
    {
        DEST[0] = SRC1[0];
    }
    else
    {
        DEST[0] = SRC2[0];
    }
    if (COND[1])
    {
        DEST[1] = SRC1[1];
    }
    else
    {
        DEST[1] = SRC2[1];
    }
    if (COND[2])
    {
        DEST[2] = SRC1[2];
    }
    else
    {
        DEST[2] = SRC2[2];
    }
    if (COND[3])
    {
        DEST[3] = SRC1[3];
    }
    else
    {
        DEST[3] = SRC2[3];
    }
}

void
SIMD_ADDPS (float DEST[4], float SRC1[4], float SRC2[4])
{
    DEST[0] = SRC1[0] + SRC2[0];
    DEST[1] = SRC1[1] + SRC2[1];
    DEST[2] = SRC1[2] + SRC2[2];
    DEST[3] = SRC1[3] + SRC2[3];
}

void
SIMD_SUBPS (float DEST[4], float SRC1[4], float SRC2[4])
{
    DEST[0] = SRC1[0] - SRC2[0];
    DEST[1] = SRC1[1] - SRC2[1];
    DEST[2] = SRC1[2] - SRC2[2];
    DEST[3] = SRC1[3] - SRC2[3];
}

void
SIMD_MULPS (float DEST[4], float SRC1[4], float SRC2[4])
{
    DEST[0] = SRC1[0] * SRC2[0];
    DEST[1] = SRC1[1] * SRC2[1];
    DEST[2] = SRC1[2] * SRC2[2];
    DEST[3] = SRC1[3] * SRC2[3];
}

void
SIMD_DIVPS (float DEST[4], float SRC1[4], float SRC2[4])
{
    DEST[0] = SRC1[0] / SRC2[0];
    DEST[1] = SRC1[1] / SRC2[1];
    DEST[2] = SRC1[2] / SRC2[2];
    DEST[3] = SRC1[3] / SRC2[3];
}

void
SIMD_MAXPS (float DEST[4], float SRC1[4], float SRC2[4])
{
    DEST[0] = DMAX (SRC1[0], SRC2[0]);
    DEST[1] = DMAX (SRC1[1], SRC2[1]);
    DEST[2] = DMAX (SRC1[2], SRC2[2]);
    DEST[3] = DMAX (SRC1[3], SRC2[3]);
}

void
SIMD_LOAD_CONSTANT_V2SI (int VEC[2], int HIGH, int LOW)
{

    VEC[0] = LOW;
    VEC[1] = HIGH;
}

void
SIMD_LOAD_V2SI (int VEC[2], int BASE[2])
{  
    VEC[0] = BASE[0];
    VEC[1] = BASE[1];
}

void
SIMD_LOAD_GENERIC_V2SI (int VEC[2], int X1, int X2)
{

    VEC[0] = X1;
    VEC[1] = X2;
}

void
SIMD_SAVE_V2SI (int VEC[2], int BASE[2])
{  BASE[0] = VEC[0];
    BASE[1] = VEC[1];
}

void
SIMD_SAVE_GENERIC_V2SI (int VEC[2], int X1[1], int X2[1])
{

    X1 [0]= VEC[0];
    X2 [0]= VEC[1];
}

void
SIMD_LOAD_V4SI (int VEC[1], int BASE[1])
{  VEC[0] = BASE[0];
    VEC[1] = BASE[1];
    VEC[2] = BASE[2];
    VEC[3] = BASE[3];
}

void
SIMD_LOAD_GENERIC_V4SI (int VEC[4], int X1, int X2, int X3, int X4)
{

    VEC[0] = X1;
    VEC[1] = X2;
    VEC[2] = X3;
    VEC[3] = X4;
}

void
SIMD_SAVE_V4SI (int VEC[4], int BASE[4])
{  BASE[0] = VEC[0];
    BASE[1] = VEC[1];
    BASE[2] = VEC[2];
    BASE[3] = VEC[3];
}

void
SIMD_SAVE_GENERIC_V4SI (int VEC[4], int X1[1], int X2[1], int X3[1], int X4[1])
{

    X1 [0]= VEC[0];
    X2 [0]= VEC[1];
    X3 [0]= VEC[2];
    X4 [0]= VEC[3];
}

void
SIMD_ADDD (int DEST[2], int SRC1[2], int SRC2[2])
{
    DEST[0] = SRC1[0] + SRC2[0];
    DEST[1] = SRC1[1] + SRC2[1];
}

void
SIMD_SUBD (int DEST[2], int SRC1[2], int SRC2[2])
{
    DEST[0] = SRC1[0] - SRC2[0];
    DEST[1] = SRC1[1] - SRC2[1];
}

void
SIMD_MULD (int DEST[2], int SRC1[2], int SRC2[2])
{
    DEST[0] = SRC1[0] * SRC2[0];
    DEST[1] = SRC1[1] * SRC2[1];
}

void
SIMD_LOAD_CONSTANT_V4HI (short VEC[4], int HIGH, int LOW)
{
    VEC[0] = (short) LOW;
    VEC[1] = (short) (LOW >> 2);
    VEC[2] = (short) HIGH;
    VEC[3] = (short) (HIGH >> 2);
}

void
SIMD_LOAD_V4HI (short VEC[4], short BASE[4])
{  VEC[0] = BASE[0];
    VEC[1] = BASE[1];
    VEC[2] = BASE[2];
    VEC[3] = BASE[3];
}

void
SIMD_LOAD_V4QI_TO_V4HI (short VEC[4], char BASE[4])
{  VEC[0] = BASE[0];
    VEC[1] = BASE[1];
    VEC[2] = BASE[2];
    VEC[3] = BASE[3];
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
SIMD_SAVE_V4HI (short VEC[4], short BASE[4])
{  BASE[0] = VEC[0];
    BASE[1] = VEC[1];
    BASE[2] = VEC[2];
    BASE[3] = VEC[3];
}

void
SIMD_SAVE_GENERIC_V4HI (short VEC[4], short X1[1],
        short X2[1], short X3[1], short X4[1])
{

    X1 [0]= VEC[0];
    X2 [0]= VEC[1];
    X3 [0]= VEC[2];
    X4 [0]= VEC[3];
}

void
SIMD_CMPGTW (LOGICAL DEST[4], short SRC1[4], short SRC2[4])
{
    DEST[0] = SRC1[0] > SRC2[0];
    DEST[1] = SRC1[1] > SRC2[1];
    DEST[2] = SRC1[2] > SRC2[2];
    DEST[3] = SRC1[3] > SRC2[3];
}

void
SIMD_PHIW (short DEST[4], LOGICAL COND[4], short SRC1[4], short SRC2[4])
{

    if (COND[0])
    {
        DEST[0] = SRC1[0];
    }
    else
    {
        DEST[0] = SRC2[0];
    }
    if (COND[1])
    {
        DEST[1] = SRC1[1];
    }
    else
    {
        DEST[1] = SRC2[1];
    }
    if (COND[2])
    {
        DEST[2] = SRC1[2];
    }
    else
    {
        DEST[2] = SRC2[2];
    }
    if (COND[3])
    {
        DEST[3] = SRC1[3];
    }
    else
    {
        DEST[3] = SRC2[3];
    }
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
SIMD_LOAD_CONSTANT_V8QI (char VEC[8], int HIGH, int LOW)
{
    VEC[0] = (char) LOW;
    VEC[1] = (char) (LOW >> 1);
    VEC[2] = (char) (LOW >> 2);
    VEC[3] = (char) (LOW >> 3);
    VEC[4] = (char) HIGH;
    VEC[5] = (char) (HIGH >> 1);
    VEC[6] = (char) (HIGH >> 2);
    VEC[7] = (char) (HIGH >> 3);
}

void
SIMD_LOAD_V8QI (char VEC[8], char BASE[8])
{  VEC[0] = BASE[0];
    VEC[1] = BASE[1];
    VEC[2] = BASE[2];
    VEC[3] = BASE[3];
    VEC[4] = BASE[4];
    VEC[5] = BASE[5];
    VEC[6] = BASE[6];
    VEC[7] = BASE[7];
}

void
SIMD_LOAD_GENERIC_V8QI (char VEC[8], char X1,
        char X2, char X3, char X4, char X5, char X6,
        char X7, char X8)
{
    VEC[0] = X1;
    VEC[1] = X2;
    VEC[2] = X3;
    VEC[3] = X4;
    VEC[4] = X5;
    VEC[5] = X6;
    VEC[6] = X7;
    VEC[7] = X8;
}

void
SIMD_SAVE_V8QI (char VEC[8], char BASE[8])
{  BASE[0] = VEC[0];
    BASE[1] = VEC[1];
    BASE[2] = VEC[2];
    BASE[3] = VEC[3];
    BASE[4] = VEC[4];
    BASE[5] = VEC[5];
    BASE[6] = VEC[6];
    BASE[7] = VEC[7];
}

void
SIMD_SAVE_GENERIC_V8QI (char VEC[8], char *X0,
        char X1[1], char X2[1], char X3[1], char X4[1], char X5[1],
        char X6[1], char X7[1])
{

    X0[0] = VEC[0];
    X1[0] = VEC[1];
    X2[0] = VEC[2];
    X3[0] = VEC[3];
    X4[0] = VEC[4];
    X5[0] = VEC[5];
    X6[0] = VEC[6];
    X7[0] = VEC[7];
}

void
SIMD_ADDB (char DEST[8], char SRC1[8], char SRC2[8])
{
    DEST[0] = SRC1[0] + SRC2[0];
    DEST[1] = SRC1[1] + SRC2[1];
    DEST[2] = SRC1[2] + SRC2[2];
    DEST[3] = SRC1[3] + SRC2[3];
    DEST[4] = SRC1[4] + SRC2[4];
    DEST[5] = SRC1[5] + SRC2[5];
    DEST[6] = SRC1[6] + SRC2[6];
    DEST[7] = SRC1[7] + SRC2[7];
}

void
SIMD_SUBB (char DEST[8], char SRC1[8], char SRC2[8])
{
    DEST[0] = SRC1[0] - SRC2[0];
    DEST[1] = SRC1[1] - SRC2[1];
    DEST[2] = SRC1[2] - SRC2[2];
    DEST[3] = SRC1[3] - SRC2[3];
    DEST[4] = SRC1[4] - SRC2[4];
    DEST[5] = SRC1[5] - SRC2[5];
    DEST[6] = SRC1[6] - SRC2[6];
    DEST[7] = SRC1[7] - SRC2[7];
}

void
SIMD_MULB (char DEST[8], char SRC1[8], char SRC2[8])
{

    DEST[0] = SRC1[0] * SRC2[0];
    DEST[1] = SRC1[1] * SRC2[1];
    DEST[2] = SRC1[2] * SRC2[2];
    DEST[3] = SRC1[3] * SRC2[3];
    DEST[4] = SRC1[4] * SRC2[4];
    DEST[5] = SRC1[5] * SRC2[5];
    DEST[6] = SRC1[6] * SRC2[6];
    DEST[7] = SRC1[7] * SRC2[7];
}

void
SIMD_MOVPS (float DEST[2], float SRC[2])
{
    DEST[0] = SRC[0];
    DEST[1] = SRC[1];
}

void
SIMD_MOVD (int DEST[2], int SRC[2])
{
    DEST[0] = SRC[0];
    DEST[1] = SRC[1];
}

void
SIMD_MOVW (short DEST[4], short SRC[4])
{
    DEST[0] = SRC[0];
    DEST[1] = SRC[1];
    DEST[2] = SRC[2];
    DEST[3] = SRC[3];
}

void
SIMD_MOVB (char DEST[8], char SRC[8])
{

    DEST[0] = SRC[0];
    DEST[1] = SRC[1];
    DEST[2] = SRC[2];
    DEST[3] = SRC[3];
    DEST[4] = SRC[4];
    DEST[5] = SRC[5];
    DEST[6] = SRC[6];
    DEST[7] = SRC[7];
}

void
SIMD_OPPPS (float DEST[2], float SRC[2])
{
    DEST[0] = -SRC[0];
    DEST[1] = -SRC[1];
}

void
SIMD_OPPD (int DEST[2], int SRC[2])
{
    DEST[0] = -SRC[0];
    DEST[1] = -SRC[1];
}

void
SIMD_OPPW (short DEST[4], short SRC[4])
{
    DEST[0] = -SRC[0];
    DEST[1] = -SRC[1];
    DEST[2] = -SRC[2];
    DEST[3] = -SRC[3];
}

void
SIMD_OPPB (char DEST[8], char SRC[8])
{
    DEST[0] = -SRC[0];
    DEST[1] = -SRC[1];
    DEST[2] = -SRC[2];
    DEST[3] = -SRC[3];
    DEST[4] = -SRC[4];
    DEST[5] = -SRC[5];
    DEST[6] = -SRC[6];
    DEST[7] = -SRC[7];
}

#undef LOGICAL
#undef DMAX
