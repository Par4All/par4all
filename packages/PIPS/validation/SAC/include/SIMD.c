#define LOGICAL int
#define DMAX(A,B) (A)>(B)?(A):(B)
void SIMD_SHUFFLE_V4SF(float a[4], float b [4], int p0, int p1, int p2, int p3) {
    a[p0]=b[0];
    a[p1]=b[1];
    a[p2]=b[2];
    a[p3]=b[3];
}
void SIMD_LOAD_BROADCAST_V4SF(float a[4],float f)
{
    a[0]=f;
    a[1]=f;
    a[2]=f;
    a[3]=f;
} 
void SIMD_LOAD_BROADCAST_V2DF(double a[2],double d)
{
    a[0]=d;
    a[1]=d;
} 
void SIMD_LOAD_V4HI_TO_V4SF(float a[4], unsigned int b[4])
{
    a[0]=b[0];
    a[1]=b[1];
    a[2]=b[2];
    a[3]=b[3];
}
void SIMD_LOAD_V4SI_TO_V4SF(float a[4], int b[4])
{
    a[0]=b[0];
    a[1]=b[1];
    a[2]=b[2];
    a[3]=b[3];
}
void SIMD_LOAD_V4HI_TO_V4SI(int a[4],unsigned int b[4])
{
    a[0]=b[0];
    a[1]=b[1];
    a[2]=b[2];
    a[3]=b[3];
}
void SIMD_STORE_V8HI_TO_V8SI(short a[8],unsigned short b[8])
{
    a[0]=b[0];
    a[1]=b[1];
    a[2]=b[2];
    a[3]=b[3];
    a[4]=b[4];
    a[5]=b[5];
    a[6]=b[6];
    a[7]=b[7];
}
void SIMD_LOAD_V8SI_TO_V8HI(unsigned short a[8],short b[8])
{
    a[0]=b[0];
    a[1]=b[1];
    a[2]=b[2];
    a[3]=b[3];
    a[4]=b[4];
    a[5]=b[5];
    a[6]=b[6];
    a[7]=b[7];
}
void SIMD_STORE_V4SF_TO_V4SI(float a[4], int b[4])
{
    b[0]=a[0];
    b[1]=a[1];
    b[2]=a[2];
    b[3]=a[3];
}
void SIMD_STORE_V2DF_TO_V2SF(double a[2],float b[2])
{
    b[0]=a[0];
    b[1]=a[1];
}
void SIMD_LOAD_V2SF_TO_V2DF(double a[2],float b[2])
{
    a[0]=b[0];
    a[1]=b[1];
}

int
PHI (LOGICAL L, int X1, int X2)
{
    return L ? X1 : X2;
}

void
SIMD_PHIW(int R[4], LOGICAL L[4], int X1[4], int X2[4])
{
    int i;
    for (i=0;i<2;i++)
        R[i]=L[i]?X1[i]:X2[i];
}

void
SIMD_GTD(int R[4], int X1[4], int X2[4])
{
    int i;
    for (i=0;i<4;i++)
        R[i]=X1[i]>X2[i];
}
void
SIMD_LOAD_V4SI (int VEC[4], int BASE[4])
{
    VEC[0] = BASE[0];
    VEC[1] = BASE[1];
    VEC[2] = BASE[2];
    VEC[3] = BASE[3];
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
SIMD_LOAD_V2DF (double VEC[2], double BASE[2])
{
    VEC[0] = BASE[0];
    VEC[1] = BASE[1];
}

void
SIMD_LOAD_GENERIC_V2DF (double VEC[2], double X0, double X1)
{
    VEC[0] = X0;
    VEC[1] = X1;
}
void
SIMD_LOAD_GENERIC_V4SI (int VEC[4], int X0, int X1, int X2, int X3)
{
    VEC[0] = X0;
    VEC[1] = X1;
    VEC[2] = X2;
    VEC[3] = X3;
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
SIMD_LOAD_CONSTANT_V4SI (int VEC[4], int X0, int X1, int X2, int X3)
{

    VEC[0] = X0;
    VEC[1] = X1;
    VEC[2] = X2;
    VEC[3] = X3;
}

void
SIMD_STORE_V4SI (int VEC[4], int BASE[4])
{  
    BASE[0] = VEC[0];
    BASE[1] = VEC[1];
    BASE[2] = VEC[2];
    BASE[3] = VEC[3];
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
SIMD_STORE_V2DF (double VEC[2], double BASE[2])
{  
    BASE[0] = VEC[0];
    BASE[1] = VEC[1];
}

void
SIMD_STORE_MASKED_V4SF(float VEC[4], float BASE[3])
{  
    BASE[0] = VEC[0];
    BASE[1] = VEC[1];
    BASE[2] = VEC[2];
}


void
SIMD_STORE_GENERIC_V2DF (double VEC[2], double X1[1], double X2[1])
{

    X1 [0]= VEC[0];
    X2 [0]= VEC[1];
}
void
SIMD_STORE_GENERIC_V4SI (int VEC[4], int X1[1], int X2[1],
        int X3[1], int X4[1])
{

    X1 [0]= VEC[0];
    X2 [0]= VEC[1];
    X3 [0]= VEC[2];
    X4 [0]= VEC[3];
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

void
SIMD_GTPS (LOGICAL DEST[4], float SRC1[4], float SRC2[4])
{
    DEST[0] = SRC1[0] > SRC2[0];
    DEST[1] = SRC1[1] > SRC2[1];
    DEST[2] = SRC1[2] > SRC2[2];
    DEST[3] = SRC1[3] > SRC2[3];
}
void
SIMD_GTPD (LOGICAL DEST[2], double SRC1[2], double SRC2[2])
{
    DEST[0] = SRC1[0] > SRC2[0];
    DEST[1] = SRC1[1] > SRC2[1];
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
SIMD_UMINPS (float DEST[4], float SRC1[4])
{
    DEST[0] =  - SRC1[0];
    DEST[1] =  - SRC1[1];
    DEST[2] =  - SRC1[2];
    DEST[3] =  - SRC1[3];
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
SIMD_DIVPD (double DEST[2], double SRC1[2], double SRC2[2])
{
    DEST[0] = SRC1[0] / SRC2[0];
    DEST[1] = SRC1[1] / SRC2[1];
}
void
SIMD_MULPD (double DEST[2], double SRC1[2], double SRC2[2])
{
    DEST[0] = SRC1[0] * SRC2[0];
    DEST[1] = SRC1[1] * SRC2[1];
}
#ifdef WITH_TRIGO
void
SIMD_SINPD (double DEST[2], double SRC1[2])
{
    DEST[0] = COS(SRC1[0]);
    DEST[1] = COS(SRC1[1]);
}
void
SIMD_COSPD (double DEST[2], double SRC1[2])
{
    DEST[0] = SIN(SRC1[0]);
    DEST[1] = SIN(SRC1[1]);
}
#endif
void
SIMD_ADDPD (double DEST[2], double SRC1[2], double SRC2[2])
{
    DEST[0] = SRC1[0] + SRC2[0];
    DEST[1] = SRC1[1] + SRC2[1];
}
void
SIMD_SUBPD (double DEST[2], double SRC1[2], double SRC2[2])
{
    DEST[0] = SRC1[0] - SRC2[0];
    DEST[1] = SRC1[1] - SRC2[1];
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
SIMD_LOAD_V2SI_TO_V2SF(int VEC[2], float TO[2])
{
    TO[0]=VEC[0];
    TO[1]=VEC[1];
}
void
SIMD_STORE_V2SI_TO_V2SF(int TO[2], float VEC[2])
{
    TO[0]=VEC[0];
    TO[1]=VEC[1];
}


void
SIMD_LOAD_CONSTANT_V2SF (float VEC[2], float HIGH, float LOW)
{

    VEC[0] = LOW;
    VEC[1] = HIGH;
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
SIMD_STORE_V2SI (int VEC[2], int BASE[2])
{  BASE[0] = VEC[0];
    BASE[1] = VEC[1];
}

void
SIMD_STORE_GENERIC_V2SI (int VEC[2], int X1[1], int X2[1])
{

    X1 [0]= VEC[0];
    X2 [0]= VEC[1];
}


void
SIMD_STORE_V2DI (int VEC[2], int BASE[2])
{  BASE[0] = VEC[0];
    BASE[1] = VEC[1];
}

void
SIMD_ADDW (short DEST[8], short SRC1[8], short SRC2[8])
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
SIMD_SUBW (short DEST[8], short SRC1[8], short SRC2[8])
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
SIMD_MULW (short DEST[8], short SRC1[8], short SRC2[8])
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
SIMD_DIVW (short DEST[8], short SRC1[8], short SRC2[8])
{
    DEST[0] = SRC1[0] / SRC2[0];
    DEST[1] = SRC1[1] / SRC2[1];
    DEST[2] = SRC1[2] / SRC2[2];
    DEST[3] = SRC1[3] / SRC2[3];
    DEST[4] = SRC1[4] / SRC2[4];
    DEST[5] = SRC1[5] / SRC2[5];
    DEST[6] = SRC1[6] / SRC2[6];
    DEST[7] = SRC1[7] / SRC2[7];
}

void SIMD_LOAD_BROADCAST_V8HI(short vec[8], short val) {
    vec[0] = val;
    vec[1] = val;
    vec[2] = val;
    vec[3] = val;
    vec[4] = val;
    vec[5] = val;
    vec[6] = val;
    vec[7] = val;
}
void
SIMD_LOAD_GENERIC_V8HI(short VEC[8], short BASE0, short BASE1, short BASE2, short BASE3, short BASE4, short BASE5, short BASE6, short BASE7)
{  
    VEC[0] = BASE0;
    VEC[1] = BASE1;
    VEC[2] = BASE2;
    VEC[3] = BASE3;
    VEC[4] = BASE4;
    VEC[5] = BASE5;
    VEC[6] = BASE6;
    VEC[7] = BASE7;
}

void
SIMD_LOAD_V8HI (short VEC[8], short BASE[8])
{  
    VEC[0] = BASE[0];
    VEC[1] = BASE[1];
    VEC[2] = BASE[2];
    VEC[3] = BASE[3];
    VEC[4] = BASE[4];
    VEC[5] = BASE[5];
    VEC[6] = BASE[6];
    VEC[7] = BASE[7];
}

void
SIMD_LOAD_V4QI_TO_V4HI (short VEC[4], char BASE[4])
{  VEC[0] = BASE[0];
    VEC[1] = BASE[1];
    VEC[2] = BASE[2];
    VEC[3] = BASE[3];
}


void
SIMD_STORE_V8HI (short VEC[8], short BASE[8])
{  
    BASE[0] = VEC[0];
    BASE[1] = VEC[1];
    BASE[2] = VEC[2];
    BASE[3] = VEC[3];
    BASE[4] = VEC[4];
    BASE[5] = VEC[5];
    BASE[6] = VEC[6];
    BASE[7] = VEC[7];
}



void
SIMD_PHID (int DEST[4], LOGICAL COND[4], int SRC1[4], int SRC2[4])
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
SIMD_ADDD (int DEST[4], int SRC1[4], int SRC2[4])
{
    DEST[0] = SRC1[0] + SRC2[0];
    DEST[1] = SRC1[1] + SRC2[1];
    DEST[2] = SRC1[2] + SRC2[2];
    DEST[3] = SRC1[3] + SRC2[3];
}

void
SIMD_SUBD (int DEST[4], int SRC1[4], int SRC2[4])
{
    DEST[0] = SRC1[0] - SRC2[0];
    DEST[1] = SRC1[1] - SRC2[1];
    DEST[2] = SRC1[2] - SRC2[2];
    DEST[3] = SRC1[3] - SRC2[3];
}

void
SIMD_MULD (int DEST[4], int SRC1[4], int SRC2[4])
{
    DEST[0] = SRC1[0] * SRC2[0];
    DEST[1] = SRC1[1] * SRC2[1];
    DEST[2] = SRC1[2] * SRC2[2];
    DEST[3] = SRC1[3] * SRC2[3];
}
void
SIMD_DIVD (int DEST[4], int SRC1[4], int SRC2[4])
{
    DEST[0] = SRC1[0] / SRC2[0];
    DEST[1] = SRC1[1] / SRC2[1];
    DEST[2] = SRC1[2] / SRC2[2];
    DEST[3] = SRC1[3] / SRC2[3];
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
SIMD_STORE_V8QI (char VEC[8], char BASE[8])
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
SIMD_STORE_GENERIC_V8QI (char VEC[8], char *X0,
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

void
SIMD_SETPS (float DEST[4], float SRC[4])
{
    DEST[0] = SRC[0];
    DEST[1] = SRC[1];
    DEST[2] = SRC[2];
    DEST[4] = SRC[4];
}
void
SIMD_SETPD (double DEST[2], double SRC[2])
{
    DEST[0] = SRC[0];
    DEST[1] = SRC[1];
}

void
SIMD_SETD (int DEST[2], int SRC[2])
{
    DEST[0] = SRC[0];
    DEST[1] = SRC[1];
    DEST[2] = SRC[2];
    DEST[3] = SRC[3];
}

void
SIMD_SETW (short DEST[4], short SRC[4])
{
    DEST[0] = SRC[0];
    DEST[1] = SRC[1];
}

void
SIMD_SETB (char DEST[8], char SRC[8])
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
SIMD_LOAD_CONSTANT_V2DF(double vec[2],double v0,double v1)
{
    vec[0]=v0;
    vec[1]=v1;
}

void
SIMD_XOR_V4SF(float vec[4])
{
    vec[0]=0.;
    vec[1]=0.;
    vec[2]=0.;
    vec[3]=0.;
}

void
SIMD_ADDREDUCE_V4SF(float val[1],float vec[4]){
    val[0]=vec[0]+vec[1]+vec[2]+vec[3];
}


#undef LOGICAL
#undef DMAX
