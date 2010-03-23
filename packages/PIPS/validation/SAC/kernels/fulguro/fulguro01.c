
enum FLGR_TYPE
{
	FLGR_INT8,
	FLGR_BIT,
	FLGR_INT16,
	FLGR_INT32,
	FLGR_FLOAT32,
	FLGR_UINT8,
	FLGR_UINT16,
	FLGR_UINT32,
	FLGR_FLOAT64
};
typedef int fgINT32;

typedef struct {
  int * array; int spp;
  int length;} FLGR_Data1D;

typedef struct{
  FLGR_Data1D ** row;
  int size;
  int size_y;
  int size_x;
  enum FLGR_TYPE type;
} FLGR_Data2D;
void flgr1d_arith_add_fgINT32(FLGR_Data1D *datdest, FLGR_Data1D *data1, FLGR_Data1D *data2)
{
   int I_4;
   int I_3;
   int I_2;
   int I_1;
   int I_0;
   int j;
   fgINT32 *psrc1 = (fgINT32 *) data1->array;
   fgINT32 *psrc2 = (fgINT32 *) data2->array;
   fgINT32 *pdest = (fgINT32 *) datdest->array;
   fgINT32 v1;
   fgINT32 v2;
   fgINT32 result;
   int spp = datdest->spp;
   int length = datdest->length*spp;
   for(j = 0; j <= length-1; j += 1) {
      I_3 = psrc1[j];
      I_1 = I_3;
      v1 = I_1;
      I_4 = psrc2[j];
      I_2 = I_4;
      v2 = I_2;
      I_0 = v1+v2;
      result = I_0;
      pdest[j] = result;
   }
   return;
}

int test()
{
    FLGR_Data1D d1={0,sizeof(fgINT32),10},d2={0,sizeof(fgINT32),10},dest={0,sizeof(fgINT32),10};
    d1.array=malloc(10*sizeof(fgINT32));
    d2.array=malloc(10*sizeof(fgINT32));
    dest.array=malloc(10*sizeof(fgINT32));
    flgr1d_arith_add_fgINT32(&dest,&d1,&d2);
}
