#include <stdio.h>
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

static __inline__ fgINT32 flgr_defop_add_fgINT32(fgINT32 a,fgINT32 b) {
  return a+b;
}

static void flgr_set_array_fgINT32(fgINT32 *array, int pos, fgINT32 value)
{
   array[pos] = value;
}

static void flgr1d_set_data_array_fgINT32(fgINT32 *array, int pos, fgINT32 value)
{
   flgr_set_array_fgINT32(array, pos, value);
}

static fgINT32 flgr_get_array_fgINT32(fgINT32 *array, int pos)
{
  return array[pos];
}

static __inline__ fgINT32 flgr1d_get_data_array_fgINT32(fgINT32* array, int pos) {
  return flgr_get_array_fgINT32(array,pos);
}

typedef enum _FLGR_Ret {
	FLGR_RET_NULL_OBJECT,
	FLGR_RET_OK,
	FLGR_RET_TYPE_UNKNOWN,
} FLGR_Ret;


void flgr1d_arith_add_fgINT32(FLGR_Data1D *datdest, FLGR_Data1D *data1, FLGR_Data1D *data2)
{
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
      v1 = flgr1d_get_data_array_fgINT32(psrc1, j);
      v2 = flgr1d_get_data_array_fgINT32(psrc2, j);
      result = flgr_defop_add_fgINT32(v1, v2);
      flgr1d_set_data_array_fgINT32(pdest, j, result);
   }
   return;
}

void flgr2d_arith_add_fgINT32(FLGR_Data2D *datdest, FLGR_Data2D *data1, FLGR_Data2D *data2)
{
   FLGR_Data1D **pdest = datdest->row;
   FLGR_Data1D **p1 = data1->row;
   FLGR_Data1D **p2 = data2->row;
   int i;
   i = 0;
   while (i<data1->size_y) {
      flgr1d_arith_add_fgINT32(*pdest, *p1, *p2);
      (i++, pdest++, p1++, p2++);
   }
   return;
}

