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




void flgr1d_arith_add_fgINT32(FLGR_Data1D *data1, FLGR_Data1D *data2)
{
  
   fgINT32 *psrc1 = (fgINT32 *) data1->array;
   fgINT32 *psrc2 = (fgINT32 *) data2->array;
   
   return;
}



