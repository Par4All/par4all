/* The code seems buggy because dat contains uninitialized pointers
 * that are dereferenced, "dat->row;"
 *
 * The call to flgrmalloc() should be uncommented and its source code
 * added.
 */

#include<stdlib.h>

typedef struct {
  int dim;
  int size_struct;
  void* array;
  void* array_phantom;
}FLGR_Data1D;

typedef struct {
  int dim;
  int size_struct;
  int link_overlap;
  int size_y;
  int size_x;
  FLGR_Data1D **row;
  void** array;
} FLGR_Data2D;

FLGR_Data2D *flgr2d_create_pixmap_link(FLGR_Data2D *datain,
		int partsNumber, int partIndex, int overlapSize)
{
  FLGR_Data2D *dat;
  int i,k,nbrow,startin;
  /* dat = (FLGR_Data2D*) flgr_malloc(sizeof(FLGR_Data2D)); */
  dat = (FLGR_Data2D*) malloc(sizeof(FLGR_Data2D));
  dat->link_overlap  = overlapSize;

  for(i=0 ; i<nbrow-overlapSize ; i++) {
    dat->row[i] = datain->row[i];
    dat->row[i]->dim = i;
    dat->array[i] = dat->row[i]->array;
  }
}
