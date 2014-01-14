#include<assert.h>
#include<stdio.h>
#include<stdlib.h>
typedef struct {
  int size_struct;
  void* array;
  void* array_phantom;
  int ref2d;
}FLGR_Data1D;
typedef struct {
  int size_struct;
  int link_overlap;
  int size_y;
  int size_x;
  FLGR_Data1D **row;
  void** array;
}FLGR_Data2D;


FLGR_Data2D *flgr2d_create_pixmap_link(FLGR_Data2D *datain, int partsNumber,
				       int partIndex, 
				       int overlapSize) {
  FLGR_Data2D *dat;
  int i,nbrow;
  dat = (FLGR_Data2D*) malloc(sizeof(FLGR_Data2D));
  dat->link_overlap = overlapSize;
  dat->size_struct   = sizeof(FLGR_Data2D);
  dat->size_x        = datain->size_x;
  nbrow = datain->size_y/partsNumber + overlapSize;
  dat->row    = (FLGR_Data1D**) malloc( (nbrow+16) * sizeof(FLGR_Data1D*));
  dat->array  = (void**)malloc( (nbrow+16) * sizeof(void*));
  dat->size_y = nbrow;
  for(i=0 ; i<nbrow-overlapSize ; i++) {
    dat->row[i] = datain->row[i];
    dat->row[i]->ref2d = i;
    dat->array[i] = dat->row[i]->array;
  }
  return dat;
}
