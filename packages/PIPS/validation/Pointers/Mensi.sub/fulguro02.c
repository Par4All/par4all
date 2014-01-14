#include<assert.h>
#include<stdio.h>
#include<stdlib.h>
//#include <png.h>
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

typedef void* png_bytep;




void fulguro02(FLGR_Data2D *img)
{

  int i;
  png_bytep *row_pointers = NULL;
  int  size_y = img->size_y;
  row_pointers = (png_bytep*) malloc(sizeof(png_bytep)*size_y);
  
  for(i=0;i<size_y;i++) {
    row_pointers[i]=(png_bytep) img->array[i];
  }
}
