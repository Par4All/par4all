#include <flgrCoreDispatch.h>
#include <flgrCoreSlideWindow.h>

#define FLGR_MACRO_RASTER_SLIDE_WINDOW_2D(dtype,get_nhb_op)	\
  int i,j,w,h,spp = imgsrc->spp;				\
  dtype *vector_array;						\
  dtype *data_array;						\
  FLGR_Vector *result;						\
  FLGR_Data2D *nhbEven,*nhbOdd,*nhbrs;				\
  FLGR_NhbBox2D *extr;						\
								\
  					\
								\
  w=imgsrc->size_x;						\
  h=imgsrc->size_y;						\
								\
  result = flgr_vector_create(imgsrc->spp,imgsrc->type);	\
								\
  vector_array = (dtype *) result->array;			\
								\
  extr = flgr2d_create_neighbor_box(nhb);			\
								\
  nhbEven=flgr2d_create_neighborhood_from(nhb);			\
  nhbOdd=flgr2d_create_neighborhood_from(nhb);			\
								\
  flgr2d_fill_nhbs_for_6_connexity(nhbEven,nhbOdd,nhb,nhb_sym);	\
								\
  if(imgdest==imgsrc) {						\
    flgr2d_apply_raster_scan_method_##dtype(nhbOdd);		\
    flgr2d_apply_raster_scan_method_##dtype(nhbEven);		\
  }								\
								\
  for(i=0 ; i<h; i++) {						\
								\
    data_array = (dtype *) (imgdest->array[i]);			\
								\
    for(j=0 ; j<w ; j++) {					\
								\
      nhbrs = (((i%2)==1) ? nhbOdd : nhbEven);			\
								\
      get_nhb_op##_##dtype(extr,imgsrc,nhbrs,i,j);		\
								\
      (*computeNhb)(result,extr);				\
      								\
      flgr_set_data_array_vector_##dtype(data_array,		\
					 vector_array,		\
					 spp,j);		\
    }								\
  }								\
								\
  flgr2d_destroy(nhbOdd);					\
  flgr2d_destroy(nhbEven);					\
  flgr2d_destroy_neighbor_box(extr);				\
  flgr_vector_destroy(result);					\
								\
  return

void flgr2d_raster_slide_window_fgUINT16(FLGR_Data2D *imgdest, FLGR_Data2D *imgsrc, FLGR_Data2D *nhb, int nhb_sym,
					  FLGR_ComputeNhb2D computeNhb) {
  FLGR_MACRO_RASTER_SLIDE_WINDOW_2D(fgUINT16,flgr2d_get_neighborhood);
}
