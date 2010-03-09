#include "../Transformations/fulguro-included.h"

void flgr2d_raster_slide_window_fgUINT16(FLGR_Data2D *imgdest, FLGR_Data2D *imgsrc, FLGR_Data2D *nhb, int nhb_sym,
					  FLGR_ComputeNhb2D computeNhb) {
  FLGR_MACRO_RASTER_SLIDE_WINDOW_2D(fgUINT16,flgr2d_get_neighborhood);
}
