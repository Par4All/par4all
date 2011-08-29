/***************************************************************
 * Author : Christophe Clienti
 * Date : 10/22/2008
 * Copyright (c) 2008 Thales Research & Technologies (FR)
 ***************************************************************/


#include <freiaDebug.h>
#include <freiaCommon.h>
#include <freiaAtomicOpLinear.h>
#include <freiaComplexOpMeasure.h>




freia_status freia_cipo_fast_correlation(freia_data2d *imout, freia_data2d *imin, freia_data2d *imref,  uint32_t horizon)
{
  uint32_t m,n;
  uint32_t sav_xStartWa, sav_yStartWa;
  uint32_t sav_widthWa, sav_heightWa;
  uint32_t sad;
  uint32_t wsize;

  if(!(freia_common_check_image_not_null(imout) &&
       freia_common_check_image_not_null(imin) &&
       freia_common_check_image_not_null(imref)))
  {
    FREIA_ERROR("Null pointers\n");
    return FREIA_INVALID_PARAM;
  }

  if(freia_common_check_image_bpp_compat(imout,imin,NULL) != true) {
    FREIA_ERROR("bpp of images are not compatibles\n");
    return FREIA_SIZE_ERROR;
  }

  if(freia_common_check_image_bpp_compat(imref,imin,NULL) != true) {
    FREIA_ERROR("bpp of images are not compatibles\n");
    return FREIA_SIZE_ERROR;
  }

  if(freia_common_check_value_compat(imref->widthWa, imin->widthWa+2*horizon) != true) {
    FREIA_ERROR("working areas of images are not compatibles\n");
    return FREIA_SIZE_ERROR;
  }

  if(freia_common_check_value_compat(imref->heightWa, imin->heightWa+2*horizon) != true) {
    FREIA_ERROR("working areas of images are not compatibles\n");
    return FREIA_SIZE_ERROR;
  }

  if(freia_common_check_value_compat(imout->widthWa, 2*horizon+1) != true) {
    FREIA_ERROR("image output width must be (2*horizon+1)\n");
    return FREIA_SIZE_ERROR;
  }

  if(freia_common_check_value_compat(imout->heightWa, 2*horizon+1) != true) {
    FREIA_ERROR("image output height must be (2*horizon+1)\n");
    return FREIA_SIZE_ERROR;
  }


  wsize = imin->heightWa * imin->widthWa;

  freia_common_get_wa(imref, &sav_xStartWa, &sav_yStartWa, &sav_widthWa, &sav_heightWa);

  for(n=0 ; n < (2*horizon+1) ; n++) {
    for(m=0 ; m < (2*horizon+1) ; m++) {
      freia_common_set_wa(imref, sav_xStartWa+m, sav_yStartWa+n, imin->widthWa, imin->heightWa);
      freia_cipo_global_sad(imin,imref, &sad);
      sad = sad / wsize;
      freia_common_set(imout, m, n, sad);
    }
  }

  freia_common_set_wa(imref, sav_xStartWa, sav_yStartWa, sav_widthWa, sav_heightWa);

  return FREIA_OK;
}
