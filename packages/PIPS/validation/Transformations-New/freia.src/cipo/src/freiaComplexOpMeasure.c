/***************************************************************
 * Author : Christophe Clienti
 * Date : 10/22/2008
 * Copyright (c) 2008 Thales Research & Technologies (FR)
 ***************************************************************/


#include <freiaDebug.h>
#include <freiaCommon.h>
#include <freiaAtomicOpArith.h>
#include <freiaAtomicOpMeasure.h>


freia_status freia_cipo_global_sad(freia_data2d *imin1, freia_data2d *imin2, uint32_t *sad) {
  int32_t vol;
  freia_status ret;
  freia_data2d *imtmp;
  
  if(freia_common_check_image_window_compat(imin1,imin2, NULL) != true) {
    FREIA_ERROR("working areas of images are not compatibles\n");
    return FREIA_SIZE_ERROR;
  }
  
  if(freia_common_check_image_bpp_compat(imin1,imin2, NULL) != true) {
    FREIA_ERROR("bpp of images are not compatibles\n");
    return FREIA_SIZE_ERROR;
  }

  imtmp = freia_common_create_data(imin1->bpp,  imin1->widthWa, imin1->heightWa);

  ret = freia_aipo_absdiff(imtmp, imin1, imin2);

  ret |= freia_aipo_global_vol(imtmp,&vol);

  ret |= freia_common_destruct_data(imtmp);

  *sad = (uint32_t) vol;

  return ret;
}
