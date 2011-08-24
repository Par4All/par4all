/***************************************************************
 * Author : Christophe Clienti
 * Date : 10/22/2008
 * Copyright (c) 2008 Thales Research & Technologies (FR)
 ***************************************************************/


#include <freiaDebug.h>
#include <freiaCommon.h>
#include <freiaAtomicOpMorpho.h>
#include <freiaAtomicOpMeasure.h>
#include <freiaAtomicOpMisc.h>
#include <freiaAtomicOpArith.h>
#include <freiaComplexOpMorpho.h>

const int32_t freia_morpho_kernel_8c[9] = {1,1,1, 1,1,1, 1,1,1};
const int32_t freia_morpho_kernel_6c[9] = {0,1,1, 1,1,1, 0,1,1};
const int32_t freia_morpho_kernel_4c[9] = {0,1,0, 1,1,1, 0,1,0};


freia_status freia_cipo_erode(freia_data2d *imout, freia_data2d *imin, int32_t connexity, uint32_t size) {
  int i;

  if(freia_common_check_image_window_compat(imout,imin, NULL) != true) {
    FREIA_ERROR("working areas of images are not compatibles\n");
    return FREIA_SIZE_ERROR;
  }

  if(freia_common_check_image_bpp_compat(imout,imin, NULL) != true) {
    FREIA_ERROR("bpp of images are not compatibles\n");
    return FREIA_SIZE_ERROR;
  }

  if(size==0) {
    freia_aipo_copy(imout,imin);
    return FREIA_OK;
  }	       

  if (connexity==4) {
    freia_aipo_erode_8c(imout,imin,freia_morpho_kernel_4c);
    for(i=1 ; i<size ; i++) freia_aipo_erode_8c(imout,imout,freia_morpho_kernel_4c); 
  }
  else if (connexity==6) {
    freia_aipo_erode_6c(imout,imin,freia_morpho_kernel_6c);
    for(i=1 ; i<size ; i++) freia_aipo_erode_6c(imout,imout,freia_morpho_kernel_6c); 
  }
  else if (connexity==8) {
    freia_aipo_erode_8c(imout,imin,freia_morpho_kernel_8c);
    for(i=1 ; i<size ; i++) freia_aipo_erode_8c(imout,imout,freia_morpho_kernel_8c); 
  }
  else
    return FREIA_INVALID_PARAM;

  return FREIA_OK;
}



freia_status freia_cipo_dilate(freia_data2d *imout, freia_data2d *imin, int32_t connexity, uint32_t size) {
  int i;

  if(freia_common_check_image_window_compat(imout,imin, NULL) != true) {
    FREIA_ERROR("working areas of images are not compatibles\n");
    return FREIA_SIZE_ERROR;
  }

  if(freia_common_check_image_bpp_compat(imout,imin, NULL) != true) {
    FREIA_ERROR("bpp of images are not compatibles\n");
    return FREIA_SIZE_ERROR;
  }

  if(size==0) {
    freia_aipo_copy(imout,imin);
    return FREIA_OK;
  }	       

  if(connexity==4) {
    freia_aipo_dilate_8c(imout,imin,freia_morpho_kernel_4c);
    for(i=1 ; i<size ; i++) freia_aipo_dilate_8c(imout,imout,freia_morpho_kernel_4c);
  }
  else if (connexity==6) {
    freia_aipo_dilate_6c(imout,imin,freia_morpho_kernel_6c);
    for(i=1 ; i<size ; i++) freia_aipo_dilate_6c(imout,imout,freia_morpho_kernel_6c);
  }
  else if (connexity==8) {
    freia_aipo_dilate_8c(imout,imin,freia_morpho_kernel_8c);
    for(i=1 ; i<size ; i++) freia_aipo_dilate_8c(imout,imout,freia_morpho_kernel_8c);
  }
  else
    return FREIA_INVALID_PARAM;

  return FREIA_OK;
}

/* freia_status freia_cipo_erode(freia_data2d *imout, freia_data2d *imin, int32_t connexity, uint32_t size) { */
/*   freia_status ret; */
/*   freia_data2d *imtmp; */
/*   int i; */

/*   if(freia_common_check_image_window_compat(imout,imin, NULL) != true) { */
/*     FREIA_ERROR("working areas of images are not compatibles\n"); */
/*     return FREIA_SIZE_ERROR; */
/*   } */

/*   if(freia_common_check_image_bpp_compat(imout,imin, NULL) != true) { */
/*     FREIA_ERROR("bpp of images are not compatibles\n"); */
/*     return FREIA_SIZE_ERROR; */
/*   } */

/*   if(size==0) { */
/*     return freia_aipo_copy(imout,imin); */
/*   }	        */

/*   imtmp = freia_common_create_data(imout->bpp,  imout->widthWa, imout->heightWa); */
  
/*   if (connexity==4) { */
/*     ret = freia_aipo_erode_8c(imout,imin,freia_morpho_kernel_4c); */
/*     for(i=1 ; i<size ; i++) { */
/*       ret |= freia_aipo_copy(imtmp,imout); */
/*       ret |= freia_aipo_erode_8c(imout,imtmp,freia_morpho_kernel_4c); */
/*     } */
/*   } else if (connexity==6) { */
/*     ret = freia_aipo_erode_6c(imout,imin,freia_morpho_kernel_6c); */
/*     for(i=1 ; i<size ; i++) { */
/*       ret |= freia_aipo_copy(imtmp,imout); */
/*       ret |= freia_aipo_erode_6c(imout,imtmp,freia_morpho_kernel_6c); */
/*     } */
/*   } else if (connexity==8) { */
/*     ret = freia_aipo_erode_8c(imout,imin,freia_morpho_kernel_8c); */
/*     for(i=1 ; i<size ; i++) { */
/*       ret |= freia_aipo_copy(imtmp,imout); */
/*       ret |= freia_aipo_erode_8c(imout,imtmp,freia_morpho_kernel_8c); */
/*     } */
/*   } else { */
/*     ret = FREIA_INVALID_PARAM; */
/*   } */

/*   ret |= freia_common_destruct_data(imtmp); */

/*   return ret; */
/* } */



/* freia_status freia_cipo_dilate(freia_data2d *imout, freia_data2d *imin, int32_t connexity, uint32_t size) { */
/*   freia_status ret; */
/*   freia_data2d *imtmp; */
/*   int i; */

/*   if(freia_common_check_image_window_compat(imout,imin, NULL) != true) { */
/*     FREIA_ERROR("working areas of images are not compatibles\n"); */
/*     return FREIA_SIZE_ERROR; */
/*   } */

/*   if(freia_common_check_image_bpp_compat(imout,imin, NULL) != true) { */
/*     FREIA_ERROR("bpp of images are not compatibles\n"); */
/*     return FREIA_SIZE_ERROR; */
/*   } */

/*   if(size==0) { */
/*     return freia_aipo_copy(imout,imin); */
/*   }	        */

/*   imtmp = freia_common_create_data(imout->bpp,  imout->widthWa, imout->heightWa); */
  
/*   if (connexity==4) { */
/*     ret = freia_aipo_dilate_8c(imout,imin,freia_morpho_kernel_4c); */
/*     for(i=1 ; i<size ; i++) { */
/*       ret |= freia_aipo_copy(imtmp,imout); */
/*       ret |= freia_aipo_dilate_8c(imout,imtmp,freia_morpho_kernel_4c); */
/*     } */
/*   } else if (connexity==6) { */
/*     ret = freia_aipo_dilate_6c(imout,imin,freia_morpho_kernel_6c); */
/*     for(i=1 ; i<size ; i++) { */
/*       ret |= freia_aipo_copy(imtmp,imout); */
/*       ret |= freia_aipo_dilate_6c(imout,imtmp,freia_morpho_kernel_6c); */
/*     } */
/*   } else if (connexity==8) { */
/*     ret = freia_aipo_dilate_8c(imout,imin,freia_morpho_kernel_8c); */
/*     for(i=1 ; i<size ; i++) { */
/*       ret |= freia_aipo_copy(imtmp,imout); */
/*       ret |= freia_aipo_dilate_8c(imout,imtmp,freia_morpho_kernel_8c); */
/*     } */
/*   } else { */
/*     ret = FREIA_INVALID_PARAM; */
/*   } */

/*   ret |= freia_common_destruct_data(imtmp); */

/*   return ret; */
/* } */



freia_status freia_cipo_inner_gradient(freia_data2d *imout, freia_data2d *imin, int32_t connexity, uint32_t size) {
  freia_status ret;

  ret = freia_cipo_erode(imout, imin, connexity, size);
  ret |= freia_aipo_sub(imout,imin,imout);

  return ret;
}



freia_status freia_cipo_outer_gradient(freia_data2d *imout, freia_data2d *imin, int32_t connexity, uint32_t size) {
  freia_status ret;

  ret = freia_cipo_dilate(imout, imin, connexity, size);
  ret |= freia_aipo_sub(imout,imout,imin);

  return ret;
}



freia_status freia_cipo_gradient(freia_data2d *imout, freia_data2d *imin, int32_t connexity, uint32_t size) {
  freia_status ret;
  freia_data2d *imtmp;

  if(freia_common_check_image_window_compat(imout,imin, NULL) != true) {
    FREIA_ERROR("working areas of images are not compatibles\n");
    return FREIA_SIZE_ERROR;
  }
  
  if(freia_common_check_image_bpp_compat(imout,imin, NULL) != true) {
    FREIA_ERROR("bpp of images are not compatibles\n");
    return FREIA_SIZE_ERROR;
  }

  imtmp = freia_common_create_data(imout->bpp,  imout->widthWa, imout->heightWa);

  ret = freia_cipo_dilate(imtmp, imin, connexity, size);
  ret |= freia_cipo_erode(imout, imin, connexity, size);
  ret |= freia_aipo_sub(imout,imtmp,imout);

  freia_common_destruct_data(imtmp);
  return FREIA_OK;
}



freia_status freia_cipo_open(freia_data2d *imout, freia_data2d *imin, int32_t connexity, uint32_t size) {
  freia_status ret;

  ret = freia_cipo_erode(imout, imin, connexity, size);
  ret |= freia_cipo_dilate(imout, imout, connexity, size); 

  return ret;
}
 


freia_status freia_cipo_close(freia_data2d *imout, freia_data2d *imin, int32_t connexity, uint32_t size) {
  freia_status ret;

  ret = freia_cipo_dilate(imout, imin, connexity, size);
  ret |= freia_cipo_erode(imout, imout, connexity, size); 

  return ret;
}



freia_status freia_cipo_open_tophat(freia_data2d *imout, freia_data2d *imin, int32_t connexity, uint32_t size) {
  freia_status ret;

  ret = freia_cipo_open(imout, imin, connexity, size);
  ret |= freia_aipo_sub(imout,imin,imout);

  return ret;
}



freia_status freia_cipo_close_tophat(freia_data2d *imout, freia_data2d *imin, int32_t connexity, uint32_t size) {
  freia_status ret;

  ret = freia_cipo_close(imout, imin, connexity, size);
  ret |= freia_aipo_sub(imout,imout,imin);

  return ret;
}



freia_status freia_cipo_geodesic_dilate(freia_data2d *imout, freia_data2d *immarker, freia_data2d *immask, int32_t connexity, uint32_t size) {
  freia_status ret;

  ret = freia_cipo_dilate(imout, immarker, connexity, size);
  ret |= freia_aipo_inf(imout,imout,immask);

  return ret;
}

freia_status freia_cipo_geodesic_erode(freia_data2d *imout, freia_data2d *immarker, freia_data2d *immask, int32_t connexity, uint32_t size) {
  freia_status ret;

  ret = freia_cipo_erode(imout, immarker, connexity, size);
  ret |= freia_aipo_sup(imout,imout,immask);

  return ret;
}

freia_status freia_cipo_geodesic_reconstruct_dual(freia_data2d *immarker, freia_data2d *immask, int32_t connexity) {
  return FREIA_OK;
}
  
freia_status freia_cipo_geodesic_reconstruct_erode(freia_data2d *immarker, freia_data2d *immask, int32_t connexity) {
  freia_status ret = FREIA_OK;
  int32_t volcurrent;
  int32_t volprevious;

  if(freia_common_check_image_window_compat(immarker,immask, NULL) != true) {
    FREIA_ERROR("working areas of images are not compatibles\n");
    return FREIA_SIZE_ERROR;
  }

  if(freia_common_check_image_bpp_compat(immarker,immask, NULL) != true) {
    FREIA_ERROR("bpp of images are not compatibles\n");
    return FREIA_SIZE_ERROR;
  }

  ret = freia_aipo_global_vol(immarker, &volcurrent);

  do{
    volprevious = volcurrent;
    ret |= freia_cipo_geodesic_erode(immarker,immarker,immask,connexity,1);
    ret |= freia_aipo_global_vol(immarker, &volcurrent);
  }while(volcurrent != volprevious);

  return ret;
}

freia_status freia_cipo_geodesic_reconstruct_dilate(freia_data2d *immarker, freia_data2d *immask, int32_t connexity) {
  freia_status ret = FREIA_OK;
  int32_t volcurrent;
  int32_t volprevious;

  if(freia_common_check_image_window_compat(immarker,immask,NULL) != true) {
    FREIA_ERROR("working areas of images are not compatibles\n");
    return FREIA_SIZE_ERROR;
  }

  if(freia_common_check_image_bpp_compat(immarker,immask, NULL) != true) {
    FREIA_ERROR("bpp of images are not compatibles\n");
    return FREIA_SIZE_ERROR;
  }

  ret = freia_aipo_global_vol(immarker, &volcurrent);

  do{
    volprevious = volcurrent;
    ret |= freia_cipo_geodesic_dilate(immarker,immarker,immask,connexity,1);
    ret |= freia_aipo_global_vol(immarker, &volcurrent);
  }while(volcurrent != volprevious);

  return ret;
}

freia_status freia_cipo_geodesic_reconstruct_open(freia_data2d *imout, freia_data2d *imin, int32_t connexity, uint32_t size) {
  freia_status ret;

  ret = freia_cipo_erode(imout,imin,connexity,size);
  ret |= freia_cipo_geodesic_reconstruct_dilate(imout,imin,connexity);

  return ret;
}

freia_status freia_cipo_geodesic_reconstruct_close(freia_data2d *imout, freia_data2d *imin, int32_t connexity, uint32_t size) {
  freia_status ret;

  ret = freia_cipo_dilate(imout,imin,connexity,size);
  ret |= freia_cipo_geodesic_reconstruct_erode(imout,imin,connexity);

  return ret;
}
  
freia_status freia_cipo_geodesic_reconstruct_open_tophat(freia_data2d *imout, freia_data2d *imin, int32_t connexity, uint32_t size) {
  freia_status ret;

  ret = freia_cipo_geodesic_reconstruct_open(imout,imin,connexity,size);
  ret |= freia_aipo_sub(imout,imin,imout);

  return ret;
}

freia_status freia_cipo_geodesic_reconstruct_close_tophat(freia_data2d *imout, freia_data2d *imin, int32_t connexity, uint32_t size) {
  freia_status ret;

  ret = freia_cipo_geodesic_reconstruct_close(imout,imin,connexity,size);
  ret |= freia_aipo_sub(imout,imout,imin);

  return ret;
}
  
freia_status freia_cipo_regional_hminima(freia_data2d *imout, freia_data2d *imin, int32_t hlevel, int32_t connexity) {
  freia_status ret;

  ret = freia_aipo_subsat_const(imout,imin,hlevel);
  ret |= freia_cipo_geodesic_reconstruct_dilate(imout,imin,connexity);

  return FREIA_OK;
}

freia_status freia_cipo_regional_hmaxima(freia_data2d *imout, freia_data2d *imin, int32_t hlevel, int32_t connexity) {
  freia_status ret;

  ret = freia_aipo_addsat_const(imout,imin,hlevel);
  ret |= freia_cipo_geodesic_reconstruct_erode(imout,imin,connexity);

  return FREIA_OK;
}


