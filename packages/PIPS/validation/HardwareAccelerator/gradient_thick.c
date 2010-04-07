#include "freia.h"

freia_status gradient_thick(freia_data2d * out, freia_data2d * in)
{
  freia_data2d
    * erode = freia_common_create_data(in->bpp, in->widthWa, in->heightWa),
    * dilate = freia_common_create_data(in->bpp, in->widthWa, in->heightWa);
  int32_t kernel[9] = {1,1,1, 1,1,1, 1,1,1};

  /* gradient thick using aipo calls*/
  freia_aipo_erode_8c(erode, in, kernel);
  freia_aipo_dilate_8c(dilate, in, kernel);
  freia_aipo_sub(out, dilate, erode);
  // freia_aipo_not(imout, imin);

  /* images destruction */
  freia_common_destruct_data(erode);
  freia_common_destruct_data(dilate);

  return FREIA_OK;
}
