#include <freiaDebug.h>
#include <freiaCommon.h>
#include <freiaAtomicOp.h>
#include <freiaComplexOp.h>

freia_status
median_1(freia_data2d *o, freia_data2d *i)
{
  int32_t c = 6;
  freia_status ret;
  freia_data2d * t = freia_common_create_data(i->bpp, i->widthWa, i->heightWa);
  ret =  freia_cipo_close(t, i, c, 1);
  ret |= freia_cipo_open(t, i, c, 1);
  ret |= freia_cipo_close(t, t, c, 1);

  ret |= freia_aipo_inf(o, t, i);

  ret |= freia_cipo_open(t, i, c, 1);
  ret |= freia_cipo_close(t, t, c, 1);
  ret |= freia_cipo_open(t, t, c, 1);

  ret |= freia_aipo_sup(o, o, t);
  ret |= freia_common_destruct_data(t);
  return ret;
}
