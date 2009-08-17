/* Test case prepared for ANR evaluation meeting on September 9, 2009 */

typedef int freia_data2d;
typedef int int32_t;

extern int freia_aipo_global_vol();
extern int freia_aipo_global_min(freia_data2d);

int main(void) {
  freia_data2d *imin;
  int32_t measure_min, measure_vol;

  freia_aipo_global_vol(imin,&measure_vol);

  freia_aipo_global_min(imin,&measure_min);

  return 0;
}
