// goes with compilation_unit01

typedef struct {
  float re;
  float im;
} CplFloat;

void init(CplFloat B[10])
{
  for (int i =0; i<10; i++)
    {
      B[i].re = 1.0 * i;
      B[i].im = 2.0 * i;
    }
}
