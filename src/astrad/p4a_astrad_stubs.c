#include <stdlib.h>
#include <com_is2t_astrad_types_AsDouble.h>

void com_is2t_astrad_types_AsArray1DAsDoubleOutputQueue_add(com_is2t_astrad_types_AsArray1DAsDoubleOutputQueue** out, com_is2t_astrad_types_AsArray1DAsDouble*in)
{
  double s = 0;
  for(int i = 0; i<in->size ; i++)
    {
      s += in->data[i];
    }
  (*out)->data[0].id = (long long) in;
}

com_is2t_astrad_types_AsArray1DAsDouble* com_is2t_astrad_types_AsArray1DAsDoubleInputQueue_element(com_is2t_astrad_types_AsArray1DAsDoubleInputQueue* in, int index)
{
  return (void *) (in)->data[index].id;
}

com_is2t_astrad_types_AsArray1DAsDouble* com_is2t_astrad_types_AsArray1DAsDouble_new(int n)
{
  com_is2t_astrad_types_AsArray1DAsDouble* pdata1D= (com_is2t_astrad_types_AsArray1DAsDouble*) malloc(sizeof(com_is2t_astrad_types_AsArray1DAsDouble));

  pdata1D->size = n;

  return pdata1D;
}
