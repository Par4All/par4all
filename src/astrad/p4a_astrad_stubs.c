#include <stdio.h>
#include <stdlib.h>
#include <com_is2t_astrad_types_AsDouble.h>
#include <com_is2t_astrad_types_AsFloat.h>

// LOG
extern void com_is2t_astrad_log_AsLog_info(char * format, ...)
{
  fprintf(stderr, format);
}

// DOUBLE

void com_is2t_astrad_types_AsArray1DAsDoubleOutputQueue_add(com_is2t_astrad_types_AsArray1DAsDoubleOutputQueue** out, com_is2t_astrad_types_AsArray1DAsDouble*in)
{
  double s = 0;
  // mimic read effects on all elements of input array
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

  pdata1D->shape[0] = n;

  return pdata1D;
}

// FLOAT

com_is2t_astrad_types_AsArray2DAsFloat* com_is2t_astrad_types_AsArray2DAsFloat_new(int n, int m)
{
  com_is2t_astrad_types_AsArray2DAsFloat* pdata2D= (com_is2t_astrad_types_AsArray2DAsFloat*) malloc(sizeof(com_is2t_astrad_types_AsArray2DAsFloat));

  pdata2D->shape[0] = n;
  pdata2D->shape[1] = m;
  return pdata2D;
}

void com_is2t_astrad_types_AsArray2DAsFloatOutputQueue_add(com_is2t_astrad_types_AsArray2DAsFloatOutputQueue**out, com_is2t_astrad_types_AsArray2DAsFloat*in)
{
  float s = 0;
  // mimic read effects on all elements of input array
  for(int i = 0; i<in->size ; i++)
    {
      s += in->data[i];
    }
  (*out)->data[(*out)->stopPosition+1].id = (long long) in;
}
