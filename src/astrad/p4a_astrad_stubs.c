#include <stdio.h>
#include <stdlib.h>
#include <com_is2t_astrad_types_AsDouble.h>
#include <com_is2t_astrad_types_AsFloat.h>



// LOG
void com_is2t_astrad_log_AsLog_info(char * format, ...)
{
  fprintf(stderr, "%s", format);
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


// as_signal library
#ifdef P4A_AS_SIGNAL_LIBRARY
#include <as_signal.h>
void as2desc_signal(com_astrad_astypes_As_signal *p, as_signal_t* descout)
{
  descout->ch_ptr_header = &(p->header);
}

com_astrad_astypes_As_signal* as_new_signal(as_signal_t* descout, as_desc_type_t ndims, as_type_t type, ...)
{
  descout = (as_signal_t *) malloc(sizeof(as_signal_t));
}

void com_astrad_astypes_AsArray1DAs_signalOutputQueue_add(com_astrad_astypes_AsArray1DAs_signalOutputQueue** out, com_astrad_astypes_AsArray1DAs_signal*in)
{
  // mimic read effects on input and output effect on output
  (*out)->data[0].id = (long long) in;
}

void com_astrad_astypes_As_signalOutputQueue_add(com_astrad_astypes_As_signalOutputQueue** out, com_astrad_astypes_As_signal* in)
{
  // mimic read effects on input and output effect on output
  (*out)->data[0].id = (long long) in;
}

com_astrad_astypes_As_signal* desc2as_signal(as_signal_t* descout)
{
  return (com_astrad_astypes_As_signal*) malloc(sizeof (com_astrad_astypes_As_signal));
}
#endif

// DESC library
#ifdef P4A_DESC_LIBRARY
#include "DESC.h"
#define EPSILON     ((float)(0.1))
void DESC_update_dim_XD(float in_sur_out,       /* Rapport de taille in/out */
                        int in_moins_out,       /* difference de taille in-out */
                        const TS_DESC_XD* in,   /* Descripteur XD d'entree */
                        TS_DESC_XD* out)
{
    int i_max = in->ch_ast_type - AST_DESC_T1D;

    out->ch_dim_desc[i_max].ch_dim= ((int)(((float)(in->ch_dim_desc[i_max].ch_dim) / in_sur_out) + EPSILON)) - in_moins_out;

    for (int i = 0; i< i_max; i++)
        out->ch_dim_desc[i].ch_dim = in->ch_dim_desc[i].ch_dim;
}

void DESC_cast_1D(const TS_DESC_XD* in, /* Descripteur XD d'entree */
                  TS_DESC_1D* out)      /* Descripteur 1D de sortie */
{

    int i_max = in->ch_ast_type - DESC_T1D;

    /* Initialisation du desripteur de sortie */
    out->ch_type     = in->ch_type;
    out->ch_dim1     = in->ch_dim_desc->ch_dim;
    out->ch_dim1_max = in->ch_dim_desc->ch_dim_max;
    out->ch_ptr      = in->ch_ptr;

    /* boucle de traitement sur les dimensions */
    for(int i = i_max; i> 0; i--)
      {
        out->ch_dim1 *= in->ch_dim_desc[i].ch_dim_max;
        out->ch_dim1_max *= in->ch_dim_desc[i].ch_dim_max;
      }

} /* Fin de la fonction DESC_cast_1D */

#endif
