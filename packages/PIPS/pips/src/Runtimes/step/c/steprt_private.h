/**
*                                                                             
*   \file             steprt_private.h
*   \author           Abdellah Kouadri.
*                     Daniel Millot.
*                     Frédérique Silber-Chaussumier.
*   \date             22/10/2009
*   \version          1.1
*   \brief            Interface file for steprt_private.c
*                     
*/

#ifndef STEP_PRIVATE_H_
#define STEP_PRIVATE_H_


#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include "mpi.h"
#include "trace.h"
#include "steprt_common.h"


/**
* \struct step_internals
* \brief Encapsulates usefull variables 
*/
struct step_internals
{
  int size_;			/*!< The number of processes */
  int rank_;			/*!< The rank of current process */
  int language;			/*!< The calling language (C or Fortran) */
  int parallel_level;		/*!< The current nesting parallel level */
  int intialized;		/*!< Did the current process called step_init */
};

/**
* \struct complexe8_
* \brief Fortran complex8 data type
*/

typedef struct complexe8_
{
  float rp;			/*!< The real part */
  float ip;			/*!< The imaginary part */
} complexe8_t;
/**
* \struct complexe8_
* \brief Fortran complex16 data type
*/
typedef struct complexe16_
{
  double rp;			/*!< The real part */
  double ip;			/*!< The imaginary part */
} complexe16_t;



void
step_diff (void *buffer_1, void *buffer_2, void *initial, STEP_Datatype type,
	   int dims, int dim_sizes[MAX_DIMS],
	   int dim_starts_1[MAX_DIMS], int dim_starts_2[MAX_DIMS],
	   int dim_ss_1[MAX_DIMS], int dim_ss_2[MAX_DIMS], int row_col);


#endif
