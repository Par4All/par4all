/**
*                                                                             
*   \file             steprt_wrapper.c
*   \author           Abdellah Kouadri.
*                     Daniel Millot.
*                     Frédérique Silber-Chaussumier.
*   \date             22/10/2009
*   \version          1.1
*   \brief            This file is present to widen the compatibility of the 
                      runtime as regard of the compiler being used. 
                      It just exports functions on steprt.c with a trailing 
                      underscore 
*/

#include <stdlib.h>
#include <stdint.h>
#include "steprt.h"
#include "trace.h"

int
step_init_fortran_order_ ()
{
  return step_init (STEP_FORTRAN);
}

void
step_finalize_ ()
{
  step_finalize ();
}

void
step_barrier_ ()
{
  step_barrier ();
}

int
step_get_size_ (int *size)
{
  return step_get_size (size);
}

int
step_get_rank_ (int *rank)
{
  return step_get_rank (rank);
}

void
step_computeloopslices_ (int *from, int *to, int *incr, int *nb_regions,
			 int *nb_proc, int *bounds)
{
  step_computeloopslices (from, to, incr, nb_regions, nb_proc, bounds);
}

int
step_sizeregion_ (int *dim, int *region)
{
  return step_sizeregion (dim, region);
}

void
step_waitall_ (int *NbReq, MPI_Request * Request)
{
  step_waitall (NbReq, Request);
}

void
step_alltoallregion_ (int *dim, int *nb_regions, int *regions, int *comm_size,
		      void *data_array, int *comm_tag, int *max_nb_request,
		      MPI_Request * requests, int *nb_request, int *algorithm,
		      STEP_Datatype * type)
{
  step_alltoallregion (dim, nb_regions, regions, comm_size,
		       data_array, comm_tag, max_nb_request,
		       requests, nb_request, algorithm, type);
}

void
step_alltoallregion_merge_ (int *dim, int *nb_regions, int *regions,
			    int *comm_size, void *data_array, void *initial,
			    void *buffer, int *comm_tag, int *max_nb_request,
			    MPI_Request * requests, int *nb_request,
			    int *algorithm, STEP_Datatype * type)
{
  step_alltoallregion_merge (dim, nb_regions, regions,
			     comm_size, data_array, initial,
			     buffer, comm_tag, max_nb_request,
			     requests, nb_request, algorithm, type);
}

void
step_initreduction_ (void *variable, void *variable_reduc, int *op,
		     STEP_Datatype * type)
{
  step_initreduction (variable, variable_reduc, op, type);
}

void
step_reduction_ (void *variable, void *variable_reduc, int *op, STEP_Datatype
		 * type)
{
  step_reduction (variable, variable_reduc, op, type);
}

void
step_mastertoallscalar_ (void *scalar, int *max_nb_request,
			 MPI_Request * requests, int *nb_request,
			 int *algorithm, STEP_Datatype * type)
{
  step_mastertoallscalar (scalar, max_nb_request,
			  requests, nb_request, algorithm, type);
}

void
step_initinterlaced_ (int *size, void *array, void *array_initial,
		      void *array_buffer, STEP_Datatype * type)
{
  step_initinterlaced (size, array, array_initial, array_buffer, type);
}

void
step_mastertoallregion_ (void *array, int *dim, int *regions, int *size,
			 int *max_nb_request, MPI_Request * requests,
			 int *nb_request, int *algorithm,
			 STEP_Datatype * type)
{
  step_mastertoallregion (array, dim, regions, size,
			  max_nb_request, requests,
			  nb_request, algorithm, type);
}
