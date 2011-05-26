/**
*                                                                             
*   \file             steprt.h
*   \author           Abdellah Kouadri.
*                     Daniel Millot.
*                     Frédérique Silber-Chaussumier.
*                     Alain Muller
*   \date             04/02/2010
*   \version          2.0
*   \brief            Interface file for the runtime (C side;)
*                                                                               
*/

#ifndef STEP_H_
#define STEP_H_

#include "steprt_private.h"	/*Internal data structures*/


int step_init (int lang);
int step_init_fortran_order ();
int step_init_c_order ();
void step_finalize ();

void step_barrier ();
int step_get_size (int *size);
int step_get_rank (int *rank);
int step_sizeregion (int *dim, int *region);
void step_waitall (int *NbReq, MPI_Request * Request);
void step_computeloopslices (int *from, int *to, int *incr, int *nb_regions,
			     int *nb_proc, int *bounds);

void step_initreduction (void *variable, void *variable_reduc, int *op,
			 STEP_Datatype * type);

void step_reduction (void *variable, void *variable_reduc, int *op,
		     STEP_Datatype * type);

void step_mastertoallscalar (void *scalar, int *max_nb_request,
			     MPI_Request * requests, int *nb_request,
			     int *algorithm, STEP_Datatype * type);

void step_mastertoallregion (void *array, int *dim, int *regions, int *size,
			     int *max_nb_request, MPI_Request *requests,
			     int *nb_request, int *algorithm,
			     STEP_Datatype *type);

void step_init_arrayregions(void *array,STEP_Datatype *type, int *dims, ...);

void step_set_sendregions(void *array, int *nb_workchunks, index_t *region);

void step_set_interlaced_sendregions(void *array, int *nb_workchunks, index_t *regions);

void step_set_recvregions(void *array, int *nb_workchunks, index_t *regions);

void step_alltoall_full(void *array, int *algorithm, int *tag) ;

void step_alltoall_full_interlaced(void *array, int *algorithm, int *tag);

void step_alltoall_partial(void *array, int *algorithm, int *tag);

void step_alltoall_partial_interlaced(void *array, int *algorithm, int *tag) ;


#endif
