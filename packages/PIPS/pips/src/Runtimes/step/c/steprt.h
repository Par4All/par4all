/**
*                                                                             
*   \file             steprt.h
*   \author           Abdellah Kouadri.
*                     Daniel Millot.
*                     Frédérique Silber-Chaussumier.
*   \date             22/10/2009
*   \version          1.1
*   \brief            Interface file for the runtime (C side;)
*                                                                               
*/

#ifndef STEP_H_
#define STEP_H_

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include "mpi.h"
#include "steprt.h"
#include "trace.h"
#include "steprt_common.h"	/*Common definitions& Declarations*/
#include "steprt_private.h"	/*Internal data structures*/
#include "steprt_comm_alg.h"	/*Communication algorithms*/








int step_init (int lang);
int step_init_fortran_order ();
int step_init_c_order ();
void step_finalize ();

void step_barrier ();
int step_get_size (int *size);
int step_get_rank (int *rank);
int step_sizeregion (int *dim, int *region);
void step_waitall (int *NbReq, MPI_Request * Request);
void
step_computeloopslices (int *from, int *to, int *incr, int *nb_regions,
			int *nb_proc, int *bounds);
void step_alltoallregion (int *dim, int *nb_regions, int *regions,
			  int *comm_size, void *data_array, int *comm_tag,
			  int *max_nb_request, MPI_Request * requests,
			  int *nb_request, int *algorithm,
			  STEP_Datatype * type);
void step_alltoallregion_merge (int *dim, int *nb_regions, int *regions,
				int *comm_size, void *data_array,
				void *initial, void *buffer, int *comm_tag,
				int *max_nb_request, MPI_Request * requests,
				int *nb_request, int *algorithm,
				STEP_Datatype * type);

void
step_initreduction (void *variable, void *variable_reduc, int *op,
		    STEP_Datatype * type);
void step_reduction (void *variable, void *variable_reduc, int *op,
		     STEP_Datatype * type);

void step_mastertoallscalar (void *scalar, int *max_nb_request,
			     MPI_Request * requests, int *nb_request,
			     int *algorithm, STEP_Datatype * type);

void step_initinterlaced (int *size, void *array, void *array_initial,
			  void *array_buffer, STEP_Datatype * type);

void step_mastertoallregion (void *array, int *dim, int *regions, int *size,
			     int *max_nb_request, MPI_Request * requests,
			     int *nb_request, int *algorithm,
			     STEP_Datatype * type);


#endif
