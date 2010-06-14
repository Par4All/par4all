/**
*                                                                             
*   \file             steprt_comm_alg.h
*   \author           Abdellah Kouadri.
*                     Daniel Millot.
*                     Frédérique Silber-Chaussumier.
*   \date             22/10/2009
*   \version          1.1
*   \brief            Interface file for steprt_comm_alg.c file
*
*/

#ifndef STEP_COMM_ALG_H_
#define STEP_COMM_ALG_H_


#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include "mpi.h"
#include "steprt_common.h"
#include "steprt_private.h"
#include "trace.h"




void
step_alltoallregion_Blocking_1 (void *array, MPI_Datatype * region_desc,
				int *have_iteration, int *nb_request,
				MPI_Request * requests);

void
step_alltoallregion_Blocking_2 (void *array, MPI_Datatype * region_desc,
				int *have_iteration, int *nb_request,
				MPI_Request * requests);

void
step_alltoallregion_Blocking_3 (void *array, MPI_Datatype * region_desc,
				int *have_iteration, int *nb_request,
				MPI_Request * requests);



void
step_alltoallregion_NBlocking_1 (void *array, MPI_Datatype * region_desc,
				 int *have_iteration, int *nb_request,
				 MPI_Request * requests);

void
step_alltoallregion_NBlocking_2 (void *array, MPI_Datatype * region_desc,
				 int *have_iteration, int *nb_request,
				 MPI_Request * requests);



void
step_alltoallregion_merge_Blocking_1 (void *array, void *initial,
				      void *recv_buffer,
				      STEP_Datatype type,
				      MPI_Datatype * region_desc,
				      int *have_iteration, int *nb_request,
				      MPI_Request * requests,
				      int dim,
				      int dim_sizes[MAX_DIMS],
				      int dim_starts[MAX_PROCESSES][MAX_DIMS],
				      int dim_ss[MAX_PROCESSES][MAX_DIMS],
				      int row_col);
void
step_alltoallregion_merge_Blocking_2 (void *array, void *initial,
				      void *recv_buffer,
				      STEP_Datatype type,
				      MPI_Datatype * region_desc,
				      int *have_iteration, int *nb_request,
				      MPI_Request * requests,
				      int dim,
				      int dim_sizes[MAX_DIMS],
				      int dim_starts[MAX_PROCESSES][MAX_DIMS],
				      int dim_ss[MAX_PROCESSES][MAX_DIMS],
				      int row_col);


void
step_alltoallregion_merge_Blocking_3 (void *array, void *initial,
				      void *recv_buffer,
				      STEP_Datatype type,
				      MPI_Datatype * region_desc,
				      int *have_iteration, int *nb_request,
				      MPI_Request * requests,
				      int dim,
				      int dim_sizes[MAX_DIMS],
				      int dim_starts[MAX_PROCESSES][MAX_DIMS],
				      int dim_ss[MAX_PROCESSES][MAX_DIMS],
				      int row_col);


void
step_alltoallregion_merge_NBlocking_1 (void *array, void *initial,
				       void *recv_buffer,
				       STEP_Datatype type,
				       MPI_Datatype * region_desc,
				       int *have_iteration, int *nb_request,
				       MPI_Request * requests,
				       int dim,
				       int dim_sizes[MAX_DIMS],
				       int
				       dim_starts[MAX_PROCESSES][MAX_DIMS],
				       int dim_ss[MAX_PROCESSES][MAX_DIMS],
				       int row_col);
void step_alltoallregion_merge_NBlocking_2 (void *array, void *initial,
					    void *recv_buffer,
					    STEP_Datatype type,
					    MPI_Datatype * region_desc,
					    int *have_iteration,
					    int *nb_request,
					    MPI_Request * requests, int dim,
					    int dim_sizes[MAX_DIMS],
					    int
					    dim_starts[MAX_PROCESSES]
					    [MAX_DIMS],
					    int
					    dim_ss[MAX_PROCESSES][MAX_DIMS],
					    int row_col);


#endif
