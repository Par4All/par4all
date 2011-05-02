/**
*                                                                             
*   \file             steprt_comm_alg.c
*   \author           Abdellah Kouadri.
*                     Daniel Millot.
*                     Frédérique Silber-Chaussumier.
*                     Alain Muller
*   \date             04/02/2010
*   \version          2.0
*   \brief            Various communication algorithms for consistency recovery
*
*/

#include "steprt_comm_alg.h"

#include <assert.h>
#include <stdlib.h>

static void alltoall_NBlocking_1(array_identifier_t *array_desciptor, MPI_Datatype *SendMPI, MPI_Datatype *RecvMPI,
				 int *array_sizes, int *array_of_starts, int *array_of_sizes,
				 void *buffer, int tag, int *nb_request, MPI_Request *requests)
{  
  int id_node;
  int nb_node = step_.size_;
  int local_node = step_.rank_;
  int dim = array_desciptor->dims;      

  for (id_node = 0; id_node < nb_node; id_node++)
    if ((id_node != local_node) && (SendMPI[id_node] != MPI_DATATYPE_NULL))
      MPI_Isend(array_desciptor->array_id, 1, SendMPI[id_node], id_node, tag, MPI_COMM_WORLD, &requests[(*nb_request)++]);
  
  
  if (array_desciptor->saved_data != NULL) // interlaced
    {
      assert(buffer != NULL);
      // step_diff n'accepte pas des unions de regions RECV
      for (id_node = 0; id_node < nb_node; id_node++)
	if ((id_node != local_node) && (RecvMPI[id_node] != MPI_DATATYPE_NULL))
	  {
	    MPI_Recv(buffer, 1, RecvMPI[id_node], id_node, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    
	    step_diff (array_desciptor->array_id, buffer, array_desciptor->saved_data,
		       array_desciptor->array_type, dim, array_sizes,
		       &(array_of_starts[dim*local_node]), &(array_of_starts[dim*id_node]),
		       &(array_of_sizes[dim*local_node]), &(array_of_sizes[dim*id_node]),
		       step_.language);
	  }
    }
  else
    {
      for (id_node = 0; id_node < nb_node; id_node++)
	if ((id_node != local_node) && (RecvMPI[id_node] != MPI_DATATYPE_NULL))
	  MPI_Irecv(array_desciptor->array_id, 1, RecvMPI[id_node], id_node, tag, MPI_COMM_WORLD, &requests[(*nb_request)++]);
    }
}

void step_alltoall(array_identifier_t *array_desciptor, int algorithm, int tag)
{
  int id_node;
  int nb_node = step_.size_;

  /*implicite association process<->workchunk*/
  assert(nb_node == array_desciptor->region_descriptor[0]->nb_region);
  assert(nb_node == array_desciptor->recv_region.nb_region);

  MPI_Request *request_list = (MPI_Request*)malloc(sizeof(MPI_Request)*2*nb_node);
  int req_count = 0;

  void *buffer = NULL;
  MPI_Datatype SendMPI[nb_node];
  MPI_Datatype RecvMPI[nb_node];
  step_region_to_mpi_type(array_desciptor, SendMPI, RecvMPI);

  /*
    calcul du descripteur des regions RECV pour faire le diff (devrait etre retourne par step_region_to_mpi_type)

    ATTENTION le diff n'accepte qu'une region par noeud
  */
  int dim = array_desciptor->dims;
  int array_sizes[dim];
  int array_of_starts[nb_node * dim];
  int array_of_sizes[nb_node * dim];
  if (array_desciptor->saved_data != NULL) // interlaced
    {
      BOUNDS_2_START_SIZES(dim, array_desciptor->bounds, array_desciptor->bounds,
			   array_of_starts, array_sizes);
      index_t *regions_send = array_desciptor->region_descriptor[0]->regions;
      for (id_node = 0; id_node < nb_node ; id_node++)	//for all processes     
	BOUNDS_2_START_SIZES(dim, array_desciptor->bounds, &(regions_send[2*dim*id_node]),
			     &(array_of_starts[dim*id_node]), &(array_of_sizes[dim*id_node]));

      // allocation du buffer
      int type_size;
      MPI_Type_size(step_types_table[array_desciptor->array_type] ,&type_size);
      buffer =  malloc(type_size * array_desciptor->array_size);
      assert(buffer != NULL);      
  }
  

  switch (algorithm)
    {
    case STEP_NBLOCKING_ALG :
      alltoall_NBlocking_1(array_desciptor, SendMPI, RecvMPI,
			   array_sizes, array_of_starts, array_of_sizes,
			   buffer, tag, &req_count, request_list);
      break;
    default :
      assert(0);
    }


  if (array_desciptor->saved_data != NULL) // interlaced
    {
      free(buffer);
      free(array_desciptor->saved_data);
      array_desciptor->saved_data = NULL;
    }

  /*free datatypes*/
  for (id_node=0; id_node<nb_node; id_node++)
    {
      if (SendMPI[id_node] != MPI_DATATYPE_NULL)
	MPI_Type_free(&(SendMPI[id_node]));
      if (RecvMPI[id_node] != MPI_DATATYPE_NULL)
	MPI_Type_free(&(RecvMPI[id_node]));
    }
  
  step_hash_table_delete_send_region(array_desciptor->array_id);
  step_hash_table_delete_recv_region(array_desciptor->array_id);

  MPI_Waitall (req_count, request_list, MPI_STATUS_IGNORE);
}



#if 0
/*##############################################################################################

  alltoall_NBlocking_1 regroupe :
   - step_alltoallregion_NBlocking_1
   - step_alltoallregion_merge_NBlocking_1

  les autres algo de communication restant a porter suivent...


##############################################################################################*/

/**
* \fn void step_alltoallregion_Blocking_0 (void *array, MPI_Datatype * region_desc,
				int *have_iteration, int *nb_request,
				MPI_Request * requests)
* \brief  First version of the blocking alltoall algorithm 
*         (MPI_Sendrecv+hypercube algorihtm)
* \param [in,out] *array the data array to be exchanged
* \param [in] *region_desc The region descriptor
* \param [in] *have_iteration array that indicates if a given process has 
                   \n executed any iterations
* \param [in] *nb_request the number of requests  (equal to zero)
* \param [out] *requests array  (used in step_waitall())
* \return none
*/
void
step_alltoallregion_Blocking_0 (void *array, MPI_Datatype * region_desc,
				int *have_iteration, int *nb_request,
				MPI_Request * requests)
{
  IN_TRACE
    ("array = 0x%8.8X region_description = 0x%8.8X have_iteration = 0x%8.8X,nb_request = %d",
     array, region_desc, have_iteration, *nb_request);
  int i, pdf2, src, dst;	
  i = 1;
  while (i < step_.size_)
    i *= 2;
  if (i == step_.size_)
    pdf2 = 1;
  else
    pdf2 = 0;
  for (i = 1; i < step_.size_; i++)
    {
      if (pdf2 == 1)
	src = dst = step_.rank_ ^ i;
      else
	{
	  src = (step_.rank_ - i + step_.size_) % step_.size_;
	  dst = (step_.rank_ + i) % step_.size_;
	}
      if ((have_iteration[step_.rank_]) && (have_iteration[src]))
	{
	  MPI_Sendrecv ((void *) array, 1, region_desc[step_.rank_], dst,
			STEP_TAG_DEFAULT, (void *) array, 1, region_desc[src],
			src, STEP_TAG_DEFAULT, MPI_COMM_WORLD,
			MPI_STATUS_IGNORE);
	}
      else if (have_iteration[step_.rank_])
	{
	  MPI_Send ((void *) array, 1, region_desc[step_.rank_],
		    dst, STEP_TAG_DEFAULT, MPI_COMM_WORLD);
	}
      else if (have_iteration[src])
	{
	  MPI_Recv ((void *) array, 1, region_desc[src], src,
		    STEP_TAG_DEFAULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
    }
  OUT_TRACE ("");
}

/**
* \fn step_alltoallregion_Blocking_1 (void *array, MPI_Datatype * region_desc,
				int *have_iteration, int *nb_request,
				MPI_Request * requests)

* \brief  Second version of the blocking alltoall algorithm (MPI_Isend+MPI_Recv)
* \param [in,out] *array the data array to be exchanged
* \param [in] *region_desc The region descriptor
* \param [in] *have_iteration array that indicates if a given process has 
                   \n executed any iterations
* \param [in] *nb_request the number of requests  (equal to zero)
* \param [out] *requests array  (used in step_waitall())
* \return none
*/
void
step_alltoallregion_Blocking_1 (void *array, MPI_Datatype * region_desc,
				int *have_iteration, int *nb_request,
				MPI_Request * requests)
{
  IN_TRACE
    ("array = 0x%8.8X region_description = 0x%8.8X have_iteration = 0x%8.8X,nb_request = %d",
     array, region_desc, have_iteration, *nb_request);
  int i;
  int req_count;
  req_count = 0;
  for (i = 0; i < step_.size_; i++)
    if ((i != step_.rank_) && ((have_iteration[step_.rank_] == 1)))
      {
	MPI_Isend ((void *) array, 1, region_desc[step_.rank_], i,
		   STEP_TAG_DEFAULT, MPI_COMM_WORLD, &requests[req_count]);
	req_count++;
      }
  for (i = 0; i < step_.size_; i++)
    if ((i != step_.rank_) && (have_iteration[i] == 1))
      {
	MPI_Recv ((void *) array, 1, region_desc[i], i, STEP_TAG_DEFAULT,
		  MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
  OUT_TRACE ("");
}

/**
* \fn void step_alltoallregion_Blocking_2 (void *array, MPI_Datatype * region_desc,
				int *have_iteration, int *nb_request,
				MPI_Request * requests)

* \brief  Third version for the blocking alltoall algorithm (MPI_Send+MPI_Recv)
* \param [in,out] *array the data array to be exchanged
* \param [in] *region_desc The region descriptor
* \param [in] *have_iteration array that indicates if a given process has 
                   \n executed any iterations
* \param [in] *nb_request the number of requests  (equal to zero)
* \param [out] *requests array of requests (used in step_waitall())
* \return none
*/

void
step_alltoallregion_Blocking_2 (void *array, MPI_Datatype * region_desc,
				int *have_iteration, int *nb_request,
				MPI_Request * requests)
{
  IN_TRACE
    ("array = 0x%8.8X region_description = 0x%8.8X have_iteration = 0x%8.8X,nb_request = %d",
     array, region_desc, have_iteration, *nb_request);
  int target, i;
  target = (step_.size_ - 1) - step_.rank_;
  if (target == step_.rank_)
    target = (target + 1) % step_.size_;
  i = 0;
  while (i < step_.size_ - 1)
    {
      if (step_.rank_ < target)
	{
	  if (have_iteration[step_.rank_] == 1)
	    MPI_Send ((void *) array, 1, region_desc[step_.rank_],
		      target, STEP_TAG_DEFAULT, MPI_COMM_WORLD);
	  if (have_iteration[target] == 1)
	    MPI_Recv ((void *) array, 1, region_desc[target], target,
		      STEP_TAG_DEFAULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
      else
	{
	  if (have_iteration[target] == 1)
	    MPI_Recv ((void *) array, 1, region_desc[target], target,
		      STEP_TAG_DEFAULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  if (have_iteration[step_.rank_] == 1)
	    MPI_Send ((void *) array, 1, region_desc[step_.rank_],
		      target, STEP_TAG_DEFAULT, MPI_COMM_WORLD);
	}
      target = (target + 1) % step_.size_;
      if (target == step_.rank_)
	target = (target + 1) % step_.size_;
      i++;
    }
  OUT_TRACE ("");
}

/**
* \fn void step_alltoallregion_Blocking_3 (void *array, MPI_Datatype * region_desc,
				int *have_iteration, int *nb_request,
				MPI_Request * requests)

* \brief  Fourth version for the blocking alltoall algorithm (uses MPI_Sendrecv)
* \param [in,out] *array the data array to be exchanged
* \param [in] *region_desc The region descriptor
* \param [in] *have_iteration array that indicates if a given process has 
                   \n executed any iterations
* \param [in] *nb_request the number of requests  (equal to zero)
* \param [out] *requests array of requests (used in step_waitall())
* \return none
*/

void
step_alltoallregion_Blocking_3 (void *array, MPI_Datatype * region_desc,
				int *have_iteration, int *nb_request,
				MPI_Request * requests)
{
  IN_TRACE
    ("array = 0x%8.8X region_description = 0x%8.8X have_iteration = 0x%8.8X,nb_request = %d",
     array, region_desc, have_iteration, *nb_request);
  int target, i;
  target = (step_.size_ - 1) - step_.rank_;
  if (target == step_.rank_)
    target = (target + 1) % step_.size_;
  i = 0;
  while (i < step_.size_ - 1)
    {
     /*
      +------------------------+-----------------------------+-----------+
      | have_iteration[target] | have_iteration[step_.rank_] | quel_cas  |
      +------------------------+-----------------------------+-----------+
      |           0            |           0                 |     0     |            
      |           0            |           1                 |     2     |
      |           1            |           0                 |     1     |
      |           1            |           1                 |     3     |
      +------------------------+-----------------------------+-----------+
     */

      int quel_cas =		
	have_iteration[target] + have_iteration[step_.rank_] * 2;
      switch (quel_cas)
	{
	case 0:		/*Both processes have nothing to communicate */
	  break;
	case 1:		/*Only process target sends its region */
	  MPI_Recv ((void *) array, 1, region_desc[target], target,
		    STEP_TAG_DEFAULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  break;
	case 2:		/*Only local process sends its region */
	  MPI_Send ((void *) array, 1, region_desc[step_.rank_],
		    target, STEP_TAG_DEFAULT, MPI_COMM_WORLD);
	  break;
	case 3:		/*Booth processes (local and target) target send their regions */
	  MPI_Sendrecv ((void *) array, 1, region_desc[step_.rank_],
			target, STEP_TAG_DEFAULT, (void *) array, 1,
			region_desc[target], target, STEP_TAG_DEFAULT,
			MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  break;
	default:
	  assert (0) /*what ? unreachable region */ ;
	}
      target = (target + 1) % step_.size_;
      if (target == step_.rank_)
	target = (target + 1) % step_.size_;
      i++;
    }
  OUT_TRACE ("");
}

/**
* \fn void step_alltoallregion_NBlocking_1 (void *array, MPI_Datatype * region_desc,
				int *have_iteration, int *nb_request,
				MPI_Request * requests)

* \brief First  version for the non blocking alltoall algorithm 
*        (MPI_Irecv+MPI_Isend)     
* \param [in,out] *array the data array to be exchanged
* \param [in] *region_desc The region descriptor
* \param [in] *have_iteration array that indicates if a given process has 
                   \n executed any iterations
* \param [in,out] *nb_request the number of requests  (Non zero)
* \param [out] *requests array of requests (used in step_waitall())
* \return none
*/
void
step_alltoallregion_NBlocking_1 (void *array, MPI_Datatype * region_desc,
				 int *have_iteration, int *nb_request,
				 MPI_Request * requests)
{
  IN_TRACE
    ("array = 0x%8.8X region_description = 0x%8.8X have_iteration = 0x%8.8X,nb_request = %d",
     array, region_desc, have_iteration, *nb_request);
  int i;

  for (i = 0; i < step_.size_; i++)
    if ((i != step_.rank_) && (have_iteration[i] == 1))
      {
	MPI_Irecv ((void *) array, 1, region_desc[i], i, STEP_TAG_DEFAULT,
		   MPI_COMM_WORLD, &requests[*nb_request]);
	*nb_request+=1;

      }
  for (i = 0; i < step_.size_; i++)
    if ((i != step_.rank_) && (have_iteration[step_.rank_] == 1))
      {
	MPI_Isend ((void *) array, 1, region_desc[step_.rank_], i,
		   STEP_TAG_DEFAULT, MPI_COMM_WORLD, &requests[*nb_request]);
	*nb_request+=1;
      }

  OUT_TRACE ("");

}

/**
* \fn void step_alltoallregion_merge_Blocking_1 (void *array, void *initial,
				      void *recv_buffer,
				      STEP_Datatype type,
				      MPI_Datatype * region_desc,
				      int *have_iteration, int *nb_request,
				      MPI_Request * requests,
				      int dim,
				      int dim_sizes[MAX_DIMS],
				      int dim_starts[MAX_PROCESSES][MAX_DIMS],
				      int dim_ss[MAX_PROCESSES][MAX_DIMS],
				      int row_col)

* \brief First  version of the blocking alltoall algorithm using diff&merge.
         \n This routine is called in the case of overlapping regions.
         \n Uses MPI_Isend+MPI_Recv
* \param [in,out] *array the data array to be exchanged
* \param [in] *initial the initial version of *array 
* \param [in,out] *recv_buffer the scratchpad buffer used to receive data
* \param [in] type The data type of *array 
* \param [in] *region_desc The region descriptor
* \param [in] *have_iteration array that indicates if a given process has 
                   \n executed any iterations
* \param [in,out] *nb_request the number of requests  (Equal to zero)
* \param [out] *requests array  (used in step_waitall())
* \param [in] dim the number of dimensions of *array
* \param [in] dim_sizes[MAX_DIMS] gives the size of each dimension of *array
* \param [in] dim_starts[MAX_PROCESSES][MAX_DIMS] gives regions starts on 
              each dimension for each process
* \param [in] dim_ss[MAX_PROCESSES][MAX_DIMS] gives the regions sizes on each 
              dimension for each process
* \param [in] row_col indicates if we are working on a row or column major 
              ordered array
* \return none
*/

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
				      int row_col)
{
  IN_TRACE
    ("array = 0x%8.8X region_description = 0x%8.8X have_iteration = 0x%8.8X,nb_request = %d",
     array, region_desc, have_iteration, *nb_request);
  int i,req_count;
  req_count = 0;

  for (i = 0; i < step_.size_; i++)
    if ((i != step_.rank_) && (have_iteration[i] == 1))
      {
	MPI_Isend ((void *) array, 1, region_desc[step_.rank_], i,
		   STEP_TAG_DEFAULT, MPI_COMM_WORLD, &requests[req_count]);
	req_count++;
      }
  for (i = 0; i < step_.size_; i++)
    if ((i != step_.rank_) && (have_iteration[i] == 1))
      {
	MPI_Recv ((void *) recv_buffer, 1, region_desc[i], i,
		  STEP_TAG_DEFAULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	step_diff ((void *) array, (void *) recv_buffer, (void *) initial,
		   type, dim, dim_sizes, dim_starts[step_.rank_],
		   dim_starts[i], dim_ss[step_.rank_], dim_ss[i],
		   step_.language);
      }
  OUT_TRACE ("");
}

/**
* \fn void step_alltoallregion_merge_Blocking_2 (void *array, void *initial,
				      void *recv_buffer,
				      STEP_Datatype type,
				      MPI_Datatype * region_desc,
				      int *have_iteration, int *nb_request,
				      MPI_Request * requests,
				      int dim,
				      int dim_sizes[MAX_DIMS],
				      int dim_starts[MAX_PROCESSES][MAX_DIMS],
				      int dim_ss[MAX_PROCESSES][MAX_DIMS],
				      int row_col)

* \brief Second  version of the blocking alltoall algorithm using diff&merge.
         \n This routine is called in the case of overlapping regions.
         \n Uses MPI_Send+MPI_Recv
* \param [in,out] *array the data array to be exchanged
* \param [in] *initial the initial version of *array 
* \param [in,out] *recv_buffer the scratchpad buffer used to receive data
* \param [in] type The data type of *array 
* \param [in] *region_desc The region descriptor
* \param [in] *have_iteration array that indicates if a given process has 
                   \n executed any iterations
* \param [out] *nb_request the number of requests  (Equal to zero)
* \param [out] *requests array  (used in step_waitall())
* \param [in] dim the number of dimensions of *array
* \param [in] dim_sizes[MAX_DIMS] gives the size of each dimension of *array
* \param [in] dim_starts[MAX_PROCESSES][MAX_DIMS] gives regions starts on 
              each dimension for each process
* \param [in] dim_ss[MAX_PROCESSES][MAX_DIMS] gives the regions sizes on each 
              dimension for each process
* \param [in] row_col indicates if we are working on a row or column major 
              ordered array
* \return none
*/

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
				      int row_col)
{

  IN_TRACE ("array = 0x%8.8X initial = 0x%8.8X recv_buffer = 0x%8.8X ", array,
	    initial, recv_buffer);
  IN_TRACE ("type = %d region_desc = 0x%8.8X have_iteration = 0x%8.8X", type,
	    initial, have_iteration);
  IN_TRACE ("nb_request = %d dim  = %d", *nb_request, dim);
  int i, target;

  target = (step_.size_ - 1) - step_.rank_;
  if (target == step_.rank_)
    target = (target + 1) % step_.size_;
  i = 0;
  while (i < step_.size_ - 1)
    {
      if (step_.rank_ < target)
	{

	  if ((have_iteration[step_.rank_] == 1))
	    MPI_Send ((void *) array, 1, region_desc[step_.rank_],
		      target, STEP_TAG_DEFAULT, MPI_COMM_WORLD);

	  if ((have_iteration[target] == 1))
	    MPI_Recv ((void *) recv_buffer, 1, region_desc[target],
		      target, STEP_TAG_DEFAULT, MPI_COMM_WORLD,
		      MPI_STATUS_IGNORE);
	  /* a revoir!!! les cas ou on doit faire un merge sans diff */
	  step_diff ((void *) array, (void *) recv_buffer,
		     (void *) initial, type, dim, dim_sizes,
		     dim_starts[step_.rank_], dim_starts[target],
		     dim_ss[step_.rank_], dim_ss[target], step_.language);
	}
      else
	{
	  if ((have_iteration[target] == 1))
	    MPI_Recv ((void *) recv_buffer, 1, region_desc[target],
		      target, STEP_TAG_DEFAULT, MPI_COMM_WORLD,
		      MPI_STATUS_IGNORE);
	  if ((have_iteration[step_.rank_] == 1))
	    MPI_Send ((void *) array, 1, region_desc[step_.rank_],
		      target, STEP_TAG_DEFAULT, MPI_COMM_WORLD);
	  /* a revoir!!! les cas ou on doit faire un merge sans diff */
	  step_diff ((void *) array, (void *) recv_buffer,
		     (void *) initial, type, dim, dim_sizes,
		     dim_starts[step_.rank_], dim_starts[target],
		     dim_ss[step_.rank_], dim_ss[target], step_.language);
	}
      target = (target + 1) % step_.size_;
      if (target == step_.rank_)
	target = (target + 1) % step_.size_;
      i++;
    }
  OUT_TRACE ("");
}

/**
* \fn void step_alltoallregion_merge_Blocking_3 (void *array, void *initial,
				      void *recv_buffer,
				      STEP_Datatype type,
				      MPI_Datatype * region_desc,
				      int *have_iteration, int *nb_request,
				      MPI_Request * requests,
				      int dim,
				      int dim_sizes[MAX_DIMS],
				      int dim_starts[MAX_PROCESSES][MAX_DIMS],
				      int dim_ss[MAX_PROCESSES][MAX_DIMS],
				      int row_col)

* \brief Third version of the blocking alltoall algorithm using diff&merge.
         \n This routine is called in the case of overlapping regions.
         \n Uses MPI_Sendrecv if possible MPI_Send+MPI_Recv elsewhere
* \param [in,out] *array the data array to be exchanged
* \param [in] *initial the initial version of *array 
* \param [in,out] *recv_buffer the scratchpad buffer used to receive data
* \param [in] type The data type of *array 
* \param [in] *region_desc The region descriptor
* \param [in] *have_iteration array that indicates if a given process has 
                   \n executed any iterations
* \param [out] *nb_request the number of requests  (Equal to zero)
* \param [out] *requests array  (used in step_waitall())
* \param [in] dim the number of dimensions of *array
* \param [in] dim_sizes[MAX_DIMS] gives the size of each dimension of *array
* \param [in] dim_starts[MAX_PROCESSES][MAX_DIMS] gives regions starts on 
              each dimension for each process
* \param [in] dim_ss[MAX_PROCESSES][MAX_DIMS] gives the regions sizes on each 
              dimension for each process
* \param [in] row_col indicates if we are working on a row or column major 
              ordered array
* \return none
*/

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
				      int row_col)
{

  IN_TRACE ("array = 0x%8.8X initial = 0x%8.8X recv_buffer = 0x%8.8X ", array,
	    initial, recv_buffer);
  IN_TRACE ("type = %d region_desc = 0x%8.8X have_iteration = 0x%8.8X", type,
	    initial, have_iteration);
  IN_TRACE ("nb_request = %d dim  = %d", *nb_request, dim);
  int i, target;
  target = (step_.size_ - 1) - step_.rank_;
  if (target == step_.rank_)
    target = (target + 1) % step_.size_;
  i = 0;
  while (i < step_.size_ - 1)
    {
     /*
      +------------------------+-----------------------------+-----------+
      | have_iteration[target] | have_iteration[step_.rank_] | quel_cas  |
      +------------------------+-----------------------------+-----------+
      |           0            |           0                 |     0     |            
      |           0            |           1                 |     2     |
      |           1            |           0                 |     1     |
      |           1            |           1                 |     3     |
      +------------------------+-----------------------------+-----------+
     */

      int quel_cas =
	have_iteration[target] + have_iteration[step_.rank_] * 2;
      switch (quel_cas)
	{
	case 0:
	  break;
	case 1:
	  MPI_Recv ((void *) array, 1, region_desc[target], target,
		    STEP_TAG_DEFAULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  break;
	case 2:
	  MPI_Send ((void *) array, 1, region_desc[step_.rank_],
		    target, STEP_TAG_DEFAULT, MPI_COMM_WORLD);
	  break;
	case 3:
	  MPI_Sendrecv ((void *) array, 1, region_desc[step_.rank_],
			target, STEP_TAG_DEFAULT, (void *) recv_buffer, 1,
			region_desc[target], target, 0, MPI_COMM_WORLD,
			MPI_STATUS_IGNORE);
	  step_diff ((void *) array, (void *) recv_buffer,
		     (void *) initial, type, dim, dim_sizes,
		     dim_starts[step_.rank_], dim_starts[target],
		     dim_ss[step_.rank_], dim_ss[target], step_.language);
	  break;
	default:
	  assert (0) /*what ? */ ;
	}
      target = (target + 1) % step_.size_;
      if (target == step_.rank_)
	target = (target + 1) % step_.size_;
      i++;
    }
  OUT_TRACE ("");
}


/**
* \fn step_alltoallregion_merge_NBlocking_1 (void *array, void *initial,
				      void *recv_buffer,
				      STEP_Datatype type,
				      MPI_Datatype * region_desc,
				      int *have_iteration, int *nb_request,
				      MPI_Request * requests,
				      int dim,
				      int dim_sizes[MAX_DIMS],
				      int dim_starts[MAX_PROCESSES][MAX_DIMS],
				      int dim_ss[MAX_PROCESSES][MAX_DIMS],
				      int row_col)

* \brief Non Blocking version for the alltoall algorithm using diff&merge
         \n this routine is called in the case of overlapping regions
         \n uses MPI_Isend+MPI_Recv.  We need a blocking Recv in order to
            perform the diff&merge
* \param [in,out] *array the data array to be exchanged
* \param [in] *initial the initial version of *array 
* \param [in,out] *recv_buffer the scratchpad buffer used to receive data
* \param [in] type The data type of *array 
* \param [in] *region_desc The region descriptor
* \param [in] *have_iteration array that indicates if a given process has 
                   \n executed any iterations
* \param [in,out] *nb_request the number of requests  (Equal to zero)
* \param [out] *requests array  (used in step_waitall())
* \param [in] dim the number of dimensions of *array
* \param [in] dim_sizes[MAX_DIMS] gives the size of each dimension of *array
* \param [in] dim_starts[MAX_PROCESSES][MAX_DIMS] gives regions starts on 
              each dimension for each process
* \param [in] dim_ss[MAX_PROCESSES][MAX_DIMS] gives the regions sizes on each 
              dimension for each process
* \param [in] row_col indicates if we are working on a row or column major 
              ordered array
* \return none
*/
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
				       int row_col)
{

  IN_TRACE ("array = 0x%8.8X initial = 0x%8.8X recv_buffer = 0x%8.8X ", array,
	    initial, recv_buffer);
  IN_TRACE ("type = %d region_desc = 0x%8.8X have_iteration = 0x%8.8X", type,
	    initial, have_iteration);
  IN_TRACE ("nb_request = %d dim  = %d", *nb_request, dim);
  int i, req_count;
  req_count = 0;

  for (i = 0; i < step_.size_; i++)
    if ((i != step_.rank_) && (have_iteration[i] == 1))
      {
	MPI_Isend ((void *) array, 1, region_desc[step_.rank_], i,
		   STEP_TAG_DEFAULT, MPI_COMM_WORLD, &requests[req_count]);
	req_count++;
      }
  for (i = 0; i < step_.size_; i++)
    if ((i != step_.rank_) && (have_iteration[i] == 1))
      {
	MPI_Recv ((void *) recv_buffer, 1, region_desc[i], i,
		  STEP_TAG_DEFAULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	step_diff ((void *) array, (void *) recv_buffer, (void *) initial,
		   type, dim, dim_sizes, dim_starts[step_.rank_],
		   dim_starts[i], dim_ss[step_.rank_], dim_ss[i],
		   step_.language);
      }
  OUT_TRACE ("");
}

#endif
