/**
*                                                                             
*   \file             steprt.c
*   \author           Abdellah Kouadri.
*                     Daniel Millot.
*                     Frédérique Silber-Chaussumier.
*   \date             22/10/2009
*   \version          1.1
*   \brief            This file contains core routines of the runtime
*                     (C or Fortran)                           
*/





#include "steprt.h"

extern struct step_internals step_;
extern MPI_Comm step_comm;
extern MPI_Datatype step_types_table[STEP_MAX_TYPES];


/**
* \fn int step_init_fortran_order()
* \brief  Fortran specific initialization routine (must be called from Fortran 
   code)
* \param none 
* \return STEP_OK or STEP_ALREADY_INITIALIZED if success 
*/
int
step_init_fortran_order ()
{
  return step_init (STEP_FORTRAN);
}

/**
* \fn int step_init_c_order()
* \brief  C specific initialization routine (must be called from C code)
* \param none 
* \return STEP_OK or STEP_ALREADY_INITIALIZED if success 
*/
int
step_init_c_order ()
{
  return step_init (STEP_C);
}

/**
* \fn step_init(int lang)
* \brief Generic Initialization routine 
* \param[in] lang must be either STEP_C or STEP_FORTRAN
* \return STEP_OK or STEP_ALREADY_INITIALIZED if success 
*/
int
step_init (int lang)
{
  int ig, ps;
  IN_TRACE ("%d\n", lang);
  MPI_Initialized (&ig);

  if ((!step_.intialized) && (!ig))
    {
      assert (MPI_Init_thread (NULL, NULL, MPI_THREAD_FUNNELED, &ps) ==
	      MPI_SUCCESS);
      if ((ps != MPI_THREAD_FUNNELED) && (ps != MPI_THREAD_MULTIPLE))
	fprintf (stdout,
	 "Warnning in step_init() : FUNNELED thread support not available\n");
      MPI_Comm_size (MPI_COMM_WORLD, &step_.size_);
      MPI_Comm_rank (MPI_COMM_WORLD, &step_.rank_);
      step_.language = lang;
      step_.parallel_level = 0;
      step_.intialized = 1;
      OUT_TRACE ("Initialization OK\n");
      return STEP_OK;
    }
  else
    {
      step_.language = lang;
      step_.parallel_level = 0;
      if (step_.intialized == 1)
	{
	  OUT_TRACE ("Initialization OK [Runtime alreday initialized]\n");
	  return (STEP_ALREADY_INITIALIZED);
	}
      else
	{
	  step_.intialized = 1;
	  OUT_TRACE ("Initialization OK\n");
	  return STEP_OK;
	}
    }
}

/**
* \fn step_finalize()
* \brief Finalization routine must be called at the very end of the execution
* \param none 
* \return void
*/
void
step_finalize ()
{
  IN_TRACE ("Finalization...\n");
  MPI_Finalize ();
  OUT_TRACE ("");
}

/**
* \fn void step_barrier()
* \brief Puts a synchronization barrier on all processes of step_comm 
* \param none 
* \return void
*/
void
step_barrier ()
{
  IN_TRACE ("");
  MPI_Barrier (MPI_COMM_WORLD);
  OUT_TRACE ("");
}

/**
* \fn int step_get_size(int *size)
* \brief Returns the number of processes belonging  to step_comm
* \param[out] *size the number of processes of step_comm  
* \return STEP_OK 
*/
int
step_get_size (int *size)
{
  IN_TRACE ("%8.8X\n", size);
  MPI_Comm_size (MPI_COMM_WORLD, size);
  OUT_TRACE ("");
  return STEP_OK;
}

/**
* \fn int step_get_rank(int *rank)
* \brief Returns the rank within step_comm of the calling process
* \param[out] *rank the rank of caller
* \return STEP_OK 
*/
int
step_get_rank (int *rank)
{
  IN_TRACE ("%8.8X\n", rank);
  *rank = step_.rank_;
  OUT_TRACE ("rank=%d\n", *rank);
  return STEP_OK;
}

/**
* \fn void step_computeloopslices (int *from, int *to, 
*       int *incr, int *nb_regions,int *nb_proc, int *bounds)
* \brief Distributes iterations of a loop amongs all processes 
*        of step_comm the resulting distribution is returned in 
*        bounds array
* \param[in] *from   loop's start
* \param[in] *to     loop's end
* \param[in] *incr   loop's increment
* \param[in] *nb_regions the number of loop slices (actually equal to nb_proc)
* \param[in] *nb_proc the number of processes
* \param[out] *bounds the resulting distribition  
*             bounds[0][0] = *from
*             bounds[0][1] = *to
*             bounds[i][0] = the loop start for the  process having rank (i-1)
*             bounds[i][1] = the loop end for the  process having rank (i-1)
* \return    none 
* \remarks   Bounds[*][*] is organized either in a row-major or a 
*            column major order depending on the calling language
* \remarks   Bounds must be declared as INTEGER bounds(2,MAX_PROCESSES) in 
             FORTRAN and int bounds[MAX_PROCESSES][2] in C (user's code side)
*/
void
step_computeloopslices (int *from, int *to, int *incr, int *nb_regions,
			int *nb_proc, int *bounds)
{

  int i;
  int nb_indices, nb_i, nb_e;
  IN_TRACE
    ("id = %d\t from = %d\t to = %d\tincr = %d\tnb_region = %d\tnb_proc = %d\t",
     step_.rank_, *from, *to, *incr, *nb_regions, *nb_proc);
  if (step_.language == STEP_FORTRAN)
    {
      for (i = 0; i < *nb_regions; i++)
	{
	  bounds[i * 2] = -1;
	  bounds[i * 2 + 1] = -1;
	}

      if (((*incr > 0) && (*to - *from < *incr))	/* no iterations */
	  || ((*incr < 0) && (*from - *to < -*incr)))
	{
	  for (i = 0; i < *nb_regions; i++)
	    {
	      bounds[i * 2] = *to;
	      bounds[1 + i * 2] = *from;
	    }
	}
      else
	{
	  nb_indices = (*to - *from) / (*incr) + 1;
	  if (nb_indices < *nb_regions)	/*Less iterations than processes */
	    {
	      bounds[0] = *from;
	      bounds[1] = *from;
	      for (i = 1; i < nb_indices; i++)
		{
		  bounds[i * 2] = bounds[(i - 1) * 2] + *incr;
		  bounds[i * 2 + 1] = bounds[i * 2];
		}
	      for (i = nb_indices; i < *nb_regions; i++)
		{
		  bounds[i * 2] = *to;
		  bounds[i * 2 + 1] = *from;
		}
	    }
	  else			/*Distribute extra iterations */
	    {
	      nb_i = nb_indices % (*nb_regions);
	      nb_e = (nb_indices - nb_i) / (*nb_regions);
	      if (nb_i == 0)
		{
		  nb_i = *nb_regions;
		  nb_e = nb_e - 1;
		}
	      bounds[0] = *from;
	      bounds[1] = *from + nb_e * (*incr);
	      for (i = 1; i < nb_i; i++)
		{
		  bounds[i * 2] = bounds[(i - 1) * 2 + 1] + (*incr);
		  bounds[i * 2 + 1] = bounds[i * 2] + nb_e * (*incr);
		}
	      for (i = nb_i; i < *nb_regions; i++)
		{
		  bounds[i * 2] = bounds[(i - 1) * 2 + 1] + (*incr);
		  bounds[i * 2 + 1] = bounds[i * 2] + (nb_e - 1) * (*incr);
		}
	    }
	}

    }
  else				/*C language not implemented yet*/
    {
      assert (0);
    }
  OUT_TRACE ("id = %d\t loop start = %d\t loop_end = %d", step_.rank_,
	     bounds[2 * step_.rank_], bounds[1 + 2 * step_.rank_]);
#ifdef STEP_DEBUG_LOOP
  int j;
  if (step_.rank_ == 0)
    {
      fprintf (stdout, "\n from = %d \t to = %d\n", *from, *to);
      for (i = 0; i < *nb_regions; i++)
	for (j = 0; j < 2; j++)
	  {
	    fprintf (stdout, "\n bounds[%d][%d] = %d ", j, i,
		     bounds[j + 2 * i]);
	  }
    }
#endif
}

/**
* \fn int step_sizeregion(int *dim, int *region)
* \brief Returns the size for region *region in the ith dimension
* \param[in] *dim the dimension identifier 
* \param[in] *region the region descriptor
* \return the ith size for region *region
*/
int
step_sizeregion (int *dim, int *region)
{
  int i;
  int ret = 1;
  for (i = 0; i < *dim; i++)
    {
      ret = ret * (region[i * (*dim) + 1] - region[i * (*dim)] + 1);
    }
  return ret;
}

/**
* \fn void step_waitall (int *NbReq, MPI_Request * Request)
* \brief Waits for all requests in *Request array to complete  
* \param[in] *NbReq Number of request to wait on
* \param[in] *Request An array of MPI_Request
* \return none
*/
void
step_waitall (int *NbReq, MPI_Request * Request)
{
  MPI_Waitall (*NbReq, Request, MPI_STATUS_IGNORE);
}

/**
* \fn void step_alltoallregion (int *dim, int *nb_regions, int *regions, 
                     int *comm_size,
		     void *data_array, int *comm_tag, int *max_nb_request,
		     MPI_Request * requests, int *nb_request, int *algorithm,
		     STEP_Datatype * type)
* \brief This routine performs all necessary data exchange in order to 
          recover data consistency for *array in the case where all regions 
          are disjoint (i.e no overlap)
* \param[in] *dim  number of dimensions of the array
* \param[in] *nb_regions  number of regions 
* \param[in] *regions  the region descriptor for each process
* \param[in] *comm_size   the number of processes
* \param[in,out] *data_array the data array
* \param[in] *comm_tag value used as tag for MPI communications
* \param[in] *max_nb_request  the maximum number of MPI requests
* \param[out] *requests array of MPI requests
* \param[out] *nb_request the actual number of requests 
* \param[in] *algorithm the communication algorithm to be used
* \param[in] *type the data type of *data_array
* \return none
* \remarks here is a list of supported algorithms
* \n	STEP_NBLOCKING_ALG  => Non blocking Send and  non blocking Recv
* \n	STEP_BLOCKING_ALG_1 => Non blocking Send and  blocking Recv
* \n	STEP_BLOCKING_ALG_2 =>  blocking Send and  blocking Recv
* \n	STEP_BLOCKING_ALG_3 =>  blocking Send and  blocking Recv (MPI_Sendrecv)
* \n	STEP_BLOCKING_ALG_4 =>  not implemented
*/
void
step_alltoallregion (int *dim, int *nb_regions, int *regions, int *comm_size,
		     void *data_array, int *comm_tag, int *max_nb_request,
		     MPI_Request * requests, int *nb_request, int *algorithm,
		     STEP_Datatype * type)
{
  int i, j, k, req_count, target, my_id;
  int dim_sizes[MAX_DIMS];
  int dim_sub_sizes[MAX_PROCESSES][MAX_DIMS];
  int dim_starts[MAX_PROCESSES][MAX_DIMS];
  MPI_Datatype region_mpidesc[MAX_PROCESSES];
  my_id = step_.rank_;
  int have_iteration[MAX_PROCESSES];

  IN_TRACE ("rank = %d, algorithm = %d, base_type_size = %d, dims_count = %d",
	    my_id, *algorithm, *type, *dim);
  if (step_.language == STEP_FORTRAN)
    {
      for (i = 0; i < *dim; i++)
	dim_sizes[i] = regions[1 + 2 * i] - regions[2 * i] + 1;
      for (i = 1; i < *nb_regions + 1; i++)	//for all processes     
	{
	  have_iteration[i - 1] = 0;
	  for (j = 0; j < *dim; j++)	// for all dims
	    {
	      dim_sub_sizes[i - 1][j] =
		regions[1 + j * 2 + 2 * *dim * i] - regions[j * 2 +
							    2 * *dim * i] + 1;
	      dim_starts[i - 1][j] = regions[j * 2 + 2 * *dim * i] - regions[2 * j];

	      if (dim_sub_sizes[i - 1][j] > 0)
		have_iteration[i - 1]++;
	    }
	  if (have_iteration[i - 1] >= *dim)
	    have_iteration[i - 1] = 1;
	  else
	    have_iteration[i - 1] = 0;
	}
    }
  else				/*C language */
    {
      assert (0);
    }
#ifdef STEP_DEBUG_REGIONS
  if (step_.rank_ == 0)
    {
      fprintf (stdout,
	       "\n_______________[ALLTOALL]____________________________\n");
      for (j = 0; j < step_.size_; j++)
	{
	  fprintf (stdout,
		   "\n PROCESS N°%d  DIMS_COUNT = %d HAVE_ITERATION = %d\n",
		   j, *dim, have_iteration[j]);
	  for (i = 0; i < *dim; i++)
	    {
	      fprintf (stdout,
		       "\n\t\tDIMS N°%d \t SIZE = %d\t START = %d \t END = %d",
		       i, dim_sizes[i], dim_starts[j][i],
		       dim_starts[j][i] + dim_sub_sizes[j][i]);
	    }
	}
      fprintf (stdout,
	       "\n_____________________________________________________\n");
    }
#endif
  for (i = 0; i < *nb_regions; i++)
    {
      if (have_iteration[i] == 1)
	{

	  MPI_Type_create_subarray (*dim, &(dim_sizes[0]),
				    &(dim_sub_sizes[i][0]),
				    &(dim_starts[i][0]),
				    step_.language,
				    step_types_table[*type],
				    &region_mpidesc[i]);
	  MPI_Type_commit (&region_mpidesc[i]);
	}
    }
  switch (*algorithm)
    {
    case STEP_NBLOCKING_ALG:	/*tested */
      step_alltoallregion_NBlocking_1 (data_array, region_mpidesc,
				       have_iteration, nb_request, requests);
      break;
    case STEP_BLOCKING_ALG_1:

      step_alltoallregion_Blocking_1 (data_array, region_mpidesc,
				      have_iteration, nb_request, requests);
      break;
    case STEP_BLOCKING_ALG_2:	/*tested : same! */
      step_alltoallregion_Blocking_2 (data_array, region_mpidesc,
				      have_iteration, nb_request, requests);
      break;
    case STEP_BLOCKING_ALG_3:	/*tested : same! */
      step_alltoallregion_Blocking_3 (data_array, region_mpidesc,
				      have_iteration, nb_request, requests);
      break;
    case STEP_BLOCKING_ALG_4:	/*Snow ball algorithm not implemented yet */
      printf ("This algorithm is not implemented yet ...");
      assert (0);
    default:
      assert (0);
    }
  if (*algorithm != STEP_NBLOCKING_ALG)
    for (i = 0; i < *nb_regions; i++)
      {
	if (have_iteration[i] == 1)
	  MPI_Type_free (&region_mpidesc[i]);
      }
  OUT_TRACE ("");
}

/**
* \fn void step_alltoallregion_merge(int *dim, int *nb_regions, int *regions,
			   int *comm_size, void *data_array, void *initial,
			   void *buffer, int *comm_tag, int *max_nb_request,
			   MPI_Request * requests, int *nb_request,
			   int *algorithm, STEP_Datatype * type)
* \brief This routine performs all necessary data exchange in order to 
          recover data consistency for *array in the case where regions 
          overlap
* \param[in] *dim  number of dimension of the array
* \param[in] *nb_regions  number fo regions 
* \param[in] *regions  the region descriptor for each process
* \param[in] *comm_size   the number of processes
* \param[in,out] *data_array the data array
* \param[in]  *initial copy of *array before the parallel processing has occured
* \param[in]  *buffer a scratchpad buffer used to receive local copies from 
               other processes
* \param[in] *comm_tag value used as tag for MPI communications
* \param[in] *max_nb_request  the maximum number of MPI requests
* \param[in] *requests array of MPI requests
* \param[out] *nb_request the actual number of requests 
* \param[in] *algorithm the communication algorithm to be used
* \param[in] *type the data type of *data_array
* \return none
*/

void
step_alltoallregion_merge (int *dim, int *nb_regions, int *regions,
			   int *comm_size, void *data_array, void *initial,
			   void *buffer, int *comm_tag, int *max_nb_request,
			   MPI_Request * requests, int *nb_request,
			   int *algorithm, STEP_Datatype * type)
{
  int i, j, k, req_count, target, my_id;
  int dim_sizes[MAX_DIMS];
  int dim_sub_sizes[MAX_PROCESSES][MAX_DIMS];
  int dim_starts[MAX_PROCESSES][MAX_DIMS];
  MPI_Datatype region_mpidesc[MAX_PROCESSES];
  my_id = step_.rank_;
  int have_iteration[MAX_PROCESSES];
  IN_TRACE ("rank = %d, algorithm = %d, base_type_size = %d, dims_count = %d",
	    my_id, *algorithm, *type, *dim);

  for (i = 0; i < *dim; i++)
    dim_sizes[i] = regions[1 + 2 * i] - regions[2 * i] + 1;
  for (i = 1; i < *nb_regions + 1; i++)	//for all processes     
    {
      have_iteration[i - 1] = 0;
      for (j = 0; j < *dim; j++)	// for all dims
	{
	  dim_sub_sizes[i - 1][j] =
	    regions[1 + j * 2 + 2 * *dim * i] - regions[j * 2 +
							2 * *dim * i] + 1;
	  dim_starts[i - 1][j] = regions[j * 2 + 2 * *dim * i] - regions[2 * j];
	  if (dim_sub_sizes[i - 1][j] > 0)
	    have_iteration[i - 1]++;
	}
      if (have_iteration[i - 1] >= *dim)
	have_iteration[i - 1] = 1;
      else
	have_iteration[i - 1] = 0;
    }

#ifdef STEP_DEBUG_REGIONS
  if (step_.rank_ == 0)
    {
      fprintf (stdout,
	       "\n_____________________________________________________\n");
      for (j = 0; j < step_.size_; j++)
	{
	  fprintf (stdout, "\n PROCESS N°%d  DIMS_COUNT = %d\n", j, *dim);
	  for (i = 0; i < *dim; i++)
	    {
	      fprintf (stdout,
		       "\n\t\tDIMS N°%d \t SIZE = %d\t START = %d \t END = %d",
		       i, dim_sizes[i], dim_starts[j][i],
		       dim_starts[j][i] + dim_sub_sizes[j][i]);
	    }
	}
      fprintf (stdout,
	       "\n_____________________________________________________\n");
    }
#endif
  for (i = 0; i < *nb_regions; i++)
    {
      if (have_iteration[i] == 1)
	{

	  MPI_Type_create_subarray (*dim, &(dim_sizes[0]),
				    &(dim_sub_sizes[i][0]),
				    &(dim_starts[i][0]),
				    step_.language,
				    step_types_table[*type],
				    &region_mpidesc[i]);
	  MPI_Type_commit (&region_mpidesc[i]);
	}
    }
  switch (*algorithm)
    {
    case STEP_NBLOCKING_ALG:

      step_alltoallregion_merge_NBlocking_1 (data_array, initial, buffer,
					     *type, region_mpidesc,
					     have_iteration, nb_request,
					     requests, *dim, dim_sizes,
					     dim_starts, dim_sub_sizes,
					     step_.language);
      break;
    case STEP_BLOCKING_ALG_1:

      step_alltoallregion_merge_Blocking_1 (data_array, initial, buffer,
					    *type, region_mpidesc,
					    have_iteration, nb_request,
					    requests, *dim, dim_sizes,
					    dim_starts, dim_sub_sizes,
					    step_.language);
      break;
    case STEP_BLOCKING_ALG_2:
      step_alltoallregion_merge_Blocking_2 (data_array, initial, buffer,
					    *type, region_mpidesc,
					    have_iteration, nb_request,
					    requests, *dim, dim_sizes,
					    dim_starts, dim_sub_sizes,
					    step_.language);

      break;

    case STEP_BLOCKING_ALG_3:
      step_alltoallregion_merge_Blocking_3 (data_array, initial, buffer,
					    *type, region_mpidesc,
					    have_iteration, nb_request,
					    requests, *dim, dim_sizes,
					    dim_starts, dim_sub_sizes,
					    step_.language);
      break;
    case STEP_BLOCKING_ALG_4:	/*Snow ball algorithm not implemented yet */
      step_alltoallregion_merge_Blocking_4 (data_array, initial, buffer,
					    *type, region_mpidesc,
					    have_iteration, nb_request,
					    requests, *dim, dim_sizes,
					    dim_starts, dim_sub_sizes,
					    step_.language);

      break;
    default:
      assert (0);
    }
  for (i = 0; i < *nb_regions; i++)
    {
      if (have_iteration[i] == 1)
	MPI_Type_free (&region_mpidesc[i]);
    }

  OUT_TRACE ("");
}

/**
* \fn void step_initreduction (void *variable, void *variable_reduc, int *op,
		    STEP_Datatype * type)

* \brief Initializes a reduction and save the initial value of *variable
* \param[in] *variable the variable to save
* \param[in,out] *variable_reduc where to save the copy
* \param[in] *op the reduction's operator
* \param[in] *type the data type of *variable
* \return none
*/
void
step_initreduction (void *variable, void *variable_reduc, int *op,
		    STEP_Datatype * type)
{

  IN_TRACE ("var =0x%8.8X var_reduc = 0x%8.8X, op = %d, type=%d", variable,
	    variable_reduc, *op, *type);
  if (*type == STEP_INTEGER1)
    {
      memcpy (variable_reduc, variable, sizeof (int8_t));
      switch (*op)
	{
	case STEP_SUM:
	  *((int8_t *) variable) = 0;
	  break;
	case STEP_PROD:
	  *((int8_t *) variable) = 1;
	  break;
	case STEP_MAX_:
	  *((int8_t *) variable) = INT8_MIN;
	  break;
	case STEP_MIN_:
	  *((int8_t *) variable) = INT8_MAX;
	  break;
	default:
	  assert (0);
	}
      OUT_TRACE ("");
      return;
    }
  if (*type == STEP_INTEGER2)
    {
      memcpy (variable_reduc, variable, sizeof (int16_t));
      switch (*op)
	{
	case STEP_SUM:
	  *((int16_t *) variable) = 0;
	  break;
	case STEP_PROD:
	  *((int16_t *) variable) = 1;
	  break;
	case STEP_MAX_:
	  *((int16_t *) variable) = INT16_MIN;
	  break;
	case STEP_MIN_:
	  *((int16_t *) variable) = INT16_MAX;
	  break;
	default:
	  assert (0);
	}
      OUT_TRACE ("");
      return;
    }
  if ((*type == STEP_INTEGER4))
    {
      memcpy (variable_reduc, variable, sizeof (int32_t));
      switch (*op)
	{
	case STEP_SUM:
	  *((int32_t *) variable) = 0;
	  break;
	case STEP_PROD:
	  *((int32_t *) variable) = 1;
	  break;
	case STEP_MAX_:
	  *((int32_t *) variable) = INT32_MIN;
	  break;
	case STEP_MIN_:
	  *((int32_t *) variable) = INT32_MAX;
	  break;
	default:
	  assert (0);
	}
      OUT_TRACE ("");
      return;
    }
  if (*type == STEP_INTEGER8)
    {
      memcpy (variable_reduc, variable, sizeof (int64_t));
      switch (*op)
	{
	case STEP_SUM:
	  *((int64_t *) variable) = 0;
	  break;
	case STEP_PROD:
	  *((int64_t *) variable) = 1;
	  break;
	case STEP_MAX_:
	  *((int64_t *) variable) = INT64_MIN;
	  break;
	case STEP_MIN_:
	  *((int64_t *) variable) = INT64_MAX;
	  break;
	default:
	  assert (0);
	}
      OUT_TRACE ("");
      return;
    }
  if ((*type == STEP_REAL4) || (*type == STEP_REAL))
    {
      memcpy (variable_reduc, variable, sizeof (float));
      switch (*op)
	{
	case STEP_SUM:
	  *((float *) variable) = 0;
	  break;
	case STEP_PROD:
	  *((float *) variable) = 1;
	  break;
	case STEP_MAX_:
	  *((float *) variable) = FLT_MIN;
	  break;
	case STEP_MIN_:
	  *((float *) variable) = FLT_MAX;
	  break;
	default:
	  assert (0);
	}
      OUT_TRACE ("");
      return;
    }
  if ((*type == STEP_REAL8) || (*type == STEP_DOUBLE_PRECISION))
    {
      memcpy (variable_reduc, variable, sizeof (double));
      switch (*op)
	{
	case STEP_SUM:
	  *((double *) variable) = 0;
	  break;
	case STEP_PROD:
	  *((double *) variable) = 1;
	  break;
	case STEP_MAX_:
	  *((double *) variable) = DBL_MIN;
	  break;
	case STEP_MIN_:
	  *((double *) variable) = DBL_MAX;
	  break;
	default:
	  assert (0);
	}
      OUT_TRACE ("");
      return;
    }

  if ((*type == STEP_COMPLEX8) || (*type == STEP_COMPLEX))
    {
      memcpy (variable_reduc, variable, sizeof (complexe8_t));
      switch (*op)
	{
	case STEP_SUM:
	  ((complexe8_t *) variable)->rp = 0;
	  ((complexe8_t *) variable)->ip = 0;
	  break;
	case STEP_PROD:
	  ((complexe8_t *) variable)->rp = 1;
	  ((complexe8_t *) variable)->ip = 0;
	  break;
	case STEP_MAX_:
	  assert (0);
	case STEP_MIN_:
	  assert (0);
	default:
	  assert (0);
	}
      OUT_TRACE ("");
      return;
    }

  if (*type == STEP_COMPLEX16)
    {
      memcpy (variable_reduc, variable, sizeof (complexe16_t));
      switch (*op)
	{
	case STEP_SUM:
	  ((complexe16_t *) variable)->rp = 0;
	  ((complexe16_t *) variable)->ip = 0;
	  break;
	case STEP_PROD:
	  ((complexe16_t *) variable)->rp = 1;
	  ((complexe16_t *) variable)->ip = 0;
	  break;
	case STEP_MAX_:
	  assert (0);
	case STEP_MIN_:
	  assert (0);
	default:
	  assert (0);
	}
      OUT_TRACE ("");
      return;
    }

  if (*type == STEP_INTEGER)
    {
      memcpy (variable_reduc, variable, sizeof (int));
      switch (*op)
	{
	case STEP_SUM:
	  *((int *) variable) = 0;
	  break;
	case STEP_PROD:
	  *((int *) variable) = 1;
	  break;
	case STEP_MAX_:
	  *((int *) variable) = INT_MIN;
	  break;
	case STEP_MIN_:
	  *((int *) variable) = INT_MAX;
	  break;
	default:
	  assert (0);
	}
      OUT_TRACE ("");
      return;
    }
  /*No more data types are supported */
  assert (0);
}

/**
* \fn void step_reduction (void *variable, void *variable_reduc, 
                          int *op, STEP_Datatype * type)

* \brief Performs a reduction  and puts the result in *variable
* \param[in,out] *variable where to put the result
* \param[in] *variable_reduc the initial value of *variable
* \param[in] *op the reduction's operator
* \param[in] *type the data type of *variable
* \return none
*/
void
step_reduction (void *variable, void *variable_reduc, int *op, STEP_Datatype
		* type)
{
  IN_TRACE ("var =0x%8.8X var_reduc = 0x%8.8X, op = %d, type=%d", variable,
	    variable_reduc, *op, *type);

  if (*type == STEP_INTEGER1)	
    {
      int8_t tmp_buffer;
      switch (*op)
	{
	case STEP_SUM:
	  MPI_Allreduce (variable, &tmp_buffer, 1, MPI_CHAR, MPI_SUM,
			 MPI_COMM_WORLD);
	  *(int8_t *) variable = *(int8_t *) variable_reduc + tmp_buffer;
	  break;
	case STEP_PROD:
	  MPI_Allreduce (variable, &tmp_buffer, 1, MPI_CHAR, MPI_PROD,
			 MPI_COMM_WORLD);
	  *(int8_t *) variable = *(int8_t *) variable_reduc *tmp_buffer;
	  break;
	case STEP_MAX_:
	  MPI_Allreduce (variable, &tmp_buffer, 1, MPI_CHAR, MPI_MAX,
			 MPI_COMM_WORLD);
	  *(int8_t *) variable = MAX (*(int8_t *) variable_reduc, tmp_buffer);
	  break;
	case STEP_MIN_:
	  MPI_Allreduce (variable, &tmp_buffer, 1, MPI_CHAR, MPI_MIN,
			 MPI_COMM_WORLD);
	  *(int8_t *) variable = MIN (*(int8_t *) variable_reduc, tmp_buffer);
	  break;
	default:
	  assert (0);
	}
      OUT_TRACE ("");
      return;
    }
  if (*type == STEP_INTEGER2)
    {
      int16_t tmp_buffer;

      switch (*op)
	{
	case STEP_SUM:
	  MPI_Allreduce (variable, &tmp_buffer, 1, MPI_INTEGER2, MPI_SUM,
			 MPI_COMM_WORLD);
	  *(int16_t *) variable = *(int16_t *) variable_reduc + tmp_buffer;
	  break;
	case STEP_PROD:
	  MPI_Allreduce (variable, &tmp_buffer, 1, MPI_INTEGER2, MPI_PROD,
			 MPI_COMM_WORLD);
	  *(int16_t *) variable = *(int16_t *) variable_reduc *tmp_buffer;
	  break;
	case STEP_MAX_:
	  MPI_Allreduce (variable, &tmp_buffer, 1, MPI_INTEGER2, MPI_MAX,
			 MPI_COMM_WORLD);
	  *(int16_t *) variable =
	    MAX (*(int16_t *) variable_reduc, tmp_buffer);
	  break;
	case STEP_MIN_:
	  MPI_Allreduce (variable, &tmp_buffer, 1, MPI_INTEGER2, MPI_MIN,
			 MPI_COMM_WORLD);
	  *(int16_t *) variable =
	    MIN (*(int16_t *) variable_reduc, tmp_buffer);
	  break;
	default:
	  assert (0);
	}
      OUT_TRACE ("");
      return;
    }

  if (*type == STEP_INTEGER4)
    {
      int32_t tmp_buffer;
      switch (*op)
	{
	case STEP_SUM:
	  MPI_Allreduce (variable, &tmp_buffer, 1, MPI_INTEGER4, MPI_SUM,
			 MPI_COMM_WORLD);
	  *(int32_t *) variable = *(int32_t *) variable_reduc + tmp_buffer;
	  break;
	case STEP_PROD:
	  MPI_Allreduce (variable, &tmp_buffer, 1, MPI_INTEGER4, MPI_PROD,
			 MPI_COMM_WORLD);
	  *(int32_t *) variable = *(int32_t *) variable_reduc *tmp_buffer;
	  break;
	case STEP_MAX_:
	  MPI_Allreduce (variable, &tmp_buffer, 1, MPI_INTEGER4, MPI_MAX,
			 MPI_COMM_WORLD);
	  *(int32_t *) variable =
	    MAX (*(int32_t *) variable_reduc, tmp_buffer);
	  break;
	case STEP_MIN_:
	  MPI_Allreduce (variable, &tmp_buffer, 1, MPI_INTEGER4, MPI_MIN,
			 MPI_COMM_WORLD);
	  *(int32_t *) variable =
	    MIN (*(int32_t *) variable_reduc, tmp_buffer);
	  break;
	default:
	  assert (0);
	}
      OUT_TRACE ("");
      return;
    }

  if (*type == STEP_INTEGER8)
    {
      int64_t tmp_buffer = 0;
      switch (*op)
	{
	case STEP_SUM:
	  MPI_Allreduce (variable, &tmp_buffer, 1, MPI_INTEGER8, MPI_SUM,
			 MPI_COMM_WORLD);
	  *(int64_t *) variable = *(int64_t *) variable_reduc + tmp_buffer;
	  break;
	case STEP_PROD:
	  MPI_Allreduce (variable, &tmp_buffer, 1, MPI_INTEGER8, MPI_PROD,
			 MPI_COMM_WORLD);
	  *(int64_t *) variable = *(int64_t *) variable_reduc *tmp_buffer;

	  break;
	case STEP_MAX_:
	  MPI_Allreduce (variable, &tmp_buffer, 1, MPI_INTEGER8, MPI_MAX,
			 MPI_COMM_WORLD);
	  *(int64_t *) variable =
	    MAX (*(int64_t *) variable_reduc, tmp_buffer);
	  break;
	case STEP_MIN_:
	  MPI_Allreduce (variable, &tmp_buffer, 1, MPI_INTEGER8, MPI_MIN,
			 MPI_COMM_WORLD);
	  *(int64_t *) variable =
	    MIN (*(int64_t *) variable_reduc, tmp_buffer);
	  break;
	default:
	  assert (0);
	}
      OUT_TRACE ("");
      return;
    }
  if (*type == STEP_INTEGER)
    {
      int tmp_buffer = 0;
      switch (*op)
	{
	case STEP_SUM:
	  MPI_Allreduce (variable, &tmp_buffer, 1, MPI_INTEGER, MPI_SUM,
			 MPI_COMM_WORLD);
	  *(int *) variable = *(int *) variable_reduc + tmp_buffer;
	  break;
	case STEP_PROD:
	  MPI_Allreduce (variable, &tmp_buffer, 1, MPI_INTEGER, MPI_PROD,
			 MPI_COMM_WORLD);
	  *(int *) variable = *(int *) variable_reduc *tmp_buffer;

	  break;
	case STEP_MAX_:
	  MPI_Allreduce (variable, &tmp_buffer, 1, MPI_INTEGER, MPI_MAX,
			 MPI_COMM_WORLD);
	  *(int *) variable = MAX (*(int *) variable_reduc, tmp_buffer);
	  break;
	case STEP_MIN_:
	  MPI_Allreduce (variable, &tmp_buffer, 1, MPI_INTEGER, MPI_MIN,
			 MPI_COMM_WORLD);
	  *(int *) variable = MIN (*(int *) variable_reduc, tmp_buffer);
	  break;
	default:
	  assert (0);
	}
      OUT_TRACE ("");
      return;
    }
  if ((*type == STEP_REAL4) || (*type == STEP_REAL))
    {
      float tmp_buffer;
      switch (*op)
	{
	case STEP_SUM:
	  MPI_Allreduce (variable, &tmp_buffer, 1, MPI_REAL4, MPI_SUM,
			 MPI_COMM_WORLD);
	  *(float *) variable = *(float *) variable_reduc + tmp_buffer;
	  break;
	case STEP_PROD:
	  MPI_Allreduce (variable, &tmp_buffer, 1, MPI_REAL4, MPI_PROD,
			 MPI_COMM_WORLD);
	  *(float *) variable = *(float *) variable_reduc *tmp_buffer;
	  break;
	case STEP_MAX_:
	  MPI_Allreduce (variable, &tmp_buffer, 1, MPI_REAL4, MPI_MAX,
			 MPI_COMM_WORLD);
	  *(float *) variable = MAX (*(float *) variable_reduc, tmp_buffer);
	  break;
	case STEP_MIN_:
	  MPI_Allreduce (variable, &tmp_buffer, 1, MPI_REAL4, MPI_MIN,
			 MPI_COMM_WORLD);
	  *(float *) variable = MIN (*(float *) variable_reduc, tmp_buffer);
	  break;
	default:
	  assert (0);
	}
      OUT_TRACE ("");
      return;
    }
  if ((*type == STEP_REAL8) || (*type == STEP_DOUBLE_PRECISION))
    {
      double tmp_buffer;
      switch (*op)
	{
	case STEP_SUM:
	  MPI_Allreduce (variable, &tmp_buffer, 1, MPI_REAL8, MPI_SUM,
			 MPI_COMM_WORLD);
	  *(double *) variable = *(double *) variable_reduc + tmp_buffer;
	  break;
	case STEP_PROD:
	  MPI_Allreduce (variable, &tmp_buffer, 1, MPI_REAL8, MPI_PROD,
			 MPI_COMM_WORLD);
	  *(double *) variable = *(double *) variable_reduc *tmp_buffer;
	  break;
	case STEP_MAX_:
	  MPI_Allreduce (variable, &tmp_buffer, 1, MPI_REAL8, MPI_MAX,
			 MPI_COMM_WORLD);
	  *(double *) variable = MAX (*(double *) variable_reduc, tmp_buffer);
	  break;
	case STEP_MIN_:
	  MPI_Allreduce (variable, &tmp_buffer, 1, MPI_REAL8, MPI_MIN,
			 MPI_COMM_WORLD);
	  *(double *) variable = MIN (*(double *) variable_reduc, tmp_buffer);
	  break;
	default:
	  assert (0);
	}
      OUT_TRACE ("");
      return;
    }
  if ((*type == STEP_COMPLEX8) || (*type == STEP_COMPLEX))
    {
      complexe8_t tmp_buffer;
      switch (*op)
	{
	case STEP_SUM:
	  MPI_Allreduce (variable, &tmp_buffer, 1, MPI_COMPLEX, MPI_SUM,
			 MPI_COMM_WORLD);
	  ((complexe8_t *) variable)->rp =
	    ((complexe8_t *) variable_reduc)->rp + tmp_buffer.rp;
	  ((complexe8_t *) variable)->ip =
	    ((complexe8_t *) variable_reduc)->ip + tmp_buffer.ip;
	  break;
	case STEP_PROD:
	  assert (0);
	case STEP_MAX_:
	  assert (0);
	case STEP_MIN_:
	  assert (0);
	default:
	  assert (0);
	}
      OUT_TRACE ("");
      return;
    }
  if (*type == STEP_COMPLEX16)
    {

      complexe16_t tmp_buffer;
      switch (*op)
	{
	case STEP_SUM:
	  MPI_Allreduce (variable, &tmp_buffer, 1, MPI_DOUBLE_COMPLEX,
			 MPI_SUM, MPI_COMM_WORLD);
	  ((complexe16_t *) variable)->rp =
	    ((complexe16_t *) variable_reduc)->rp + tmp_buffer.rp;
	  ((complexe16_t *) variable)->ip =
	    ((complexe16_t *) variable_reduc)->ip + tmp_buffer.ip;
	  break;
	case STEP_PROD:
	  assert (0);
	case STEP_MAX_:
	  assert (0);
	case STEP_MIN_:
	  assert (0);
	default:
	  break;
	}
      OUT_TRACE ("");
      return;
    }
  assert (0);
}

/**
* \fn void step_mastertoallscalar (void *scalar, int *max_nb_request,
			MPI_Request * requests, int *nb_request,
			int *algorithm, STEP_Datatype * type)


* \brief Propagates the value of *sacalar from the master process 
         to all other processes
* \param[in,out]  *scalar the value to propagate/to receive
* \param[in]  *max_nb_request the maximum number of requests
* \param[out] *requests array of MPI_Request
* \param[out] *nb_request the number of request in requests
* \param[in]  *algorithm the algorithm to be used for data propagation 
* \param[in]  *type the data type of *scalar
* \return none
*/
void
step_mastertoallscalar (void *scalar, int *max_nb_request,
			MPI_Request * requests, int *nb_request,
			int *algorithm, STEP_Datatype * type)
{
  IN_TRACE
  ("scalar=0x%8.8X \t max_nb_request = %d\t nb_request = %d, algorithm = %d ,type= %d",
   scalar, max_nb_request, *nb_request, *algorithm, *type);
  MPI_Bcast (scalar, 1, step_types_table[*type], 0, MPI_COMM_WORLD);
  OUT_TRACE ("");
}

/**
* \fn void step_initinterlaced (int *size, void *array, void *array_initial,
		     void *array_buffer, STEP_Datatype * type)
* \brief This routine must be used in conjonction with the merge version of 
         alltoall_region.\n It makes a copy of *array in *array_initial
* \param[in]  *size            For backward compatibility
* \param[in]  *array           The array to be saved
* \param[out] *array_initial   Where to save the array
* \param[in]  *array_buffer    For backward compatibility
* \param[in]  *type            the datatype of *array
* \return none
*/
void
step_initinterlaced (int *size, void *array, void *array_initial,
		     void *array_buffer, STEP_Datatype * type)
{
  int copy_size;;
  IN_TRACE
    ("size = %d array = 0x%8.8X array_initial = 0x%8.8X array_buffer = 0x%8.8X type = %d",
    *size, array, array_buffer, type);
  MPI_Type_size (step_types_table[*type], &copy_size);
  memcpy (array_initial, array, copy_size * (*size));
  OUT_TRACE ("");
}

/**
* \fn void step_mastertoallregion(void *array, int *dim, int *regions, 
                        int *size, int *max_nb_request, MPI_Request * requests,
			int *nb_request, int *algorithm, STEP_Datatype * type)

* \brief Propagate an array's region from the master process to the all others
* \param[in,out]  *array the data array 
* \param[in]  *dim number of dimensions in *array
* \param[in]  *region regions descriptor 
* \param[in]  *size unused
* \param[in]  *max_nb_request the maximum number of requests
* \param[out] *requests array of MPI_Request
* \param[out] *nb_request the number of request in *requests
* \param[in]  *algorithm the algorithm to be used for data propagation 
* \param[in]  *type the data type of *array
* \return none
*/
void
step_mastertoallregion (void *array, int *dim, int *regions, int *size,
			int *max_nb_request, MPI_Request * requests,
			int *nb_request, int *algorithm, STEP_Datatype * type)
{
  int i, j, k, req_count;
  int dim_sizes[MAX_DIMS];
  int dim_sub_sizes[1][MAX_DIMS];
  int dim_starts[1][MAX_DIMS];
  MPI_Datatype region_mpidesc;
  int have_iteration[MAX_PROCESSES];
  IN_TRACE
    ("array = 0x%8.8X dim = %d regions = 0x%8.8X size = %d max_nb_request=%d nb_request=%d algorithm =%d type = %d",
       array, *dim, regions, *size, 
      *max_nb_request, *nb_request, *algorithm,*type);

  if (step_.language == STEP_FORTRAN)
    {

      for (i = 0; i < *dim; i++)
	dim_sizes[i] = regions[1 + 2 * i] - regions[2 * i] + 1;

      have_iteration[0] = 0;
      for (j = 0; j < *dim; j++)	// for all dims
	{
	  dim_sub_sizes[0][j] = regions[1 + j * 2] - regions[j * 2] + 1;

	  dim_starts[0][j] = regions[2 * j] - 1;

	  if (dim_sub_sizes[0][j] > 0)
	    have_iteration[0]++;
	}
      if (have_iteration[0] >= *dim)
	have_iteration[0] = 1;
      else
	have_iteration[0] = 0;

      if (have_iteration[0])
	{
	  MPI_Type_create_subarray (*dim, &(dim_sizes[0]),
				    &(dim_sub_sizes[0][0]),
				    &(dim_starts[0][0]), MPI_ORDER_FORTRAN,
				    step_types_table[*type], &region_mpidesc);
	  MPI_Type_commit (&region_mpidesc);
	}


    }
  else				/*C language */
    {
      assert (0);
    }
  /* Default algorithm uses MPI_Bcast but we may have other implementations */
  switch (*algorithm)
    {
    default:
      MPI_Bcast (array, 1, region_mpidesc, 0, MPI_COMM_WORLD);
    }
  MPI_Type_free (&region_mpidesc);
  OUT_TRACE ("");
}

#ifdef TEST
#include "timings.h"
#define master_print(...)  if (my_id==1) fprintf(stdout,__VA_ARGS__)
#define process_print(...) fprintf(stdout,__VA_ARGS__)
#define STYLE_DISJOINT    0
#define STYLE_OVERLAP     1
char alg_name[][20] =
  { "STEP_NBLOCKING_ALG", "STEP_BLOCKING_ALG_1", "STEP_BLOCKING_ALG_2",
  "STEP_BLOCKING_ALG_3", "STEP_BLOCKING_ALG_4"
};

int data_array[16 * MAX_PROCESSES][16 * MAX_PROCESSES];
int *recv_buffer, *initial;
int bounds[MAX_PROCESSES][2];
int my_id;
int start, end, incr, dim;
int comm_tag, max_nb_request, nb_request;
int regions_desc[MAX_PROCESSES][2][2];
int i, j, k, l;
int process_nb;
int lang;
int dims_count, algorithme, type;
MPI_Request requests[MAX_PROCESSES];

void
init_tests ()
{
  step_init_fortran_order ();
  step_get_rank (&my_id);
  step_get_size (&process_nb);
  start = 0;
  end = 16 * MAX_PROCESSES;
  incr = 1;
  dim = 2;
  comm_tag = 0;
  max_nb_request = MAX_PROCESSES;
  algorithme = STEP_BLOCKING_ALG_1;
  type = STEP_INTEGER4;

}


void
build_disjoint_regions ()
{
  /*MINs & MAXs values for each dim */
  regions_desc[0][0][0] = 1;
  regions_desc[0][0][1] = 16 * MAX_PROCESSES;
  regions_desc[0][1][0] = 1;
  regions_desc[0][1][1] = 16 * MAX_PROCESSES;
  /*Region descriptors */
  for (j = 1; j <= process_nb; j++)
    {
      regions_desc[j][0][0] = bounds[j - 1][0] + 1;
      regions_desc[j][0][1] = bounds[j - 1][1];
      regions_desc[j][1][0] = 1;
      regions_desc[j][1][1] = 16 * MAX_PROCESSES;
    }
}


void
build_overlapping_regions ()
{
  /*MINs & MAXs values for each dim */
  regions_desc[0][0][0] = 1;
  regions_desc[0][0][1] = 16 * MAX_PROCESSES;
  regions_desc[0][1][0] = 1;
  regions_desc[0][1][1] = 16 * MAX_PROCESSES;
  /*Region descriptors */
  for (j = 1; j <= process_nb; j++)
    {
      regions_desc[j][0][0] = 1;
      regions_desc[j][0][1] = 16 * MAX_PROCESSES;
      regions_desc[j][1][0] = 1;
      regions_desc[j][1][1] = 16 * MAX_PROCESSES;
    }
}


void
init_data (int style)
{
  /*Put somthing particular in the data array we will use those values later in 
     the test process */

  if (style == STYLE_DISJOINT)	//disjoint 
    {
      for (i = bounds[my_id][0]; i <= bounds[my_id][1]; i++)
	for (j = 0; j < 16 * MAX_PROCESSES; j++)
	  {
	    data_array[j][i] = my_id + 1;
	  }
    }
  else				//STYLE_OVERLAP
    {
      for (i = 0; i < 16 * MAX_PROCESSES; i++)
	for (j = 0; j < 16 * MAX_PROCESSES; j++)
	  {
	    if (i % process_nb == my_id)
	      data_array[j][i] = my_id + 1;
	    else
	      data_array[j][i] = 0;
	  }
    }

}


void
all_to_all_test ()
{
  for (algorithme = STEP_NBLOCKING_ALG; algorithme <= STEP_BLOCKING_ALG_3;
       algorithme++)
    {
      init_data (STYLE_DISJOINT);
      nb_request = 0;
      step_alltoallregion (&dim, &process_nb, &regions_desc[0][0][0],
			   &process_nb, &data_array[0][0], &comm_tag,
			   &max_nb_request, &requests[0], &nb_request,
			   &algorithme, &type);
      step_waitall (&nb_request, &requests[0]);
      /*check data */
      for (l = 0; l < process_nb; l++)
	for (i = bounds[l][0]; i < bounds[l][1]; i++)
	  for (j = 0; j < 16 * MAX_PROCESSES; j++)
	    {
	      if (data_array[j][i] != l + 1)
		{
		  process_print ("\n[PROCESS %d] Test failed\n", my_id);
		  step_finalize ();
		  exit (1);
		}
	      else
		{
		  if (l != my_id)
		    data_array[j][i] = 0;	//reset data for the next test
		}
	    }
      master_print ("Test with algorithm %s\t...\tsucceeded\n",
		    alg_name[algorithme]);
      step_barrier ();
    }
}


void
all_to_all_bench ()
{
  int inx, itr;
  int array_size = 10;
  int d_array[1024][1024];
  double stats_1[50], stats_2[50];
  start = 0;
  incr = 1;
  dim = 2;
  comm_tag = 0;
  max_nb_request = MAX_PROCESSES;
  algorithme = STEP_BLOCKING_ALG_1;
  type = STEP_INTEGER4;
  for (inx = 0; inx < 50; inx++)
    {
      end = array_size;
      step_computeloopslices (&start, &end, &incr, &process_nb, &process_nb,
			      &bounds[0][0]);
      /*All_to_All */
      /*MINs & MAXs values for each dim */
      regions_desc[0][0][0] = 1;
      regions_desc[0][0][1] = array_size;
      regions_desc[0][1][0] = 1;
      regions_desc[0][1][1] = array_size;
      /*Region descriptors */
      for (j = 1; j <= process_nb; j++)
	{
	  regions_desc[j][0][0] = bounds[j - 1][0] + 1;
	  regions_desc[j][0][1] = bounds[j - 1][1];
	  regions_desc[j][1][0] = 1;
	  regions_desc[j][1][1] = array_size;
	}
      timings_init ();
      for (itr = 0; itr < 10; itr++)
	{
	  step_alltoallregion (&dim, &process_nb, &regions_desc[0][0][0],
			       &process_nb, &d_array[0][0], &comm_tag,
			       &max_nb_request, &requests[0], &nb_request,
			       &algorithme, &type);
	  step_waitall (&nb_request, &requests[0]);
	}
      stats_1[inx] = timings_event () / 10;

      /* AlltoAll with Diff&Merge */
      for (j = 1; j <= process_nb; j++)
	{
	  regions_desc[j][0][0] = 1;
	  regions_desc[j][0][1] = array_size;
	  regions_desc[j][1][0] = 1;
	  regions_desc[j][1][1] = array_size;
	}
      timings_init ();
      for (itr = 0; itr < 10; itr++)
	{
	  step_alltoallregion_merge (&dim, &process_nb,
				     &regions_desc[0][0][0], &process_nb,
				     &d_array[0][0], d_array, d_array,
				     &comm_tag, &max_nb_request, &requests[0],
				     &nb_request, &algorithme, &type);
	  step_waitall (&nb_request, &requests[0]);
	}
      stats_2[inx] = timings_event () / 10;

      array_size += 20;
    }
  master_print
    ("\n+-----------------------------------------------------------------+\n");
  master_print
 ("|\tData size (Kb)\t|\tAlltoAll time(µsec)\t|\tAlltoAll_merge time(µsec) \t|");
  master_print
    ("\n+-----------------------------------------------------------------+\n");
  for (inx = 0; inx < 50; inx++)
    {
      master_print ("|\t%f\t|\t%f\t|\t%f\t|\n",
		    8 * (10 +
			 20 * (double) inx * (double) sizeof (int)) / 1024,
		    stats_1[inx], stats_2[inx]);
    }
  master_print
    ("+--------------------------------------------------------------------+\n");

}


void
all_to_all_merge_test ()
{
  recv_buffer =
    malloc (sizeof (int) * 16 * MAX_PROCESSES * 16 * MAX_PROCESSES);
  initial = malloc (sizeof (int) * 16 * MAX_PROCESSES * 16 * MAX_PROCESSES);
  assert (initial != NULL);
  assert (recv_buffer != NULL);

  for (algorithme = STEP_NBLOCKING_ALG; algorithme <= STEP_BLOCKING_ALG_3;
       algorithme++)
    {
      init_data (STYLE_OVERLAP);
      memset (initial, 0,
	      sizeof (int) * 16 * MAX_PROCESSES * 16 * MAX_PROCESSES);
      nb_request = 0;
      step_alltoallregion_merge (&dim, &process_nb, &regions_desc[0][0][0],
				 &process_nb, &data_array[0][0], initial,
				 recv_buffer, &comm_tag, &max_nb_request,
				 &requests[0], &nb_request, &algorithme,
				 &type);
      step_waitall (&nb_request, &requests[0]);
      /*check data */
      for (i = 0; i < 16 * MAX_PROCESSES; i++)
	for (j = 0; j < 16 * MAX_PROCESSES; j++)
	  {

	    if ((data_array[j][i]) != (i % process_nb + 1))
	      {
		process_print
		  ("\n[PROCESS %d] Test failed (%d,%d,[%d!=%d])\n", my_id, j,
		   i, data_array[j][i], (i % process_nb + 1));
		step_finalize ();
		exit (1);
	      }
	  }
      master_print ("Test with algorithm %s\t...\tsucceeded\n",
		    alg_name[algorithme]);
      step_barrier ();
    }

  free (recv_buffer);
  free (initial);
}


int
main ()
{
  init_tests ();
  step_barrier ();
  master_print ("\nRunning test with %d processes.....\n\n\n", process_nb);
  step_computeloopslices (&start, &end, &incr, &process_nb, &process_nb,
			  &bounds[0][0]);
  master_print
    ("+---------------------------------------------------------------+\n");
  master_print
    ("|        Performing tests for alltoall communications           |\n");
  master_print
    ("+---------------------------------------------------------------+\n");
  /* alltoall */
  build_disjoint_regions ();
  all_to_all_test ();
  /* alltoall_merge */
  master_print
    ("+---------------------------------------------------------------+\n");
  master_print
    ("|        Performing tests for alltoall communications (merge)   |\n");
  master_print
    ("+---------------------------------------------------------------+\n");
  build_overlapping_regions ();
  all_to_all_merge_test ();
  /* benchmarking alltoall */
  master_print
    ("+---------------------------------------------------------------+\n");
  master_print
    ("|        Performing performances test for alltoall              |\n");
  master_print
    ("+---------------------------------------------------------------+\n");
  all_to_all_bench ();
  /*test finished */
  step_finalize ();
  return 0;
}
#endif

