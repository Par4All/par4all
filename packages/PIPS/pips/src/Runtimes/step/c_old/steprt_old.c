#ifdef OLD
/*******************************************************************************
  from steprt.c
*******************************************************************************/

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
  int i, j, my_id;
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
  int i, j, my_id;
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
    case STEP_BLOCKING_ALG_4:	/*Snow ball algorithm not implemented yet 
				  step_alltoallregion_merge_Blocking_4 (data_array, initial, buffer,
				  *type, region_mpidesc,
				  have_iteration, nb_request,
				  requests, *dim, dim_sizes,
				  dim_starts, dim_sub_sizes,
				  step_.language);

				  break;*/
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



/******************************************************************************
  from steprt_comm_alg.c
******************************************************************************/


/**
* \fn void step_alltoallregion_NBlocking_2 (void *array, MPI_Datatype * region_desc,
				int *have_iteration, int *nb_request,
				MPI_Request * requests)

* \brief Second  version for the non blocking alltoall algorithm 
*        (not implemented yet)        
* \param [in,out] *array the data array to be exchanged
* \param [in] *region_desc The region descriptor
* \param [in] *have_iteration array that indicates if a given process has 
                   \n executed any iterations
* \param [in,out] *nb_request the number of requests  (Non zero)
* \param [out] *requests array of requests (used in step_waitall())
* \return none
*/

void
step_alltoallregion_NBlocking_2 (void *array, MPI_Datatype * region_desc,
				 int *have_iteration, int *nb_request,
				 MPI_Request * requests)
{
  IN_TRACE
    ("array = 0x%8.8X region_description = 0x%8.8X have_iteration = 0x%8.8X,nb_request = %d",
     array, region_desc, have_iteration, *nb_request);
  assert (0);
  OUT_TRACE ("");
}

/**
* \fn void step_alltoallregion_merge_Blocking_4 (void *array, void *initial,
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

* \brief Fourth  version for the blocking alltoall algorithm using diff&merge
         \n (not yet implemented)

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
step_alltoallregion_merge_Blocking_4 (void *array, void *initial,
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
  assert (0);
  OUT_TRACE ("");
}

/**
* \fn step_alltoallregion_merge_NBlocking_2 (void *array, void *initial,
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

* \brief Sencond non Blocking imlementation for the alltoall algorithm 
         using diff&merge
         \n this routine is called in the case of overlapping regions
         \n Not implemented yet
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
step_alltoallregion_merge_NBlocking_2 (void *array, void *initial,
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
  assert (0);
  OUT_TRACE ("");
}


/***********************************************************************************
  from addon.c : the all file
***********************************************************************************/


#endif
