/********************************************************************************
 *   Description	: 	STEP API                                        *
 *   Authors 		:  	Abdellah. Kouadri                               *
 *				Daniel Millot                                   *	
 *                              Fr√©d√©rique Silber-Chaussumier                   *
 *				                                                *
 *   Date		:       18/03/2009                                      *
 *						                                *
 *   File		:	step.h		                                *
 *									       	*
 *   Version		:     	1.0					        *
 *        									*
 ********************************************************************************/

#include "step.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "util/trace.h"


#undef STEP_DEBUG_REGIONS

/* Global variables */
Step_Array_Stack_t Array_stack[MAX_ARRAYS];
Step_Scalar_Desc_t Shared_scalar_List[MAX_SHARED];
int stack_top,shared_ptr;
static int STEP_loop_bounds[MAX_THREADS][2];
STEP_BYTE *heap;
int STEP_region_overlap[MAX_THREADS];
double com_time;
/* 
Performs required initializations
A call to this function is inserted 
by GCC at the beginning of main()
*/
int
STEP_Init ()
{
  if (MPI_Init (NULL, NULL) == MPI_SUCCESS)
    {
      VT_USER_START("Step_AlltoAll");
      heap = (STEP_BYTE *) malloc (MAX_SIZE);
      assert (heap != NULL);
      stack_top = 0;
      shared_ptr = 0;
      SET_TRACES ("traces", NULL, 1, 0);
      return MPI_SUCCESS;
    }
  else
    {
      return MPI_ERR_INTERN;
    }
}

/* 
Finalization. Calls are inserted 
before any 'return' in main().
Actually just a wrapper for 
MPI_Finalize
*/
int
STEP_Finalize ()
{
  int result;
  IN_TRACE ("");
  free (heap);
  VT_USER_END("Step_AlltoAll");
  result = MPI_Finalize ();
  OUT_TRACE ("result = %d", result);
  return result;
}

/*
Returns the unique identifier of the calling 
process 'thread' if under an hybrid execution 
hybrid means OpenMP+MPI
*/
int
STEP_Get_Comm_rank (int *rank)
{
  int result;
  IN_TRACE ("rank = %p", rank);
  result = MPI_Comm_rank (COMM, rank);
  OUT_TRACE ("result = %d", result);
  return result;
}

/* 
The number of process executing the parallel 
region
*/
int
STEP_Get_Comm_size (int *size)
{
  int result;
  IN_TRACE ("size = %p", size);
  result = MPI_Comm_size (COMM, size);
  OUT_TRACE ("result = %d", result);
  return result;
}

/*
Computes the new loop bounds for each calling process
for (i = initial_i;i<=maximum_i;i=i+incr) 
for (i = initial_i;i>=minimum_i;i=i+incr) 
NB. 1-Some parameters are not used and are here for future implementations
    2-Notice the non-strict inequalities
*/
int
STEP_Set_LoopBounds (int id, int initial_i, int maximum_i, int incr,
		     STEP_loop_direction direction, int *start, int *end,
		     int bounds[MAX_THREADS][2])
{

  int my_id, thread_count, inx, chunk_size, sharedw;
  IN_TRACE ("id = %d\tinit=%d\tmaximum=%d\t ", id, initial_i, maximum_i);
  MPI_Comm_rank (COMM, &my_id);
  STEP_Get_Comm_size (&thread_count);
  chunk_size = ((maximum_i - initial_i + 1) / (thread_count));
  sharedw = ((maximum_i - initial_i + 1) % (thread_count));


  for (inx = 0; inx < thread_count; inx++)
    {
       /*STARTS*/ if (inx == 0)
	{
	  STEP_loop_bounds[inx][0] = initial_i;
	  bounds[inx][0] = initial_i;
	}
      else
	{
	  STEP_loop_bounds[inx][0] = STEP_loop_bounds[inx - 1][1] + 1;
	  bounds[inx][0] = STEP_loop_bounds[inx - 1][1] + 1;
	}
       /*ENDS*/ if (inx < thread_count - 1)
	{
	  STEP_loop_bounds[inx][1] = STEP_loop_bounds[inx][0] + chunk_size;
	  bounds[inx][1] = STEP_loop_bounds[inx][0] + chunk_size;
	  /*Here we redistribute additional iterations */
	  if (sharedw > 0)
	    {
	      STEP_loop_bounds[inx][1]++;
/	      bounds[inx][1]++;
	      sharedw--;
	    }
	}
      else
	{
	  STEP_loop_bounds[inx][1] = maximum_i;
	  bounds[inx][1] = maximum_i;
	}
    }

  *start = STEP_loop_bounds[my_id][0];
  *end = STEP_loop_bounds[my_id][1];


  OUT_TRACE ("THREAD N∞%d\t LOOP_START = %d\t LOOP_END = %d\n", my_id,
	     *start, *end);
  return MPI_SUCCESS;
}



/*
This functions is called at the end of each parallel loop
It performs all necessary data exchanges and merges in order to 
ensure data consistency  
Executes as follow
1)SEND(to all others)==>2)RECV(from all othres)==>3)COMPUTE IF REGIONS OVERLAP
		==>IF NO OVERLAPP==>4)MERGE==>5)END
		==>IF OVERLAPP ==>4') DIFF==> 5)MERGE ==>6)END

*/
int
STEP_AlltoAll_Region (void *data_array, int rank, STEP_alg_type algorithm,
		      int base_type_size, int dims_count,
		      int dim_sizes[MAX_DIMS],
		      int dim_sub_sizes[MAX_THREADS][MAX_DIMS],
		      int dim_starts[MAX_THREADS][MAX_DIMS])
{
  MPI_Datatype send_region_desc;
  MPI_Datatype recv_region_desc;
  MPI_Datatype send_recv_type;
  MPI_Request Request[16];
  MPI_Status Status;
  int i, j;
  int thread_count = 0;
  int my_id = 0;
  int start_index = 0;
  int offset = 1;
  int disp;
  int dims_index[MAX_DIMS];
  int dims_stops[MAX_DIMS];
  STEP_BYTE *org_vals_byte;
  STEP_WORD *org_vals_word;
  STEP_DWORD *org_vals_dword;
  int local_dim_sizes[MAX_DIMS];
  int local_dim_sub_sizes[MAX_THREADS][MAX_DIMS];
  int local_dim_starts[MAX_THREADS][MAX_DIMS];
  double start_time = MPI_Wtime ();

  IN_TRACE ("rank = %d, algorithm = %d, base_type_size = %d, dims_count = %d",
	    rank, algorithm, base_type_size, dims_count)
  MPI_Comm_rank (COMM, &my_id);
  MPI_Comm_size (COMM, &thread_count);
  TRACE_P ("id = %d sur %d", my_id, thread_count);
  /* The only one supported for instance */
  if (algorithm != STEP_ALG_ALLTOALL)
    {
      assert (0);
    }
  memset (&local_dim_sizes[0], 0, sizeof (int) * MAX_DIMS);
  memset (&local_dim_starts[0][0], 0, sizeof (int) * MAX_THREADS * MAX_DIMS);
  memset (&local_dim_sub_sizes[0][0], 0,
	  sizeof (int) * MAX_THREADS * MAX_DIMS);
  memset (&STEP_region_overlap[0], 0, MAX_THREADS * sizeof (int));
  /*Gcc stores sizes, subsizes & starts in inverted order correct that here! */
  for (j = 0; j < thread_count; j++)
    {
      for (i = 0; i < dims_count; i++)
	{
	  local_dim_sizes[i] = dim_sizes[(dims_count - 1) - i];
	  local_dim_sub_sizes[j][i] = dim_sub_sizes[j][(dims_count - 1) - i];
	  local_dim_starts[j][i] = dim_starts[j][(dims_count - 1) - i];
	}
    }

/*COMPUTE OVERLAP BETWEEN REGIONS*/
  for (i = 0; i < thread_count; i++)
    {
      if (i != my_id)
	{
	  int overlapp = 1;
	  for (j = 0; j < dims_count; j++)
	    {
	      if (((local_dim_starts[my_id][j] >= local_dim_starts[i][j])
		   && (local_dim_starts[my_id][j] <=
		       local_dim_starts[i][j] + local_dim_sub_sizes[i][j] -
		       1))
		  || ((local_dim_starts[i][j] >= local_dim_starts[my_id][j])
		      && (local_dim_starts[i][j] <=
			  local_dim_starts[my_id][j] +
			  local_dim_sub_sizes[my_id][j] - 1)))
		{
		  overlapp = overlapp * 1;
		}
	      else
		{
		  overlapp = overlapp * 0;
		}
	    }
	  STEP_region_overlap[i] = overlapp;
	}
    }

/* SEND REGION */
  switch (base_type_size)
    {
    case sizeof (STEP_BYTE):
      send_recv_type = MPI_BYTE;
      break;
    case sizeof (STEP_WORD):
      send_recv_type = MPI_UNSIGNED_SHORT;
      break;
    case sizeof (STEP_DWORD):
      send_recv_type = MPI_UNSIGNED;
      break;
    case sizeof (STEP_QWORD):
      send_recv_type = MPI_UNSIGNED_LONG;
      break;
    default:
      assert (0);		/*No more types are handled */
    }



#ifdef STEP_DEBUG_REGIONS
  if (my_id == 1)
    {
      fprintf (stdout,
	       "\n_________________________________________________________________\n");
      for (j = 0; j < thread_count; j++)
	{
	  fprintf (stdout, "\n THREAD N∞%d  DIMS_COUNT = %d\n", j,
		   dims_count);
	  for (i = 0; i < dims_count; i++)
	    {
	      fprintf (stdout,
		       "\n\t\tDIMS N∞%d \t SIZE = %d\t START = %d \t END = %d",
		       i, local_dim_sizes[i], local_dim_starts[j][i],
		       local_dim_starts[j][i] + local_dim_sub_sizes[j][i] -
		       1);
	    }
	}
      fprintf (stdout,
	       "\n_________________________________________________________________\n");
    }
#endif


  MPI_Type_create_subarray (dims_count, &(local_dim_sizes[0]),
			    &(local_dim_sub_sizes[my_id][0]),
			    &(local_dim_starts[my_id][0]), MPI_ORDER_C,
			    send_recv_type, &send_region_desc);

  MPI_Type_commit (&send_region_desc);

/* Non-Blocking SEND */
  for (i = 0; i < thread_count; i++)
    {
      if ((i != my_id))
	{
	  MPI_Isend ((void *) data_array, 1, send_region_desc, i, i, COMM,
		     &Request[i]);
	}
    }

/*Blocking RECV & MERGE*/
  org_vals_byte = (STEP_BYTE *) STEP_Pop_array ();
  org_vals_word = (STEP_WORD *) org_vals_byte;
  org_vals_dword = (STEP_DWORD *) org_vals_byte;

  for (i = 0; i < thread_count; i++)
    if (i != my_id)
      {
	MPI_Type_create_subarray (dims_count, &(local_dim_sizes[0]),
				  &(local_dim_sub_sizes[i][0]),
				  &(local_dim_starts[i][0]), MPI_ORDER_C,
				  send_recv_type, &recv_region_desc);
	MPI_Type_commit (&recv_region_desc);
	/* RECV */
	MPI_Recv ((void *) heap, 1, recv_region_desc, i, my_id, COMM,
		  MPI_STATUS_IGNORE);
	/* DIFF & MERGE */
	start_index = 0;
	int offset[MAX_DIMS];
	int uu;
	for (uu = 0; uu < MAX_DIMS; uu++)
	  {
	    if (local_dim_sizes[uu] == 0)
	      {
		dims_stops[uu] = 1;
		offset[uu] = 1;
	      }
	    else
	      {
		dims_stops[uu] =
		  local_dim_starts[i][uu] + local_dim_sub_sizes[i][uu];
		offset[uu] = local_dim_sizes[uu];
	      }
	  }


/*DIFF and MERGE*/
	for (dims_index[0] = local_dim_starts[i][0];
	     dims_index[0] < dims_stops[0]; dims_index[0]++)
	  {
	    for (dims_index[1] = local_dim_starts[i][1];
		 dims_index[1] < dims_stops[1]; dims_index[1]++)
	      {
		for (dims_index[2] = local_dim_starts[i][2];
		     dims_index[2] < dims_stops[2]; dims_index[2]++)
		  {
		    for (dims_index[3] = local_dim_starts[i][3];
			 dims_index[3] < dims_stops[3]; dims_index[3]++)
		      {
			for (dims_index[4] = local_dim_starts[i][4];
			     dims_index[4] < dims_stops[4]; dims_index[4]++)
			  {
			    disp = dims_index[4] +
			      dims_index[3] * offset[4] +
			      dims_index[2] * offset[3] * offset[4] +
			      dims_index[1] * offset[2] * offset[3] *
			      offset[4] +
			      dims_index[0] * offset[1] * offset[2] *
			      offset[3] * offset[4];
			    if (STEP_region_overlap[i] == 0)
			      {
				 /*MERGE*/ switch (base_type_size)
				  {
				  case sizeof (STEP_BYTE):
				    ((STEP_BYTE *) data_array)[disp] =
				      ((STEP_BYTE *) heap)[disp];
				    break;
				  case sizeof (STEP_WORD):
				    ((STEP_WORD *) data_array)[disp] =
				      ((STEP_WORD *) heap)[disp];
				    break;
				  case sizeof (STEP_DWORD):
				    ((STEP_DWORD *) data_array)[disp] =
				      ((STEP_DWORD *) heap)[disp];
				    break;
				  case sizeof (STEP_QWORD):
				    ((STEP_QWORD *) data_array)[disp] =
				      ((STEP_QWORD *) heap)[disp];
				    break;
				  default:
				    assert (0);
				  }
			      }
			    else	/*DIFF AND MERGE */
			      {
				switch (base_type_size)
				  {
				  case sizeof (STEP_BYTE):
				    if ((((STEP_BYTE *) org_vals_byte)[disp]
					 != ((STEP_BYTE *) heap)[disp]))
				      ((STEP_BYTE *) data_array)[disp] =
					((STEP_BYTE *) heap)[disp];
				    break;
				  case sizeof (STEP_WORD):
				    if ((((STEP_WORD *) org_vals_byte)[disp]
					 != ((STEP_WORD *) heap)[disp]))
				      ((STEP_WORD *) data_array)[disp] =
					((STEP_WORD *) heap)[disp];
				    break;
				  case sizeof (STEP_DWORD):
				    if ((((STEP_DWORD *) org_vals_byte)[disp]
					 != ((STEP_DWORD *) heap)[disp]))
				      ((STEP_DWORD *) data_array)[disp] =
					((STEP_DWORD *) heap)[disp];
				    break;
				  case sizeof (STEP_QWORD):
				    if ((((STEP_QWORD *) org_vals_byte)[disp]
					 != ((STEP_QWORD *) heap)[disp]))
				      ((STEP_QWORD *) data_array)[disp] =
					((STEP_QWORD *) heap)[disp];
				    break;
				  default:
				    assert (0);
				  }
			      }
			  }
		      }
		  }
	      }
	  }
      }

  MPI_Type_free (&send_region_desc);
  MPI_Type_free (&recv_region_desc);
  MPI_Barrier (COMM);
  com_time = MPI_Wtime () - start_time;
  OUT_TRACE ("time = %f", com_time);
  return 0;
}


/*
This function saves an entire array in case we need it for the diff
We use a stack data structure because we may have to handle multiple 
arrays 
*/

void *
STEP_Push_array (void *array, int array_size, int base_type_size)
{

  Array_stack[stack_top].array =
    (void *) malloc (array_size * base_type_size);
  assert (Array_stack[stack_top].array != NULL);
  memcpy (Array_stack[stack_top].array, array, array_size * base_type_size);
  stack_top++;
  assert (stack_top < MAX_ARRAYS);
  return (void *) (Array_stack[stack_top].array);
}

/*
returns a pointer to the next array in the stack
*/

void *
STEP_Pop_array ()
{
  assert (stack_top > 0);
  stack_top--;
  assert (Array_stack[stack_top].array != NULL);
  return ((void *) Array_stack[stack_top].array);
}


/* 
Used for region's arithmetics 
*/
int
STEP_MIN (int v1, int v2)
{
  return ((v1 <= v2) ? : v1, v2);
}

int
STEP_MAX (int v1, int v2)
{
  return ((v1 >= v2) ? : v1, v2);
}

/*Performs the communication necessary for a reduction*/
int STEP_Reduction(int id,  STEP_Op op)
{

		
		unsigned char *recv_buff = malloc(sizeof(char)*Shared_scalar_List[id].size);

		if (Shared_scalar_List[id].sign) 
		{
			switch (Shared_scalar_List[id].size)
			{
			case 1 : 		MPI_Allreduce((void *)Shared_scalar_List[id].data,recv_buff, 1,MPI_CHAR , op, COMM);
									break;
			case 2 : 		MPI_Allreduce((void *)Shared_scalar_List[id].data,recv_buff, 1,MPI_SHORT, op, COMM);
									break;
			case 4 : 		MPI_Allreduce((void *)Shared_scalar_List[id].data,recv_buff, 1,MPI_INT, op, COMM);    
									break;
			default : 	assert(0);
									break; 	
			}
		}
		else
		{

			switch (Shared_scalar_List[id].size)
			{
			case 1 : 		MPI_Allreduce((void *)Shared_scalar_List[id].data,recv_buff, 1,MPI_UNSIGNED_CHAR , op, COMM);
									break;
			case 2 : 		MPI_Allreduce((void *)Shared_scalar_List[id].data,recv_buff, 1,MPI_UNSIGNED_SHORT, op, COMM);
									break;
			case 4 : 		MPI_Allreduce((void *)Shared_scalar_List[id].data,recv_buff, 1,MPI_UNSIGNED, op, COMM);    
									break;
			default : 	assert(0);
									break; 	
			}
		}
	memcpy((void *)Shared_scalar_List[id].data,recv_buff, Shared_scalar_List[id].size);
	free(recv_buff);	
	return (0);

}

int STEP_Share(void *data, int type_size, int sign)
{
	Shared_scalar_List[shared_ptr].data =	(void *) data;
	Shared_scalar_List[shared_ptr].size = type_size;
	Shared_scalar_List[shared_ptr].sign = sign;
	shared_ptr ++;
	return (shared_ptr-1);
}

int STEP_UnShare_All()
{
	shared_ptr = 0;
}


#ifdef TEST_
/* Only for testing*/
int
main ()
{
  int data_array[16 * MAX_THREADS][16 * MAX_THREADS];
  int dim_sizes[MAX_DIMS];
  int dim_sub_sizes[MAX_THREADS][MAX_DIMS];
  int dim_starts[MAX_THREADS][MAX_DIMS];
  int bounds[MAX_THREADS][2];
  int my_id;
  int start, end;
  int i, j, k, l;
  int process_nb;
  int dims_count;
  /*
     Following code is equivalent to code generated by GCC for this loop
     #pragma omp parallel for
     for (i=0;i<16*MAX_THREADS;i++)
     for (j=0;j<16*MAX_THREADS;j++)
     {
     data_array[i][j] = my_id+1;

     }
   */
  /* Code starts here --> */
  STEP_Init ();
  STEP_Get_Comm_rank (&my_id);
  STEP_Get_Comm_size (&process_nb);
  STEP_Set_LoopBounds (my_id, 0, 16 * MAX_THREADS - 1, 1, LOOP_DIRECTION_UP,
		       &start, &end, bounds);
  STEP_Push_array ((void *) &data_array[0][0],
		   16 * MAX_THREADS * 16 * MAX_THREADS, sizeof (int));
  if (my_id == 0)
    {
      fprintf (stdout, "Running test with %d processes.....\n", process_nb);
    }
  //Put somthing particular in the data array we will use those values later in the test process
  for (i = start; i <= end; i++)
    for (j = 0; j < 16 * MAX_THREADS; j++)
      {
	data_array[i][j] = my_id + 1;
      }
  /* gcc builds this way the region descriptor for the current process (notice : inverted order corrected) */
  dim_sizes[1] = 16 * MAX_THREADS;
  dim_sizes[0] = 16 * MAX_THREADS;
  dim_sub_sizes[my_id][1] = end - start + 1;
  dim_sub_sizes[my_id][0] = 16 * MAX_THREADS;
  dim_starts[my_id][0] = 0;
  dim_starts[my_id][1] = start;
  dims_count = 2;
  /* and this way descriptors for the other processes */
  for (k = 0; k < process_nb; k++)
    {
      if (k != my_id)
	{
	  dim_sub_sizes[k][1] = bounds[k][1] - bounds[k][0] + 1;
	  dim_starts[k][1] = bounds[k][0];
	  dim_sub_sizes[k][0] = dim_sizes[0];
	  dim_starts[k][0] = 0;
	}

    }
  STEP_AlltoAll_Region ((void *) data_array, my_id, STEP_ALG_ALLTOALL,
			sizeof (int), 2, dim_sizes, dim_sub_sizes,
			dim_starts);
  /*Here we test that data communications and merging were okay */
  for (i = 0; i < 16 * MAX_THREADS; i++)
    for (j = 0; j < 16 * MAX_THREADS; j++)
      {
	for (l = 0; l < process_nb; l++)
	  if ((i >= bounds[l][0]) && (i <= bounds[l][1]))
	    {
	      k = l;
	      break;
	    }
	if (data_array[i][j] != k + 1)
	  {
	    fprintf (stdout,
		     "Test failed in process with ID %d [i=%d,j=%d,val = %d (%d)]\n",
		     my_id, i, j, data_array[i][j], k + 1);
	    STEP_Finalize ();
	    return (-1);
	  }
      }
  fprintf (stdout, "Test was successful in process with ID %d\n", my_id);
 
   /* Reduction test*/
   MPI_Barrier(COMM);
   if (my_id==0)
   {
    fprintf(stdout,"Running now Reduction Test with %d processes.....\n",process_nb);
   }

   start = my_id;
   int id = STEP_Share(&start,sizeof(int),1);
   STEP_Reduction(id,STEP_SUM);
   
   int n = process_nb-1;
   if (start == n*(n+1)/2)
   {
	fprintf(stdout,"Reduction test was successful in process with ID∞%d\n",my_id);
   }
   else
   {
	fprintf(stdout,"Reduction test failed in process with ID∞%d [%d - %d]\n",my_id,start,n*(n+1)/2);
   }



  STEP_Finalize ();
  return 0;
}
#endif
