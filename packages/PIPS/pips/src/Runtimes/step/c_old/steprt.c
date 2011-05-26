/**
 *                                                                             
 *   \file             steprt.c
 *   \author           Abdellah Kouadri.
 *                     Daniel Millot.
 *                     Frédérique Silber-Chaussumier.
 *                     Alain Muller
 *   \date             04/02/2010
 *   \version          2.0
 *   \brief            This file contains core routines of the runtime
 *                     (C or Fortran)                           
 */


#include "steprt.h"
#include "steprt_comm_alg.h"	/*Communication algorithms*/

#include "trace.h"
#include <stdarg.h>
#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <limits.h>
#include <stdlib.h>

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
int
step_init_fortran_order_ ()
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
      step_init_hash_table();
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
void
step_finalize_ ()
{
  step_finalize ();
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
void
step_barrier_ ()
{
  step_barrier ();
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
int
step_get_size_ (int *size)
{
  return step_get_size (size);
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
int
step_get_rank_ (int *rank)
{
  return step_get_rank (rank);
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
void
step_computeloopslices_ (int *from, int *to, int *incr, int *nb_regions,
			 int *nb_proc, int *bounds)
{
  step_computeloopslices (from, to, incr, nb_regions, nb_proc, bounds);
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
int
step_sizeregion_ (int *dim, int *region)
{
  return step_sizeregion (dim, region);
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
void
step_waitall_ (int *NbReq, MPI_Request * Request)
{
  step_waitall (NbReq, Request);
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
void
step_initreduction_ (void *variable, void *variable_reduc, int *op,
		     STEP_Datatype * type)
{
  step_initreduction (variable, variable_reduc, op, type);
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
void
step_reduction_ (void *variable, void *variable_reduc, int *op, STEP_Datatype
		 * type)
{
  step_reduction (variable, variable_reduc, op, type);
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
void
step_mastertoallscalar_ (void *scalar, int *max_nb_request,
			 MPI_Request * requests, int *nb_request,
			 int *algorithm, STEP_Datatype * type)
{
  step_mastertoallscalar (scalar, max_nb_request,
			  requests, nb_request, algorithm, type);
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
  int i, j;
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


/******************************************************************************/
static void step_init_arrayregions_body(void *array,STEP_Datatype *type, int *dims, va_list args_list)
{
  if(step_hash_table_find_entry(array)==-1) /* not already in the table*/  
    {
      index_t bounds[2* (*dims)];
      int  i;

      for (i=0; i< 2* (*dims); i++)
	bounds[i] =  *(va_arg(args_list, index_t *));

      step_hash_table_insert_array(array, *dims, bounds, *type);
    }
}

void 
step_init_arrayregions(void *array,STEP_Datatype *type, int *dims, ...)
{
  va_list args_list;
  va_start (args_list, dims);
  step_init_arrayregions_body(array,type, dims, args_list);
  va_end(args_list);
}

void 
step_init_arrayregions_(void *array,STEP_Datatype *type, int *dims, ...)
{
  va_list args_list;
  va_start (args_list, dims);
  step_init_arrayregions_body(array,type, dims, args_list);
  va_end(args_list);
}

/******************************************************************************/

void 
step_set_sendregions(void *array, int *nb_workchunks, index_t *region)
{
  step_hash_table_add_region(array,region, *nb_workchunks, 0);
}
void 
step_set_sendregions_(void *array, int *nb_workchunks, index_t *region)
{
  step_hash_table_add_region(array,region, *nb_workchunks, 0);
}

/******************************************************************************/

void 
step_set_interlaced_sendregions(void *array, int *nb_workchunks, index_t *regions) 
{
  step_hash_table_add_region(array, regions, *nb_workchunks, 1);

  array_identifier_t *array_desciptor=&(step_hash_table.arrays_table[step_hash_table_find_entry(array)]);
  int type_size;

  if (array_desciptor->saved_data != NULL) 
    free(array_desciptor->saved_data);

  MPI_Type_size(step_types_table[array_desciptor->array_type], &type_size);
  array_desciptor->saved_data = malloc(type_size * array_desciptor->array_size);
  assert(array_desciptor->saved_data!=NULL);
  memcpy(array_desciptor->saved_data, array, type_size * array_desciptor->array_size); 
}
void 
step_set_interlaced_sendregions_(void *array, int *nb_workchunks, index_t *regions) 
{
  step_set_interlaced_sendregions(array, nb_workchunks, regions);
}

/******************************************************************************/
void 
step_set_recvregions(void *array, int *nb_workchunks, index_t *regions) 
{
  assert(array != NULL);
  int i = step_hash_table_find_entry(array);
  assert(i>=0);/* check that array id exists*/
  array_identifier_t *array_desciptor=&(step_hash_table.arrays_table[i]);
  int dims = array_desciptor->dims; 

  assert(array_desciptor->recv_region.nb_region == -1);
  array_desciptor->recv_region.nb_region = *nb_workchunks;
  array_desciptor->recv_region.regions = (index_t *)malloc(sizeof(index_t)*2*dims*(*nb_workchunks));
  array_desciptor->recv_region.is_interlaced = false;

  memcpy(array_desciptor->recv_region.regions, regions, sizeof(index_t)*2*dims*(*nb_workchunks));
}
void 
step_set_recvregions_(void *array, int *nb_workchunks, index_t *regions) 
{
  step_set_recvregions(array, nb_workchunks, regions);
}

/******************************************************************************/

static void alltoall_full(void *array, int *algorithm, int *tag)
{
  assert(array != NULL);
  int i = step_hash_table_find_entry(array);
  assert(i >= 0);
  array_identifier_t *array_desciptor=&(step_hash_table.arrays_table[i]);

  if (array_desciptor->nb_region_descriptor == 0) return;

  /* Exactly one SEND region  */
  assert(array_desciptor->nb_region_descriptor == 1);

  /* Initialize RECV region as the whole array*/
  {
    int i;
    int dims = array_desciptor->dims;
    int nb_workchunks = array_desciptor->region_descriptor[0]->nb_region;
    assert(array_desciptor->recv_region.nb_region == -1);
    array_desciptor->recv_region.nb_region = nb_workchunks;
    array_desciptor->recv_region.regions = (index_t *)malloc(sizeof(index_t)*2*dims*nb_workchunks);
    for (i=0; i<nb_workchunks; i++)
      memcpy(&(array_desciptor->recv_region.regions[2*dims*i]), array_desciptor->bounds, sizeof(index_t)*2*dims);
  }
  
  step_alltoall(array_desciptor, *algorithm, *tag);
}

void step_alltoall_full(void *array, int *algorithm, int *tag) 
{
  alltoall_full(array, algorithm, tag);
}
void step_alltoall_full_(void *array, int *algorithm, int *tag) 
{
  alltoall_full(array, algorithm, tag);
}
void step_alltoall_full_interlaced(void *array, int *algorithm, int *tag) 
{
  alltoall_full(array, algorithm, tag);
}
void step_alltoall_full_interlaced_(void *array, int *algorithm, int *tag) 
{
  alltoall_full(array, algorithm, tag);
}

/******************************************************************************/
static void alltoall_partial(void *array, int *algorithm, int *tag)
{
  assert(array != NULL);
  int i = step_hash_table_find_entry(array);
  assert(i >= 0);
  array_identifier_t *array_desciptor=&(step_hash_table.arrays_table[i]);

  if (array_desciptor->nb_region_descriptor == 0) 
    {
      array_desciptor->recv_region.nb_region = -1;
      return;
    }

  /* Exactly one SEND region  */
  assert(array_desciptor->nb_region_descriptor == 1);

  /* Some RECV region */
  if (array_desciptor->recv_region.nb_region == -1) return;

  compute_non_sent_regions(array_desciptor);

  step_alltoall(array_desciptor, *algorithm, *tag);
}
void step_alltoall_partial(void *array, int *algorithm, int *tag)
{
  alltoall_partial(array, algorithm, tag);
}
void step_alltoall_partial_(void *array, int *algorithm, int *tag)
{
  alltoall_partial(array, algorithm, tag);
}


/******************************************************************************/
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

/******************************************************************************/
#ifdef TEST_NEW
int
main()
{

#define N 10
  int A[N][N];
  index_t  SR_A_region[64];
  index_t  RR_A_region[64];
  int algorithm,tag;
  int dims;
  int ds1,de1;
  int ds2,de2;
  int inx1,inx2;
  int type = STEP_INTEGER;
  int nb_proc,my_id;
  dims    = 2;
  ds1     = 0;
  de1     = N-1;
  ds2     = 0;
  de2     = N-1;
  tag = 0;
  algorithm =  STEP_NBLOCKING_ALG;
  step_get_rank (&my_id);

  step_init_fortran_order();
  step_get_size (&nb_proc);
  step_init_arrayregions(A,&type,&dims,&ds1,&de1,&ds2,&de2);
  step_get_rank(&my_id);

  // Send regions
  for (inx1=0;inx1<nb_proc;inx1++)
    {
      SR_A_region[0+2*(0+dims*inx1)] = 0;
      SR_A_region[1+2*(0+dims*inx1)] = N-1;
      SR_A_region[0+2*(1+dims*inx1)] = (N / nb_proc)*inx1;
      if (inx1<nb_proc-1)
	SR_A_region[1+2*(1+dims*inx1)] = SR_A_region[0+2*(1+dims*inx1)]+(N / nb_proc)-1; 
      else
	SR_A_region[1+2*(1+dims*inx1)] = N-1; 
    }

  //Recv regions
  for (inx1=0;inx1<nb_proc;inx1++)
    {
      RR_A_region[0+2*(0+dims*inx1)] = 0;
      RR_A_region[1+2*(0+dims*inx1)] = N-1;

      if (inx1==0)
	RR_A_region[0+2*(1+dims*inx1)] = 0;
      else 
	RR_A_region[0+2*(1+dims*inx1)] = (N / nb_proc)*inx1-2;
      if (inx1<nb_proc-1)
	RR_A_region[1+2*(1+dims*inx1)] = (RR_A_region[0+2*(1+dims*inx1)]+(N / nb_proc)-1+2)%N; 
      else
	RR_A_region[1+2*(1+dims*inx1)] = N-1; 
    }

  //Data init
  memset(&(A[0][0]),0,sizeof(int)*N*N);
  for (inx1=SR_A_region[0+2*(0+dims*my_id)];inx1<=SR_A_region[1+2*(0+dims*my_id)];inx1++)
    for (inx2=SR_A_region[0+2*(1+dims*my_id)];inx2<=SR_A_region[1+2*(1+dims*my_id)];inx2++)
      {
	A[inx2][inx1] = (my_id+1);
      }

  step_set_sendregions(A,&nb_proc,SR_A_region);
  step_set_recvregions(A,&nb_proc,RR_A_region); 
  step_alltoall_partial(A,&algorithm, &tag);
        
  printf("\n Test finished ......\n");
  if (my_id==0)
    for (inx1=0;inx1<N;inx1++)
      {        
        for (inx2=0;inx2<N;inx2++)
	  {
	    printf("\t%d",A[inx2][inx1]); 
	  }
        printf("\n");
      }
  step_finalize();
}
#endif

