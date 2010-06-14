/**
*                                                                             
*   \file             steprt_private.c
*   \author           Abdellah Kouadri.
*                     Daniel Millot.
*                     Frédérique Silber-Chaussumier.
*   \date             22/10/2009
*   \version          1.1
*   \brief            Variables and functions declarations used internally 
*                     in the runtime
*/


#include "steprt_private.h"


MPI_Request 		Reqs_list[MAX_REQUESTS];
struct step_internals 	step_;
MPI_Comm 		step_comm;


#define CHECK_IF_ZERO(a,b) (a<b)?1:0

/**
* \brief Given size of each dimension and indexes this macro
* computes the linear offset in a column major ordered array 
*/
#define COL_MAJOR_LINEAR_OFFSET(offset,s,i) \
        offset = i[0]+s[0]*(i[1]+s[1]*(i[2]+s[2]*(i[3]+s[3]*(i[4]+s[4]*i[5]))))

/**
* \brief Make a differential merge of buff2 and buff1 
*  puts the result in buff1
*/
#define DIFF_AND_MERGE(datatype,offset,init,buff1,buff2) \
        if (((datatype *) init)[offset] != \
		  ((datatype *) buff2)[offset])\
		((datatype *) buff1)[offset] =\
		  ((datatype *) buff2)[offset]


/**
* \brief Increments indices of a multidimensional array 
* starting from dimesnion 0 to dimesnion N
*/
#define INC_INDICES(ind,s,e) \
        if (ind[0] < e[0] - 1) ind[0]++;else{ind[0] = s[0];\
        if (ind[1] < e[1] - 1) ind[1]++;else{ind[1] = s[1];\
        if (ind[2] < e[2] - 1) ind[2]++;else{ind[2] = s[2];\
        if (ind[3] < e[3] - 1) ind[3]++;else{ind[3] = s[3];\
        if (ind[4] < e[4] - 1) ind[4]++;else{ind[4] = s[4];\
        if (ind[5] < e[5] - 1) ind[5]++;else ind[5] = s[5];}}}}}


/**
* \brief Handles case where  MPI_REAL16 and MPI_COMPLEX32 are not available
*/
#ifndef  MPI_REAL16
#define  MPI_REAL16      MPI_LONG_DOUBLE
#endif
#ifndef  MPI_COMPLEX32
#define  MPI_COMPLEX32   MPI_LONG_DOUBLE
#endif



/**
* \brief Indirection table used to allow an exact match
*  between C, Fortran and MPI data types  
*/
MPI_Datatype step_types_table[STEP_MAX_TYPES] = {
  STEP_TYPE_UNDIFINED,		/*Since STEP_INTEGER1==1 */
  MPI_INTEGER1,			/*STEP_INTEGER1 */
  MPI_INTEGER2,			/*STEP_INTEGER2 */
  MPI_INTEGER4,			/*STEP_INTEGER4 */
  MPI_INTEGER8,			/*STEP_INTEGER8 */
  MPI_REAL4,			/*STEP_REAL4 */
  MPI_REAL8,			/*STEP_REAL8 */
  MPI_REAL16,			/*STEP_REAL16 */
  MPI_COMPLEX8,			/*STEP_COMPLEX8 */
  MPI_COMPLEX16,		/*STEP_COMPLEX16 */
  MPI_COMPLEX32,		/*STEP_COMPLEX32 */
  MPI_INTEGER,			/*STEP_INTEGER */
  MPI_REAL,			/*STEP_REAL */
  MPI_COMPLEX,			/*STEP_COMPLEX */
  MPI_DOUBLE_PRECISION		/*STEP_DOUBLE_PRECISION */
};



/**
* \fn step_diff (void *buffer_1, void *buffer_2, void *initial, 
           STEP_Datatype type,int dims, int dim_sizes[MAX_DIMS],
	   int dim_starts_1[MAX_DIMS], int dim_starts_2[MAX_DIMS],
	   int dim_ss_1[MAX_DIMS], int dim_ss_2[MAX_DIMS], 
           int row_col)


* \brief This function performs a differential merge between
         two data arrays 
* \param [in,out] *buffer_1 data array of the calling process
* \param [in] *buffer_2 received data
* \param [in] *initial saved copy of the data array
* \param [in] *type the data type of the data array 
* \param [in] dims the number of dimensions
* \param [in] dim_sizes[MAX_DIMS] the size of each dimension 
* \param [in] dim_starts_1[MAX_DIMS] the regions starts on each dimension for 
              the calling process
* \param [in] dim_starts_2[MAX_DIMS] the regions starts on each dimension for 
              the  process we received data from
* \param [in] dim_ss_1[MAX_DIMS]     the regions size on each dimension for the  
              calling process
* \param [in] dim_ss_2[MAX_DIMS]     the regions size on each dimension for the
              process we received from
* \param [in] row_col        row or column major ordered array
*/

void
step_diff (void *buffer_1, void *buffer_2, void *initial, STEP_Datatype type,
	   int dims, int dim_sizes[MAX_DIMS],
	   int dim_starts_1[MAX_DIMS], int dim_starts_2[MAX_DIMS],
	   int dim_ss_1[MAX_DIMS], int dim_ss_2[MAX_DIMS], int row_col)
{


  IN_TRACE ("buffer_1 = 0x%8.8X buffer_2 = 0x%8.8X initial = 0x%8.8X ",
	    buffer_1, buffer_2, initial);
  IN_TRACE ("type = %d dims = %d ", type, dims);
  int index[MAX_DIMS], i, linear_offset;
  int non_zero_sizes[MAX_DIMS];
  if (row_col == STEP_C)	/* Row major order */
    {

      assert (0);
    }
  else if (row_col == STEP_FORTRAN)	/* Column major order */
    {

      int start[MAX_DIMS];
      int end[MAX_DIMS];
      start[0] =
	(MAX (dim_starts_1[0], dim_starts_2[0])) * (CHECK_IF_ZERO (0, dims));
      end[0] =
	(MIN (dim_ss_1[0] + dim_starts_1[0], dim_ss_2[0] + dim_starts_2[0])) *
	(CHECK_IF_ZERO (0, dims));

      start[1] =
	(MAX (dim_starts_1[1], dim_starts_2[1])) * CHECK_IF_ZERO (1, dims);
      end[1] =
	(MIN (dim_ss_1[1] + dim_starts_1[1], dim_ss_2[1] + dim_starts_2[1])) *
	(CHECK_IF_ZERO (1, dims));

      start[2] =
	(MAX (dim_starts_1[2], dim_starts_2[2])) * CHECK_IF_ZERO (2, dims);
      end[2] =
	(MIN (dim_ss_1[2] + dim_starts_1[2], dim_ss_2[2] + dim_starts_2[2])) *
	(CHECK_IF_ZERO (2, dims));

      start[3] =
	(MAX (dim_starts_1[3], dim_starts_2[3])) * CHECK_IF_ZERO (3, dims);
      end[3] =
	(MIN (dim_ss_1[3] + dim_starts_1[3], dim_ss_2[3] + dim_starts_2[3])) *
	(CHECK_IF_ZERO (3, dims));

      start[4] =
	(MAX (dim_starts_1[4], dim_starts_2[4])) * CHECK_IF_ZERO (4, dims);
      end[4] =
	(MIN (dim_ss_1[4] + dim_starts_1[4], dim_ss_2[4] + dim_starts_2[4])) *
	(CHECK_IF_ZERO (4, dims));

      start[5] =
	(MAX (dim_starts_1[5], dim_starts_2[5])) * CHECK_IF_ZERO (5, dims);
      end[5] =
	(MIN (dim_ss_1[5] + dim_starts_1[5], dim_ss_2[5] + dim_starts_2[5])) *
	(CHECK_IF_ZERO (5, dims));
      
   
      memcpy(index,start,sizeof (int) *MAX_DIMS);

      int overall = 1;
      if (start[0] < end[0])
	overall = overall * (end[0] - start[0]);
      if (start[1] < end[1])
	overall = overall * (end[1] - start[1]);
      if (start[2] < end[2])
	overall = overall * (end[2] - start[2]);
      if (start[2] < end[3])
	overall = overall * (end[3] - start[3]);
      if (start[4] < end[4])
	overall = overall * (end[4] - start[4]);
      if (start[5] < end[5])
	overall = overall * (end[5] - start[5]);

      int cpt = 0;
      switch (type)
	{

	case STEP_INTEGER:
	  for  (cpt=0;cpt < overall;cpt++)
	    {

	      COL_MAJOR_LINEAR_OFFSET(linear_offset,dim_sizes,index);
	      DIFF_AND_MERGE(int,linear_offset,initial,buffer_1,buffer_2);
	      INC_INDICES(index,start,end);
	    }

	  break;
	case STEP_INTEGER1:
	  for  (cpt=0;cpt < overall;cpt++)
	    {
   	      COL_MAJOR_LINEAR_OFFSET(linear_offset,dim_sizes,index);
	      DIFF_AND_MERGE(int8_t,linear_offset,initial,buffer_1,buffer_2);
              INC_INDICES(index,start,end);
	    }
	  break;
	case STEP_INTEGER2:
	  for  (cpt=0;cpt < overall;cpt++)
	    {
	      COL_MAJOR_LINEAR_OFFSET(linear_offset,dim_sizes,index);
	      DIFF_AND_MERGE(int16_t,linear_offset,initial,buffer_1,buffer_2);
  	      INC_INDICES(index,start,end);
	    }
	  break;
	case STEP_INTEGER4:
	  for  (cpt=0;cpt < overall;cpt++)
	    {
	      COL_MAJOR_LINEAR_OFFSET(linear_offset,dim_sizes,index);
	      DIFF_AND_MERGE(int32_t,linear_offset,initial,buffer_1,buffer_2);
  	      INC_INDICES(index,start,end);
	    }
	  break;
	case STEP_INTEGER8:
	  for  (cpt=0;cpt < overall;cpt++)
	    {
	      COL_MAJOR_LINEAR_OFFSET(linear_offset,dim_sizes,index);
	      DIFF_AND_MERGE(int64_t,linear_offset,initial,buffer_1,buffer_2);
  	      INC_INDICES(index,start,end);
	    }
	  break;
	case STEP_REAL:
	case STEP_REAL4:
	  for  (cpt=0;cpt < overall;cpt++)
	    {
  	      COL_MAJOR_LINEAR_OFFSET(linear_offset,dim_sizes,index);
	      DIFF_AND_MERGE(float,linear_offset,initial,buffer_1,buffer_2);
  	      INC_INDICES(index,start,end);
	    }
	  break;
	case STEP_REAL8:
	case STEP_DOUBLE_PRECISION:
	  for  (cpt=0;cpt < overall;cpt++)
	    {
	      COL_MAJOR_LINEAR_OFFSET(linear_offset,dim_sizes,index);;
	      DIFF_AND_MERGE(double,linear_offset,initial,buffer_1,buffer_2);
  	      INC_INDICES(index,start,end);
	    }
	  break;
	case STEP_REAL16:
	  assert (0);
	case STEP_COMPLEX:
	case STEP_COMPLEX8:
	  for  (cpt=0;cpt < overall;cpt++)
	    {
	      COL_MAJOR_LINEAR_OFFSET(linear_offset,dim_sizes,index);
	      if ((((complexe8_t *) initial)[linear_offset].rp
		   != ((complexe8_t *) buffer_2)[linear_offset].rp)
		  || (((complexe8_t *) initial)[linear_offset].ip
		      != ((complexe8_t *) buffer_2)[linear_offset].ip))
		((complexe8_t *) buffer_1)[linear_offset] =
		  ((complexe8_t *) buffer_2)[linear_offset];
  		INC_INDICES(index,start,end);
	    }
	  break;
	case STEP_COMPLEX16:
	  for  (cpt=0;cpt < overall;cpt++)
	    {
	      COL_MAJOR_LINEAR_OFFSET(linear_offset,dim_sizes,index);
	      if ((((complexe16_t *) initial)[linear_offset].rp !=
		   ((complexe16_t *) buffer_2)[linear_offset].rp)
		  || (((complexe16_t *) initial)[linear_offset].ip !=
		      ((complexe16_t *) buffer_2)[linear_offset].ip))
		((complexe16_t *) buffer_1)[linear_offset] =
		  ((complexe16_t *) buffer_2)[linear_offset];
  		INC_INDICES(index,start,end);
	    }
	  break;
	case STEP_COMPLEX32:
	  assert (0);
	default:
	  assert (0);
	}
    }
  else
    {
      assert (0);
    }
  OUT_TRACE ("");
}
