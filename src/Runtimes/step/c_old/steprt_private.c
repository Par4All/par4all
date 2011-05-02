/**
*                                                                             
*   \file             steprt_private.c
*   \author           Abdellah Kouadri.
*                     Daniel Millot.
*                     Frédérique Silber-Chaussumier.
*                     Alain Muller
*   \date             04/02/2010
*   \version          2.0
*   \brief            Variables and functions declarations used internally 
*                     in the runtime
*/


#include "steprt_private.h"


#include "trace.h"
#include <assert.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

MPI_Request 		        Reqs_list[MAX_REQUESTS];
struct step_internals 	        step_;





#define CHECK_IF_ZERO(a,b) (a<b)?1:0

/**
 * \brief Given size of each dimension and indexes this macro
 * computes the linear offset in a column major ordered array 
 */
#define COL_MAJOR_LINEAR_OFFSET(offset,s,i)				\
  offset = i[0]+s[0]*(i[1]+s[1]*(i[2]+s[2]*(i[3]+s[3]*(i[4]+s[4]*i[5]))))

/**
 * \brief Make a differential merge of buff2 and buff1 
 *  puts the result in buff1
 */
#define DIFF_AND_MERGE(datatype,offset,init,buff1,buff2)		\
  if (((datatype *) init)[offset] != ((datatype *) buff2)[offset])	\
    ((datatype *) buff1)[offset] = ((datatype *) buff2)[offset]


/**
 * \brief Increments indices of a multidimensional array 
 * starting from dimesnion 0 to dimesnion N
 */
#define INC_INDICES(ind,s,e)						\
  if (ind[0] < e[0] - 1) ind[0]++;else{ind[0] = s[0];			\
    if (ind[1] < e[1] - 1) ind[1]++;else{ind[1] = s[1];			\
      if (ind[2] < e[2] - 1) ind[2]++;else{ind[2] = s[2];		\
        if (ind[3] < e[3] - 1) ind[3]++;else{ind[3] = s[3];		\
	  if (ind[4] < e[4] - 1) ind[4]++;else{ind[4] = s[4];		\
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
  int index[MAX_DIMS], linear_offset;
  if (row_col == STEP_C)	/* Row major order */
    {
      assert (0);
    }
  else if (row_col == STEP_FORTRAN)	/* Column major order */
    {
      int start[MAX_DIMS];
      int end[MAX_DIMS];

      start[0] = (MAX (dim_starts_1[0], dim_starts_2[0])) * (CHECK_IF_ZERO (0, dims));
      end[0] = (MIN (dim_ss_1[0] + dim_starts_1[0], dim_ss_2[0] + dim_starts_2[0])) * (CHECK_IF_ZERO (0, dims));

      start[1] = (MAX (dim_starts_1[1], dim_starts_2[1])) * CHECK_IF_ZERO (1, dims);
      end[1] = (MIN (dim_ss_1[1] + dim_starts_1[1], dim_ss_2[1] + dim_starts_2[1])) * (CHECK_IF_ZERO (1, dims));

      start[2] = (MAX (dim_starts_1[2], dim_starts_2[2])) * CHECK_IF_ZERO (2, dims);
      end[2] = (MIN (dim_ss_1[2] + dim_starts_1[2], dim_ss_2[2] + dim_starts_2[2])) * (CHECK_IF_ZERO (2, dims));

      start[3] = (MAX (dim_starts_1[3], dim_starts_2[3])) * CHECK_IF_ZERO (3, dims);
      end[3] = (MIN (dim_ss_1[3] + dim_starts_1[3], dim_ss_2[3] + dim_starts_2[3])) * (CHECK_IF_ZERO (3, dims));

      start[4] = (MAX (dim_starts_1[4], dim_starts_2[4])) * CHECK_IF_ZERO (4, dims);
      end[4] = (MIN (dim_ss_1[4] + dim_starts_1[4], dim_ss_2[4] + dim_starts_2[4])) * (CHECK_IF_ZERO (4, dims));

      start[5] = (MAX (dim_starts_1[5], dim_starts_2[5])) * CHECK_IF_ZERO (5, dims);
      end[5] = (MIN (dim_ss_1[5] + dim_starts_1[5], dim_ss_2[5] + dim_starts_2[5])) * (CHECK_IF_ZERO (5, dims));
   
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


/******************************************************************************/

/**
 * \fn step_init_hash_table()
 * \brief This function initialized internats data structures
 */
void
step_init_hash_table()
{
  step_hash_table.nb_entries = 0;
  memset(step_hash_table.arrays_table, 0, sizeof(array_identifier_t)*MAX_NB_ARRAYS);
}

/******************************************************************************/

/**
 * \fn step_hash_table_insert_array(void *id,int dims, int *sizes)

 * \brief This function stores a new array descriptor 
 */
void
step_hash_table_insert_array(void *array, int dims, index_t *bounds,STEP_Datatype type)
{
  assert(array != NULL);
  assert(step_hash_table_find_entry(array) == -1); /* not already in the table*/  
  assert(step_hash_table.nb_entries<MAX_NB_ARRAYS); // il reste de la place

  array_identifier_t *array_desciptor;
  int i = 0;

  while(step_hash_table.arrays_table[i].used)
    i++;

  array_desciptor = &(step_hash_table.arrays_table[i]);
  array_desciptor->used = true;
  array_desciptor->array_id = array;
  array_desciptor->array_type = type;
  array_desciptor->dims = dims;
  array_desciptor->saved_data = NULL;
  array_desciptor->bounds=(index_t*) malloc(sizeof(index_t)*2*dims);
  array_desciptor->nb_region_descriptor = 0;
  array_desciptor->recv_region.nb_region = -1;
  array_desciptor->array_size = 1;
  for (i=0; i< dims; i++)
    array_desciptor->array_size *=  bounds[UP(i)] - bounds[LOW(i)] + 1;
  assert(array_desciptor->bounds != NULL);
  memcpy(array_desciptor->bounds, bounds, sizeof(index_t)*2*dims);

  step_hash_table.nb_entries++;
}
/******************************************************************************/

/**
 * \fn step_hash_table_add_region(void *id,index_t *region_desc, int nb_workchunks,int is_interlaced)

 * \brief This function adds a SEND region descriptor to an stored array
 */
void
step_hash_table_add_region(void *array, index_t *region_desc, int nb_workchunks, int is_interlaced)
{
  assert(array != NULL);
  int i = step_hash_table_find_entry(array);
  assert(i>=0); /* check that array id exists*/
  array_identifier_t *array_desciptor=&(step_hash_table.arrays_table[i]);
  int dims = array_desciptor->dims; 
  int j = array_desciptor->nb_region_descriptor;
  assert(j < MAX_REGIONS); // il reste de la plase

  array_desciptor->nb_region_descriptor++;
  array_desciptor->region_descriptor[j] = (region_descriptor_t*) malloc(sizeof(region_descriptor_t)); // ????
  array_desciptor->region_descriptor[j]->nb_region = nb_workchunks;
  array_desciptor->region_descriptor[j]->regions = (index_t*) malloc(sizeof(index_t)*2*dims*nb_workchunks);
  array_desciptor->region_descriptor[j]->is_interlaced = is_interlaced;
  memcpy(array_desciptor->region_descriptor[j]->regions, region_desc, sizeof(index_t)*2*dims*nb_workchunks);
}

/******************************************************************************/
void
step_hash_table_delete_partial_regions(void *array)
{
  assert(array != NULL);
  int j, i = step_hash_table_find_entry(array);
  assert(i >=0 );
  array_identifier_t *array_desciptor=&(step_hash_table.arrays_table[i]);
  array_desciptor->nb_non_updated_regions=0;

  for (i=0; i<MAX_PROCESSES; i++)
    for (j=0; j<MAX_PROCESSES; j++)
      if (array_desciptor->updated_non_communicated_regions[i][j] != NULL)
        {
	  if (array_desciptor->updated_non_communicated_regions[i][j]->regions != NULL)
	    {
	      free(array_desciptor->updated_non_communicated_regions[i][j]->regions);
	      array_desciptor->updated_non_communicated_regions[i][j]->regions = NULL;
	    }
	  free(array_desciptor->updated_non_communicated_regions[i][j]); 
	  array_desciptor->updated_non_communicated_regions[i][j] = NULL;
        }
}


void
step_hash_table_delete_send_region(void *array)
{
  assert(array != NULL);
  int i = step_hash_table_find_entry(array);
  assert(i >= 0); 
  step_hash_table.arrays_table[i].nb_region_descriptor=0;
  // il manque des free (cf step_hash_table_add_region)
}

void
step_hash_table_delete_recv_region(void *array)
{
  assert(array != NULL);
  int i = step_hash_table_find_entry(array);
  assert(i >= 0); 
  step_hash_table.arrays_table[i].recv_region.nb_region = -1;
  // il manque des free (cf step_set_recvregions)
}

/******************************************************************************/
/**
 * \fn step_hash_table_delete_array(void *id)

 * \brief This function deletes an array descriptor
 */
void
step_hash_table_delete_array(void *array)
{
  assert(array != NULL);
  int i = step_hash_table_find_entry(array);
  assert(i >= 0); 
  array_identifier_t *array_desciptor=&(step_hash_table.arrays_table[i]);

  array_desciptor->used = false;
  if (array_desciptor->recv_region.nb_region != -1) // pas clair au niveau des free
    free(array_desciptor->recv_region.regions);

  for (i=0; i<array_desciptor->nb_region_descriptor; i++)
    {
      free(array_desciptor->region_descriptor[i]->regions);
      free(array_desciptor->region_descriptor[i]);
    }
}

/******************************************************************************/
/**
 * \fn step_hash_table_find_entry(void *id)
 * \brief Return an index pointing to the array descriptor 
 */
int
step_hash_table_find_entry(void *array)
{
  assert(array != NULL);
  int i = 0;
  while ((i < MAX_NB_ARRAYS) &&
	 !(step_hash_table.arrays_table[i].used &&
	   step_hash_table.arrays_table[i].array_id == array)
	 )
    i++;

  if (i == MAX_NB_ARRAYS)
    return -1;
  else
    return i;
}
/******************************************************************************/
void
compute_non_sent_regions(array_identifier_t *array_desciptor)
{
  int i;
  int nb_regions;
  index_t *send_regions = array_desciptor->region_descriptor[0]->regions;
  index_t *recv_regions = array_desciptor->recv_region.regions;
  int dims = array_desciptor->dims;
  int workchunks = array_desciptor->recv_region.nb_region;
  int is_interlaced = array_desciptor->region_descriptor[0]->is_interlaced == 1;
  int rank = step_.rank_;

  if (workchunks<=1) return;
      
  for (i=0;i<workchunks;i++)
    if (i!=rank)
      {
        index_t *intersection = (index_t *) malloc(MAX_DIMS*2);
        index_t *comm_region =(index_t *) malloc(2*dims*16); // 16 ????
        /***********************************************************************/
        intersect_regions(dims, &(send_regions[rank*dims*2]), &(recv_regions[i*dims*2]), intersection);                
        if (!is_interlaced)
	  {        
	    diff_region(dims,intersection,&(send_regions[i*dims*2]),&nb_regions,comm_region);
	  }
        else
	  {
	    nb_regions = 1;
	    comm_region = intersection;
	  }

        if (nb_regions>0)
	  {
	    /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
	    /* calcul des regions modifiées en non envoyées et sauvegarde          */
	    int nb_sr;
	    index_t * updated_non_communicated = (index_t *) malloc(2*dims*16); // 16 ????
	    complement_region(dims,comm_region,&(send_regions[rank*dims*2]), nb_regions,
			      &nb_sr, updated_non_communicated);
        
	    if (nb_sr>0)
	      {
                if (array_desciptor->updated_non_communicated_regions[rank][i] == NULL)
		  {
		    array_desciptor->updated_non_communicated_regions[rank][i] = (region_descriptor_t*)malloc(sizeof(region_descriptor_t));
		    array_desciptor->updated_non_communicated_regions[rank][i]->is_interlaced = is_interlaced;
		    array_desciptor->updated_non_communicated_regions[rank][i]->nb_region = 0;
		    array_desciptor->updated_non_communicated_regions[rank][i]->regions = (index_t*)malloc(sizeof(index_t)*2*dims*32); // 32 ??? 
		  }
                int nb_ex = array_desciptor->updated_non_communicated_regions[rank][i]->nb_region;
                memcpy(&(array_desciptor->updated_non_communicated_regions[rank][i]->regions[2*dims*nb_ex]),
		       updated_non_communicated, sizeof(index_t)*2*dims*nb_sr);
                array_desciptor->updated_non_communicated_regions[rank][i]->nb_region = nb_ex+nb_sr;
	      }
	  }


        /***********************************************************************/
        intersect_regions(dims, &(send_regions[i*dims*2]), &(recv_regions[rank*dims*2]), intersection);                
        if (!is_interlaced)
	  {        
	    diff_region(dims,intersection,&(send_regions[rank*dims*2]),&nb_regions,comm_region);
	  }
        else
	  {
	    nb_regions = 1;
	    comm_region = intersection;
	  }

        if (nb_regions>0)
	  {
	    int nb_sr;
	    if (array_desciptor->updated_non_communicated_regions[i][rank] == NULL)
	      {
                array_desciptor->updated_non_communicated_regions[i][rank] = (region_descriptor_t*)malloc(sizeof(region_descriptor_t));
                array_desciptor->updated_non_communicated_regions[i][rank]->is_interlaced=is_interlaced;
                array_desciptor->updated_non_communicated_regions[i][rank]->nb_region=0;
                array_desciptor->updated_non_communicated_regions[i][rank]->regions = (index_t*)malloc(sizeof(index_t)*2*dims*32);
	      }
	    index_t * updated_non_communicated = (index_t *) malloc(2*dims*16);
	    complement_region(dims,comm_region,&(send_regions[i*dims*2]), nb_regions, &nb_sr, updated_non_communicated);
	    if (nb_sr>0)
	      {
                int nb_ex = array_desciptor->updated_non_communicated_regions[i][rank]->nb_region;
                memcpy(&(array_desciptor->updated_non_communicated_regions[i][rank]->regions[2*dims*nb_ex]),
		       updated_non_communicated,sizeof(index_t)*2*dims*nb_sr);
                array_desciptor->updated_non_communicated_regions[i][rank]->nb_region = nb_ex+nb_sr;
	      }
	  }
      }
}
/******************************************************************************/

/*
  transforme une union de region en un type MPI
*/
static void regions_to_MPI(int dim, index_t *array_bounds, int nb_region, index_t *regions, STEP_Datatype STEP_type, MPI_Datatype *MPI_type)
{
  int id_region;
  MPI_Datatype regions_type[nb_region];
  index_t array_sizes[dim];
  index_t array_of_starts[dim];
  index_t array_of_sizes[dim];
  int array_of_blocklengths[nb_region];
  MPI_Aint array_of_displacements[nb_region];

  BOUNDS_2_START_SIZES(dim, array_bounds, array_bounds, array_of_starts, array_sizes);

  if(nb_region !=0)
    {
      for(id_region=0; id_region<nb_region; id_region++)
	{
	  assert(!is_region_empty(dim, &(regions[2*dim*id_region])));
	  
	  BOUNDS_2_START_SIZES(dim, array_bounds, &(regions[2*dim*id_region]), array_of_starts, array_of_sizes);
	  MPI_Type_create_subarray(dim, array_sizes,
				   array_of_sizes, array_of_starts, step_.language,
				   step_types_table[STEP_type], &(regions_type[id_region])); 
	  array_of_blocklengths[id_region] = 1;
	  array_of_displacements[id_region] = 0;
	}
      MPI_Type_create_struct(nb_region, array_of_blocklengths, array_of_displacements, regions_type,  MPI_type);
      MPI_Type_commit(MPI_type);
    }
  else
    *MPI_type = MPI_DATATYPE_NULL;
}


/******************************************************************************/
/* region[2][nb_dims][nb_workchunk]
   MPI_types_array_SEND[workchunks]
   MPI_types_array_RECV[workchunks]
*/
void
step_region_to_mpi_type (array_identifier_t *array_descriptor,
                         MPI_Datatype *MPI_types_array_SEND,
			 MPI_Datatype *MPI_types_array_RECV)
{
  int id_node;
  int nb_node=step_.size_;
  int local_node = step_.rank_;
  int dims = array_descriptor->dims;
  index_t *s_region = array_descriptor->region_descriptor[0]->regions;
  index_t *r_region = array_descriptor->recv_region.regions;
  int is_interlaced = (array_descriptor->region_descriptor[0]->is_interlaced == 1);

  assert(nb_node == array_descriptor->region_descriptor[0]->nb_region);

  for (id_node=0; id_node<nb_node; id_node++)
    {
      if (id_node == local_node)
	{
	  MPI_types_array_SEND[id_node] = MPI_DATATYPE_NULL;
	  MPI_types_array_RECV[id_node] = MPI_DATATYPE_NULL;
	}
      else
	{
	  index_t *comm_region =(index_t *) malloc(2*dims*16); // 16 ????
	  index_t intersection[2*dims];
	  int nb_regions;
	  
	  /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
	  /*COM[rank-->j] = {SEND[rank] INTER RECV [j]} MINUS (SEND(j))*/
	  //s_region[nb_workchunks][dims][2]
	  intersect_regions(dims, &(s_region[2*dims*local_node]), &(r_region[2*dims*id_node]), intersection);                
	  if (!is_interlaced)
	    diff_region(dims, intersection, &(s_region[2*dims*id_node]), &nb_regions, comm_region);
	  else
	    {
	      nb_regions = 1;
	      comm_region = intersection;
	    }
	  regions_to_MPI(dims, array_descriptor->bounds, nb_regions, comm_region, array_descriptor->array_type, &(MPI_types_array_SEND[id_node]));
	  
	  /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
	  /*COM[j-->rank] = {SEND[j] INTER RECV [rank]} MINUS (SEND(rank))*/
	  intersect_regions(dims, &(s_region[2*dims*id_node]), &(r_region[2*dims*local_node]), intersection);                
	  if (!is_interlaced)
	    diff_region(dims, intersection, &(s_region[2*dims*local_node]), &nb_regions, comm_region);
	  else
	    {
	      nb_regions = 1;
	      comm_region = intersection;
	    }
	  regions_to_MPI(dims, array_descriptor->bounds, nb_regions, comm_region, array_descriptor->array_type, &(MPI_types_array_RECV[id_node]));
	}
    }    
}
