/**
*                                                                             
*   \file             steprt_private.h
*   \author           Abdellah Kouadri.
*                     Daniel Millot.
*                     Frédérique Silber-Chaussumier.
*                     Alain Muller
*   \date             04/02/2010
*   \version          2.0
*   \brief            Interface file for steprt_private.c
*                     
*/

#ifndef STEP_PRIVATE_H_
#define STEP_PRIVATE_H_

#include "steprt_common.h"
#include "mpi.h"


#define true 1
#define false 0

#define MAX(X,Y) ((X)>(Y)) ? X:Y
#define MIN(X,Y) ((X)<(Y)) ? X:Y

#define LOW(i) (2*(i))
#define UP(i) (2*(i)+1)
#define BOUNDS_2_START_SIZES(dim, bounds_array, bounds_sub_array, start_sub_array, sizes_sub_array) \
  {int d_; for (d_=0; d_<dim; d_++){					\
      (start_sub_array)[d_] = (bounds_sub_array)[LOW(d_)] - (bounds_array)[LOW(d_)]; \
      (sizes_sub_array)[d_] = 1 + (bounds_sub_array)[UP(d_)] - (bounds_sub_array)[LOW(d_)];}}

/**
* \struct step_internals
* \brief Encapsulates usefull variables 
*/
struct step_internals
{
  int size_;			/*!< The number of processes */
  int rank_;			/*!< The rank of current process */
  int language;			/*!< The calling language (C or Fortran) */
  int parallel_level;		/*!< The current nesting parallel level */
  int intialized;		/*!< Did the current process called step_init */
};


/**
* \struct complexe8_
* \brief Fortran complex8 data type
*/

typedef struct complexe8_
{
  float rp;			/*!< The real part */
  float ip;			/*!< The imaginary part */
} complexe8_t;
/**
* \struct complexe8_
* \brief Fortran complex16 data type
*/
typedef struct complexe16_
{
  double rp;			/*!< The real part */
  double ip;			/*!< The imaginary part */
} complexe16_t;





#if defined(__arch64__)
#define  int 
#else
#define index_t   int 
#endif




typedef struct 
{
  int nb_region;
  index_t *regions;
  int          is_interlaced;
}region_descriptor_t;

/**
* \struct array_identifier_t
* \brief Send/Recv region  hash table entry 
*/
typedef struct array_identifier_
{
  int                   used;
  void*                 array_id;
  index_t               array_size; 
  STEP_Datatype         array_type;
  void*                 saved_data; 
  int                   dims;
  int                   nb_region_descriptor;
  int                   nb_non_updated_regions;
  index_t*              bounds;
  region_descriptor_t*  region_descriptor[MAX_REGIONS];
  region_descriptor_t*  updated_non_communicated_regions[MAX_PROCESSES][MAX_PROCESSES];    
  region_descriptor_t   recv_region;
}array_identifier_t;



/**
* \struct hash_table_t
* \brief  Hash table definition
*/
struct 
{
  int                     nb_entries;
  array_identifier_t      arrays_table[MAX_NB_ARRAYS];
}step_hash_table;


// array_regions.c
int complement_region(int dims,index_t *R1,index_t *R2, int nb_sub_regions,int *nb_results, index_t *Result);
int is_region_empty(int dims,index_t *R);
void display_region(int dims,index_t *region, char *reg_name);
int intersect_regions(int dims, index_t *R1, index_t *R2, index_t *Result);
int diff_region(int dims, index_t *R1, index_t *R2,int *nb_results, index_t *Result);

// steprt_private.c
extern MPI_Datatype step_types_table[STEP_MAX_TYPES];
extern struct step_internals step_;
void step_init_hash_table();
int step_hash_table_find_entry(void *id);
void step_hash_table_insert_array(void *id, int dims, index_t *bounds, STEP_Datatype type);
void step_hash_table_add_region(void *id,index_t *region_desc, int nb_workchunks, int is_interlaced);
void step_hash_table_delete_partial_regions(void *id);
void step_hash_table_delete_send_region(void *id);
void step_hash_table_delete_recv_region(void *id);
void step_hash_table_delete_array(void *id);
void compute_non_sent_regions(array_identifier_t *array_descriptor);
void step_region_to_mpi_type (array_identifier_t *array_descriptor, MPI_Datatype *MPI_types_array_SEND, MPI_Datatype *MPI_types_array_RECV);

void
step_diff (void *buffer_1, void *buffer_2, void *initial, STEP_Datatype type,
	   int dims, int dim_sizes[MAX_DIMS],
	   int dim_starts_1[MAX_DIMS], int dim_starts_2[MAX_DIMS],
	   int dim_ss_1[MAX_DIMS], int dim_ss_2[MAX_DIMS], int row_col);

#endif
