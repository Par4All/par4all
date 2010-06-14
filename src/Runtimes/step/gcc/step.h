/********************************************************************************
 *   Description	: 	STEP API                                        *
 *   Authors 		:  	Abdellah. Kouadri                               *
 *				Daniel Millot                                   *	
 *                              Frédérique Silber-Chaussumier                   *
 *				                                                *
 *   Date		:       18/03/2009                                      *
 *						                                *
 *   File		:	step.h		                                *
 *									       	*
 *   Version		:     	1.0					        *
 *        									*
 ********************************************************************************/

#ifndef STEP_H_
#define STEP_H_
#include "mpi.h"
#include <stdio.h>
#define COMM		MPI_COMM_WORLD
#define MAX_SIZE	2048*2048
#define MAX_ARRAYS	8


#if defined(__arch64__)
  #define STEP_BYTE           unsigned char
  #define STEP_WORD           unsigned short
  #define STEP_DWORD          unsigned int
  #define STEP_QWORD          long unsigned int
#else
  #define STEP_BYTE           unsigned char
  #define STEP_WORD           unsigned short
  #define STEP_DWORD          unsigned int
  #define STEP_QWORD          long long unsigned int
#endif


#define STEP_Op				MPI_Op
#define STEP_SUM			MPI_SUM
#define STEP_MAX_			MPI_MAX 
#define STEP_MIN_			MPI_MIN 
#define STEP_PROD			MPI_PROD
#define STEP_LAND			MPI_LAND 
#define STEP_BAND			MPI_BAND 
#define STEP_LOR			MPI_LOR 
#define STEP_BOR			MPI_BOR 
#define STEP_LXOR			MPI_LXOR 
#define STEP_BXOR			MPI_BXOR
#define STEP_MAXLOC			MPI_MAXLOC
#define STEP_MINLOC			MPI_MINLOC


/*Important : Do Not modify (GCC assumes those values)!!!!*/
#define MAX_DIMS	16
#define MAX_THREADS	16
#define MAX_SHARED	16

typedef enum  {STEP_ALG_ALLTOALL,STEP_ALG_CENTRALIZED} STEP_alg_type;
typedef enum  {LOOP_DIRECTION_UP,LOOP_DIRECTION_DOWN}  STEP_loop_direction;
typedef struct Step_Array_Desc
{
	void 			* array;
	int 			array_size;
	int 			base_type_size;
}Step_Array_Stack_t;


typedef struct Step_Scalar_Desc
{

	void 			*data;
	int 			 size;
	int 			 sign;

}Step_Scalar_Desc_t;


/* Initialization and Finalization of  communications */
int  STEP_Init();
int  STEP_Finalize();

/* Process identification */
int  STEP_Get_Comm_rank(int *rank);
int  STEP_Get_Comm_size(int *size);

/* Worksharing */
int STEP_Set_LoopBounds(int id, int initial_i, int maxium_i, int incr,
			STEP_loop_direction direction, int *start, int *end,
			int bounds[MAX_THREADS][2]);

/* Data update with Array region analysis */
int  STEP_AlltoAll_Region(void* data_array,int rank, STEP_alg_type algorithm,
				 int base_type_size, int dims_count,
				 int dim_sizes[MAX_DIMS], 
				 int dim_sub_sizes[MAX_THREADS][MAX_DIMS],
				 int dim_starts[MAX_THREADS][MAX_DIMS]);


/* Reduction handling */
int STEP_Share(void *data, int type_size, int sign);
int STEP_Reduction(int id,  STEP_Op op);
int STEP_UnShare_All();



/* Make a copy of an array*/
void* STEP_Push_array(void * array, int array_size, int base_type_size);
void* STEP_Pop_array();  
void STEP_Print_integer(int val);
/* Region's arithmetic*/
int STEP_MIN (int v1, int v2);
int STEP_MAX (int v1, int v2);

#endif


