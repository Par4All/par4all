#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#define INLINE static inline
#include "step_private.h"
#include "steprt.h"
#include "communications.h"
#include "step_api.h"
#include "step_common.h"
#include "trace.h"
#include <error.h>
#include <omp.h>

/*##############################################################################

  API

  Arguments are passed by address to be compatible with Fortran

##############################################################################*/


void STEP_API(step_init_fortran_order)(void)
{
  //  SET_TRACES("traces", NULL, 1, 0);
  steprt_init(STEP_FORTRAN);
}
void STEP_INIT_FORTRAN_ORDER(void)
{
  //  SET_TRACES("traces", NULL, 1, 0);
  steprt_init(STEP_FORTRAN);
}
/* FSC renommer step_spawn en step_critical_spawn */
/* FSC voir a quoi sert nb_critical. Est-il utilisé? Si oui, quand? */

void STEP_API(step_critical_spawn)()
{
  communications_critical_spawn ();
}


void STEP_API(step_init_c_order)(void)
{
  //  SET_TRACES("traces", NULL, 1, 0);
  steprt_init(STEP_C);
}
void STEP_INIT_C_ORDER(void)
{
  //  SET_TRACES("traces", NULL, 1, 0);
  steprt_init(STEP_C);
}
void STEP_API(step_finalize)(void)
{
  steprt_finalize();

  STEP_DEBUG(
	     printf("\n##### steprt_finalized #####\n");
	     )
}
void STEP_FINALIZE(void)
{
  steprt_finalize();

  STEP_DEBUG(
	     printf("\n##### steprt_finalized #####\n");
	     )
}
void STEP_API(step_get_commsize)(STEP_ARG *commsize)
{
  uint32_t s;
  
  communications_get_commsize(&s);
  *commsize = (STEP_ARG)s;
}

void STEP_API(step_get_rank)(STEP_ARG *rank)
{
  uint32_t r;

  communications_get_rank(&r);
  *rank = (STEP_ARG)r;
}


void STEP_API(step_construct_begin)(const STEP_ARG *construction)
{
  switch (*construction)
    {
    case STEP_PARALLEL:
      steprt_worksharing_set(parallel_work);
      break;
    case STEP_DO:
      steprt_worksharing_set(do_work);
      break;
    case STEP_PARALLEL_DO:
      steprt_worksharing_set(parallel_work);
      steprt_worksharing_set(do_work);
      break;
    case STEP_MASTER:
      steprt_worksharing_set(master_work);
      break;
    case STEP_CRITICAL:
      steprt_worksharing_set(critical_work);
      break;
    default: assert(0);
    }
}
void STEP_CONSTRUCT_BEGIN(const STEP_ARG construction_)
{
  STEP_ARG construction=construction_;
  STEP_API(step_construct_begin)(&construction);	
}

void STEP_API(step_construct_end)(const STEP_ARG *construction)
{
  switch (*construction)
    {
    case STEP_PARALLEL:
      assert(CURRENTWORKSHARING->type == parallel_work);
      steprt_worksharing_unset();
      break;
    case STEP_DO:
      assert(CURRENTWORKSHARING->type == do_work);
      steprt_worksharing_unset();
      break;
    case STEP_PARALLEL_DO:
      assert(CURRENTWORKSHARING->type == do_work);
      steprt_worksharing_unset();
      assert(CURRENTWORKSHARING->type == parallel_work);
      steprt_worksharing_unset();
      break;
    case STEP_MASTER:
      assert(CURRENTWORKSHARING->type == master_work);
      steprt_worksharing_unset();
      break;
     case STEP_CRITICAL:
      assert(CURRENTWORKSHARING->type == critical_work);
      steprt_worksharing_unset();
      break;
    default: assert(0);
    }
}
void STEP_CONSTRUCT_END(const STEP_ARG construction_)
{
  STEP_ARG construction=construction_;
  STEP_API(step_construct_end)(&construction);	
}
void STEP_API(step_critical_request)(const STEP_ARG *num_current_critical)
{
  communications_critical_request(*num_current_critical);
}


void STEP_API(step_critical_get_nextprocess)()
{
  communications_critical_get_nextprocess();
}

void STEP_API(step_critical_release)()
{
  communications_critical_release();
}

void STEP_API(step_init_arrayregions)(void *userArray, STEP_ARG *type, STEP_ARG *nbdims, ...)
{
  int i;
  INDEX_TYPE *bounds;
  va_list args_list;

  va_start (args_list, nbdims);

  bounds = malloc (2*(*nbdims)*sizeof(STEP_ARG));
  if (bounds == NULL)
    {
      perror("Problem when allocating bounds\n");
      exit(EXIT_FAILURE);
    }

  for(i=0; i<2*(*nbdims); i++)
    bounds[i] = *(va_arg(args_list, STEP_ARG *));

  steprt_set_userArrayTable(userArray, (uint32_t)(*type), (uint32_t)(*nbdims), bounds);

  va_end(args_list);
  free(bounds);

}
void STEP_INIT_ARRAYREGIONS(void *userArray, STEP_ARG type, STEP_ARG nbdims, ...)
{
  int i;
  INDEX_TYPE *bounds;
  va_list args_list;

  va_start (args_list, nbdims);

  bounds = malloc (2*(nbdims)*sizeof(STEP_ARG));
  if (bounds == NULL)
    {
      perror("Problem when allocating bounds\n");
      exit(EXIT_FAILURE);
    }

  for(i=0; i<2*(nbdims); i++)
    bounds[i] = va_arg(args_list, STEP_ARG);

  steprt_set_userArrayTable(userArray, (uint32_t)(type), (uint32_t)(nbdims), bounds);

  va_end(args_list);
  free(bounds);

}
void STEP_API(step_compute_loopslices)(STEP_ARG *begin, STEP_ARG *end, STEP_ARG *incr, STEP_ARG *nb_workchunks)
{
  /* Computed workchunks will be stored into the runtime
     in the workchunkArray corresponding to the current worksharing directive
  */

  steprt_compute_workchunks((INDEX_TYPE)*begin, (INDEX_TYPE)*end, (INDEX_TYPE)*incr, (STEP_ARG)*nb_workchunks);
}
void STEP_COMPUTE_LOOPSLICES(STEP_ARG begin, STEP_ARG end, STEP_ARG incr, STEP_ARG nb_workchunks)
{
  /* Computed workchunks will be stored into the runtime
     in the workchunkArray corresponding to the current worksharing directive
  */

  steprt_compute_workchunks((INDEX_TYPE)begin, (INDEX_TYPE)end, (INDEX_TYPE)incr, (STEP_ARG)nb_workchunks);
}
void STEP_API(step_get_loopbounds)(STEP_ARG *id_workchunk, STEP_ARG *begin, STEP_ARG *end)
{
  INDEX_TYPE* bounds = rg_get_simpleRegion(&(CURRENTWORKSHARING->workchunkRegions), *id_workchunk);
  INDEX_TYPE b = bounds[LOW(0)];
  INDEX_TYPE e = bounds[UP(0)];
  *begin = (STEP_ARG)b;
  *end = (STEP_ARG)e;
}
void STEP_GET_LOOPBOUNDS(STEP_ARG id_workchunk, STEP_ARG *begin, STEP_ARG *end)
{
  INDEX_TYPE* bounds = rg_get_simpleRegion(&(CURRENTWORKSHARING->workchunkRegions), id_workchunk);
  INDEX_TYPE b = bounds[LOW(0)];
  INDEX_TYPE e = bounds[UP(0)];
  *begin = (STEP_ARG)b;
  *end = (STEP_ARG)e;
}
void STEP_API(step_set_sendregions)(void *userArray, STEP_ARG *nb_workchunks, STEP_ARG *regions)
{
  steprt_set_sharedTable(userArray, (uint32_t)*nb_workchunks, NULL, regions, false);

  STEP_DEBUG({
      Descriptor_shared *desc = steprt_find_in_sharedTable(userArray);
      
      assert(desc);
      rg_composedRegion_print(&(desc->sendRegions));
    })
}
void STEP_SET_SENDREGIONS(void *userArray, STEP_ARG nb_workchunks, STEP_ARG *regions)
{
  steprt_set_sharedTable(userArray, (uint32_t)nb_workchunks, NULL, regions, false);

  STEP_DEBUG({
      Descriptor_shared *desc = steprt_find_in_sharedTable(userArray);
      
      assert(desc);
      rg_composedRegion_print(&(desc->sendRegions));
    })
}
void STEP_API(step_set_interlaced_sendregions)(void *userArray, STEP_ARG *nb_workchunks, STEP_ARG *regions)
{
  steprt_set_sharedTable(userArray, (uint32_t)*nb_workchunks, NULL, regions, true);

  STEP_DEBUG({
      Descriptor_shared *desc = steprt_find_in_sharedTable(userArray);

      assert(desc);
      rg_composedRegion_print(&(desc->sendRegions));
    })
}
void STEP_API(step_set_reduction_sendregions)(void *userArray, STEP_ARG *nb_workchunks, STEP_ARG *regions)
{
  Descriptor_reduction *desc_reduction = steprt_find_reduction(userArray);
  assert(desc_reduction);

  steprt_set_reduction_sendregions(desc_reduction, (uint32_t)*nb_workchunks, regions);

  STEP_DEBUG({
      Descriptor_shared *desc = steprt_find_in_sharedTable(userArray);

      assert(desc);
      rg_composedRegion_print(&(desc->sendRegions));
    })
}

void STEP_API(step_set_recvregions)(void *userArray, STEP_ARG *nb_workchunks, STEP_ARG *regions)
{
  steprt_set_sharedTable(userArray, (uint32_t)*nb_workchunks, regions, NULL, false);

  STEP_DEBUG({
      Descriptor_shared *desc = steprt_find_in_sharedTable(userArray);

      assert(desc);
      rg_composedRegion_print(&(desc->receiveRegions));
    })
}

void STEP_API(step_alltoall_full)(void *userArray, STEP_ARG *algorithm, STEP_ARG *tag)
{
  bool full_p = true;
  steprt_alltoall(userArray, full_p, (uint32_t)*algorithm, (int_MPI)*tag);
}
void STEP_ALLTOALL_FULL(void *userArray, STEP_ARG algorithm, STEP_ARG tag)
{
  bool full_p = true;
  steprt_alltoall(userArray, full_p, (uint32_t)algorithm, (int_MPI)tag);
}
void STEP_API(step_alltoall_full_interlaced)(void *userArray, STEP_ARG *algorithm, STEP_ARG *tag)
{
  bool full_p = true;
  /* interlaced will be determined with stored values in the runtime */
  steprt_alltoall(userArray, full_p, (uint32_t)*algorithm, (int_MPI)*tag);
}
void STEP_API(step_alltoall_partial)(void *userArray, STEP_ARG *algorithm, STEP_ARG *tag)
{
  bool full_p = false;
  steprt_alltoall(userArray, full_p, (uint32_t)*algorithm, (int_MPI)*tag);
}
void STEP_API(step_alltoall_partial_interlaced)(void *userArray, STEP_ARG *algorithm, STEP_ARG *tag)
{
  bool full_p = false;
  steprt_alltoall(userArray, full_p, (uint32_t)*algorithm, (int_MPI)*tag);
}


void STEP_API(step_mastertoallscalar)(void *scalar, STEP_ARG *algorithm, STEP_ARG *type)
{
  communications_oneToAll_Scalar(scalar, (uint32_t) *type, (uint32_t) *algorithm);
}
void STEP_API(step_mastertoallregion)(void *userArray, STEP_ARG *algorithm)
{
  STEP_DEBUG(
	     printf("\ncommunications_masterToAllRegion begin userArray=%p\n", userArray);
	     )
  Descriptor_userArray *desc_userArray = steprt_find_in_userArrayTable(userArray);

  communications_oneToAll_Array(desc_userArray, (uint32_t) *algorithm);

  STEP_DEBUG(
	     printf("\ncommunications_masterToAllRegion end userArray=%p\n", userArray);
	     )
}
void STEP_API(step_alltomasterregion)(void *userArray, STEP_ARG *algorithm)
{
  STEP_DEBUG(
	     printf("\ncommunications_AllToMasterRegion begin userArray=%p\n", userArray);
	     )    
  bool full_p = false;
  steprt_alltoall(userArray, full_p, (uint32_t)*algorithm, STEP_TAG_DEFAULT);

  STEP_DEBUG(
	     printf("\ncommunications_AllToMasterRegion end userArray=%p\n", userArray);
	     )
}

void STEP_API(step_waitall)(void)
{
  assert(CURRENTWORKSHARING);
  communications_waitall(CURRENTWORKSHARING);
}
void STEP_API(step_barrier)(void)
{
  assert(IS_INITIALIZED);
  STEP_DEBUG(
	     printf("step_barrier\n");
	     )
  communications_barrier();
  STEP_DEBUG(
	     printf("\n");
	     )
}

void STEP_API(step_initreduction)(void *variable, STEP_ARG *op, STEP_ARG *type)
{
  assert(CURRENTWORKSHARING);

  steprt_initreduction(variable, (uint32_t)(*op), (uint32_t)(*type));
}

void STEP_API(step_reduction)(void *variable)
{
  steprt_reduction(variable);
}

/* ---------------------------
CRITICAL
------------------------------ */

/* if critical is in a loop DO , we insert that function to check if ( max_threads mod nb_iterations of a thread == 0 )to avoid  having a blockage of the program's execution*/ 
void STEP_API(step_critical_check_n_threads) (STEP_ARG * STEP_I_UP, STEP_ARG * STEP_I_LOW)
{
  if ( (*STEP_I_UP - *STEP_I_LOW + 1) % omp_get_max_threads() != 0)  
    {   
      error (EXIT_FAILURE, 0, "omp_get_max_threads() ne divise pas le nombre d'itérations associé : exit...%d",MYRANK);
      //assert(MPI_Abort(MPI_COMM_WORLD,10) == MPI_SUCCESS); //ça ne marche pas correctement
    }
}

/* FSC supprimer l'algorithme des parametres du code genere */
void STEP_API(step_critical_set_currentuptodatescalar)(void *scalar, STEP_ARG *type)
{
  communications_set_currentuptodate_scalar(scalar, (uint32_t) *type);
}
void STEP_API(step_critical_set_currentuptodateregion)(void *userArray)
{
  STEP_DEBUG(
	     printf("\ncommunications_set_currentuptodate_array begin userArray=%p\n", userArray);
	     )
  Descriptor_userArray *desc_array = steprt_find_in_userArrayTable(userArray);

  communications_set_currentuptodate_array(desc_array);

  STEP_DEBUG(
	     printf("\ncommunications_set_currentuptodate_array end userArray=%p\n", userArray);
	     )
}

void STEP_API(step_critical_get_currentuptodatescalar)(void *scalar, STEP_ARG *type)
{
  communications_get_currentuptodate_scalar(scalar, (uint32_t) *type);
}
void STEP_API(step_critical_get_currentuptodateregion)(void *userArray)
{
 STEP_DEBUG(
	     printf("\ncommunications_criticalfromOneRegion begin userArray=%p\n", userArray);
	     )
  Descriptor_userArray *desc_array = steprt_find_in_userArrayTable(userArray);

  communications_get_currentuptodate_array(desc_array);

  STEP_DEBUG(
	     printf("\ncommunications_criticalFromOneRegion end userArray=%p\n", userArray);
	     )
}

/* FSC renommer en supprimant le get 
   FSC supprimer l'algorithme du code généré */

void STEP_API(step_critical_finaluptodatescalar)(void *scalar, STEP_ARG *type)
{
  communications_finaluptodate_scalar(scalar, (uint32_t) *type);
}
void STEP_API(step_critical_finaluptodateregion)(void *userArray)
{
 STEP_DEBUG(
	     printf("\ncommunications_criticalcriticallastcommunicationRegion begin userArray=%p\n", userArray);
	     )
  Descriptor_userArray *desc_array = steprt_find_in_userArrayTable(userArray);

  communications_finaluptodate_array(desc_array);

  STEP_DEBUG(
	     printf("\ncommunications_criticalcriticallastcommunicationRegion end userArray=%p\n", userArray);
	     )
}
//*******************END critical*****************************/


#ifdef TEST_STEP_API
#include <stdlib.h>
#include <stdio.h>
#include "omp.h"

void test_constructs()
{
  int construction;

  util_print_rank(0, "Beginning a DO construct\n");
  construction = STEP_DO;
  step_construct_begin_(&construction);
  if (MYRANK == 0)
    steprt_print();
  util_print_rank(0, "Ending a DO construct\n");
  step_construct_end_(&construction);
  if (MYRANK == 0)
    steprt_print();
  util_print_rank(0, "\n");

  util_print_rank(0, "Beginning a DO construct\n");
  step_construct_begin_(&construction);
  if (MYRANK == 0)
    steprt_print();
  util_print_rank(0, "Beginning a DO construct\n");
  step_construct_begin_(&construction);
  if (MYRANK == 0)
    steprt_print();
  util_print_rank(0, "Ending a DO construct\n");
  step_construct_end_(&construction);
  if (MYRANK == 0)
    steprt_print();
  util_print_rank(0, "Ending a DO construct\n");
  step_construct_end_(&construction);
  if (MYRANK == 0)
    steprt_print();
  util_print_rank(0, "\n");


  util_print_rank(0, "FSC: Pourquoi PARALLEL_LEVEL n'est-il jamais modifié?\n");
}

#define N 10
#define MAX_NB_LOOPSLICES 16
#define LOW1 0
#define UP1 N
#define LOW2 0
#define UP2 4


void test_step_api_1D(int commsize, int rank)
{
  STEP_ARG low1, up1;
  STEP_ARG my_ilow, my_iup;
  int i, p;
  STEP_ARG nbdims, type;
  int incr;
  int construction;
  /* userArray A*/
  int A[N];
  /* static declaration to be compatible with Fortran 77 */
  /* WARNING: FORTRAN and C order of dimensions are different */
  STEP_ARG send_regions_A[MAX_NB_LOOPSLICES][1][2];
  int algorithm = STEP_NBLOCKING_ALG;
  int tag = STEP_TAG_DEFAULT;

#pragma omp master
  {
    util_print_rank(0, "Process#%d Testing 1D STEP API\n", rank);
    util_print_rank(0, "Process#%d -------------------\n", rank);
    util_print_rank(0, "Process#%d 1) Beginning a PARALLEL construct\n", rank);
    construction = STEP_PARALLEL;
    step_construct_begin_(&construction);
    
    util_print_rank(0, "Process#%d 2) Creating userArray A\n", rank);
    
    low1 = LOW1;
    /* C bounds */
    up1 = UP1 - 1;
    util_print_rank(0, "Process#%d 2.1) Initializing array table with A and bounds array, type == STEP_INTEGER, nbdims == 1\n", rank);
    type = STEP_INTEGER;
    nbdims = 1;
    step_init_arrayregions_((void*)A, &type, &nbdims, &low1, &up1);
    
    util_print_rank(0, "Process#%d 3) Beginning a DO construct\n", rank);
    construction = STEP_DO;
    step_construct_begin_(&construction);
    

    incr = 1;
    util_print_rank(0, "Process#%d 4) Computing loop slices for the DO loop from %d to %d (incr = %d) with %d processes\n", rank, low1, up1, incr, commsize);
    step_compute_loopslices_(&low1, &up1, &incr, &commsize);
    
    for (p = 0; p < commsize; p++)
      {
	int i_low, i_up;
	
	step_get_loopbounds_(&p, &i_low, &i_up);
	util_print_rank(0, "Process#%d loop bounds low=%d up=%d\n", p, i_low, i_up);
      }
    
    util_print_rank(0, "Process#%d 5) Creating SEND regions for the DO loop for array A corresponding to loopslices\n", rank);
    for (p = 0; p < commsize; p++)
      {
	int i_low, i_up;
	
	step_get_loopbounds_(&p, &i_low, &i_up);
	
	/* WARNING: FORTRAN and C order of dimensions are different */
	/* 0: low index, 0: first (and only) dimension, p: process rank */
	send_regions_A[p][0][0]= i_low;
	/* 1: low index, 0: first (and only) dimension, p: process rank */
	send_regions_A[p][0][1]= i_up;
	
	util_print_rank(0, "\tProcess#%d: send_regions_A[%d][0][%d] = %d, send_regions_A[%d][0][%d] = %d\n", p, p, 0, send_regions_A[p][0][0], p, 1, send_regions_A[p][0][1]);
      }
    
    /* cast otherwise warning */
    step_set_sendregions_(A, &commsize, (int32_t *)send_regions_A);
    
    util_print_rank(0, "Process#%d 6) Where the work is done\n", rank);
    step_get_loopbounds_(&rank, &my_ilow, &my_iup);

    util_print_rank(0, "Process#%d Computing A from %d to %d: ", rank, my_ilow, my_iup);
  }

    /* WARNING <= for upper loop bound !!!!! */
#pragma omp for
  for (i = my_ilow; i <= my_iup; i++)
    {
      A[i] = rank;
      util_print_rank(0, "A[%d]=%d ", i, A[i]);
    }
  util_print_rank(0, "\n");

#pragma omp master
  {
    step_alltoall_full_(A, &algorithm, &tag);
    step_waitall_();
    
    util_print_rank(0, "Process#%d 7) Ending a DO construct\n", rank);
    construction = STEP_DO;
    step_construct_end_(&construction);
    
    util_print_rank(0, "Process#%d 8) Ending a PARALLEL construct\n", rank);
    construction = STEP_PARALLEL;
    step_construct_end_(&construction);
    
    /* to print correctly */
    step_barrier_();
    for (i=low1; i<up1; i++)
      {
	util_print_rank(0, "Process#%d A[%d]=%d\n", rank, i, A[i]);
      }
  }
}

void test_step_api_2D(int commsize, int rank)
{
  STEP_ARG low1, up1, low2, up2;
  STEP_ARG my_ilow, my_iup;
  int i, j, p;
  STEP_ARG nbdims, type;
  int incr;
  int construction;
  /* userArray B */
  int B[UP1][UP2];
  /* static declaration to be compatible with Fortran 77 */
  /* WARNING: FORTRAN and C order of dimensions are different */
  STEP_ARG send_regions_B[MAX_NB_LOOPSLICES][2][2];
  int algorithm = STEP_NBLOCKING_ALG;
  int tag = STEP_TAG_DEFAULT;
  
#pragma omp master
  {
    low1 = LOW1;
    up1 = UP1 - 1;
    low2 = LOW2;
    up2 = UP2 - 1;
    
    util_print_rank(0, "Process#%d Testing 2D STEP API\n", rank);
    util_print_rank(0, "Process#%d -------------------\n", rank);
    
    /* initialisation de B */
    for (i = low1; i <= up1; i++)
      {
	for (j = low2; j <= up2; j++)
	  {
	    B[i][j] = -1;
	  }
      }
    
    util_print_rank(0, "Process#%d 1) Beginning a PARALLEL construct\n", rank);
    construction = STEP_PARALLEL;
    step_construct_begin_(&construction);
    
    util_print_rank(0, "Process#%d 2) Creating userArray B\n", rank);
    
    util_print_rank(0, "Process#%d 2.1) Initializing array table with B and bounds array, type == STEP_INTEGER, nbdims == 2\n", rank);
    type = STEP_INTEGER;
    nbdims = 2;
    step_init_arrayregions_((void*)B, &type, &nbdims, &low1, &up1, &low2, &up2);
    
    util_print_rank(0, "Process#%d 3) Beginning a DO construct\n", rank);
    construction = STEP_DO;
    step_construct_begin_(&construction);
    
    
    incr = 1;
    util_print_rank(0, "Process#%d 4) Computing loop slices for the DO loop from %d to %d (incr = %d) with %d processes\n", rank, LOW1, UP1, incr, commsize);
    step_compute_loopslices_(&low1, &up1, &incr, &commsize);
    
    for (p = 0; p < commsize; p++)
      {
	int i_low, i_up;
	
	step_get_loopbounds_(&p, &i_low, &i_up);
	util_print_rank(0, "Process#%d loop bounds low=%d up=%d\n", p, i_low, i_up);
      }
    
    util_print_rank(0, "Process#%d 5) Creating SEND regions for the DO loop for array A corresponding to loopslices\n", rank);
    for (p = 0; p < commsize; p++)
      {
	int i_low, i_up;
	
	step_get_loopbounds_(&p, &i_low, &i_up);
	
	/* WARNING: FORTRAN and C order of dimensions are different */
	/* 0: low index, 0: first (and only) dimension, p: process rank */
	send_regions_B[p][0][0]= i_low;
	/* 1: low index, 0: first (and only) dimension, p: process rank */
	send_regions_B[p][0][1]= i_up;
	
	send_regions_B[p][1][0]= low2;
	send_regions_B[p][1][1]= up2;
	util_print_rank(0, "\tp = %d: send_regions_B[%d][0][%d] = %d, send_regions_B[%d][0][%d] = %d\n", p, p, 0, send_regions_B[p][0][0], p, 1, send_regions_B[p][0][1]);
	util_print_rank(0, "\tp = %d: send_regions_B[%d][1][%d] = %d, send_regions_B[%d][1][%d] = %d\n", p, p, 0, send_regions_B[p][1][0], p, 1, send_regions_B[p][1][1]);
      }
    /* cast otherwise warning */
    step_set_sendregions_(B, &commsize, (int32_t *)send_regions_B);
    
    util_print_rank(0, "Process#%d 6) Where the work is done\n", rank);
    step_get_loopbounds_(&rank, &my_ilow, &my_iup);
    
    util_print_rank(0, "Process#%d Computing B from %d to %d: \n", rank, my_ilow, my_iup);
  }

    /* WARNING <= for upper loop bound !!!!! */
#pragma omp for
  for (i = my_ilow; i <= my_iup; i++)
    {
      printf("Process#%d ", rank);
      for (j = low2; j <= up2; j++)
	{
	  B[i][j] = rank;
	  printf("B[%d][%d]=%d ", i, j, B[i][j]);
	}
      printf("\n");
    }
  printf("\n");

#pragma omp master
  {
    step_alltoall_full_(B, &algorithm, &tag);
    step_waitall_();
    
    util_print_rank(0, "Process#%d 7) Ending a DO construct\n", rank);
    construction = STEP_DO;
    step_construct_end_(&construction);
    
    util_print_rank(0, "Process#%d 8) Ending a PARALLEL construct\n", rank);
    construction = STEP_PARALLEL;
    step_construct_end_(&construction);
    
    /* to print correctly */
    step_barrier_();
    for (i=low1; i<=up1; i++)
      {
	util_print_rank(0, "Process#%d ", rank);
	for (j=low2; j<=up2; j++)
	  {
	    util_print_rank(0, "B[%d][%d] = %d ", i, j, B[i][j]);
	  }
	util_print_rank(0, "\n");
      }
  }
}

void test_step_api_2D_interlaced(int commsize, int rank)
{
  STEP_ARG low1, up1, low2, up2;
  STEP_ARG my_ilow, my_iup;
  int i, j, p;
  STEP_ARG nbdims, type;
  int incr;
  int construction;
  /* userArray C */
  int C[UP1][UP2];
  /* static declaration to be compatible with Fortran 77 */
  /* WARNING: FORTRAN and C order of dimensions are different */
  STEP_ARG send_regions_C[MAX_NB_LOOPSLICES][2][2];
  int algorithm = STEP_NBLOCKING_ALG;
  int tag = STEP_TAG_DEFAULT;
  
#pragma omp master
  {
    low1 = LOW1;
    up1 = UP1 - 1;
    low2 = LOW2;
    up2 = UP2 - 1;
    
    util_print_rank(0, "Process#%d Testing 2D interlaced STEP API\n", rank);
    util_print_rank(0, "Process#%d ------------------------------\n", rank);
    
    /* initialisation de C */
    for (i = low1; i <= up1; i++)
      {
	for (j = low2; j <= up2; j++)
	  {
	    C[i][j] = -1;
	  }
      }
    
    util_print_rank(0, "Process#%d 1) Beginning a PARALLEL construct\n", rank);
    construction = STEP_PARALLEL;
    step_construct_begin_(&construction);
    
    util_print_rank(0, "Process#%d 2) Creating userArray C\n", rank);
    
    util_print_rank(0, "Process#%d 2.1) Initializing array table with C and bounds array, type == STEP_INTEGER, nbdims == 2\n", rank);
    type = STEP_INTEGER;
    nbdims = 2;
    step_init_arrayregions_((void*)C, &type, &nbdims, &low1, &up1, &low2, &up2);
    
    util_print_rank(0, "Process#%d 3) Beginning a DO construct\n", rank);
    construction = STEP_DO;
    step_construct_begin_(&construction);
    
    incr = 1;
    util_print_rank(0, "Process#%d 4) Computing loop slices for the DO loop from %d to %d (incr = %d) with %d processes\n", rank, LOW1, UP1, incr, commsize);
    step_compute_loopslices_(&low1, &up1, &incr, &commsize);
    
    for (p = 0; p < commsize; p++)
      {
	int i_low, i_up;
	
	step_get_loopbounds_(&p, &i_low, &i_up);
	util_print_rank(0, "Process#%d loop bounds low=%d up=%d\n", p, i_low, i_up);
      }
    
    util_print_rank(0, "Process#%d 5) Creating SEND regions for the DO loop for array A corresponding to loopslices\n", rank);
    for (p = 0; p < commsize; p++)
      {
	int i_low, i_up;
	
	step_get_loopbounds_(&p, &i_low, &i_up);
	
	/* WARNING: FORTRAN and C order of dimensions are different */
	/* 0: low index, 0: first (and only) dimension, p: process rank */
	send_regions_C[p][0][0]= low1;
	/* 1: low index, 0: first (and only) dimension, p: process rank */
	send_regions_C[p][0][1]= up1;
	
	send_regions_C[p][1][0]= low2;
	send_regions_C[p][1][1]= up2;
	util_print_rank(0, "\tp = %d: send_regions_C[%d][0][%d] = %d, send_regions_C[%d][0][%d] = %d\n", p, p, 0, send_regions_C[p][0][0], p, 1, send_regions_C[p][0][1]);
	util_print_rank(0, "\tp = %d: send_regions_C[%d][1][%d] = %d, send_regions_C[%d][1][%d] = %d\n", p, p, 0, send_regions_C[p][1][0], p, 1, send_regions_C[p][1][1]);
      }
    /* cast otherwise warning */
    step_set_interlaced_sendregions_(C, &commsize, (int32_t *)send_regions_C);
    
    util_print_rank(0, "Process#%d 6) Where the work is done\n", rank);
    step_get_loopbounds_(&rank, &my_ilow, &my_iup);
    

  }

    printf("Process#%d Computing C from %d to %d: \n", rank, my_ilow, my_iup);
    /* WARNING <= for upper loop bound !!!!! */
#pragma omp for
  for (i = low1; i <= up1; i++)
    {
      printf("Process#%d ", rank);
      for (j = low2; j <= up2; j++)
	{
	  if (j == rank)
	    {
	      C[i][j] = rank;
	      printf("C[%d][%d]=%d ", i, j, C[i][j]);
	    }
	  
	}
      printf("\n");
    }
  printf("\n");
#pragma omp master
  {
    step_alltoall_full_interlaced_(C, &algorithm, &tag);
    step_waitall_();
    
    util_print_rank(0, "Process#%d 7) Ending a DO construct\n", rank);
    construction = STEP_DO;
    step_construct_end_(&construction);
    
    util_print_rank(0, "Process#%d 8) Ending a PARALLEL construct\n", rank);
    construction = STEP_PARALLEL;
    step_construct_end_(&construction);
    
    /* to print correctly */
    step_barrier_();
    for (i=low1; i<=up1; i++)
      {
	util_print_rank(0, "Process#%d ", rank);
	for (j=low2; j<=up2; j++)
	  {
	    util_print_rank(0, "C[%d][%d] = %d ", i, j, C[i][j]);
	  }
	util_print_rank(0, "\n");
      }
  }
}

/*
#define TEST_STEP_API_CONSTRUCTS
#define TEST_STEP_API_1D_EXECUTION 
#define TEST_STEP_API_2D_EXECUTION 
*/

int main(int argc, char **argv)
{
  int commsize, rank;

  /*  step_init_fortran_order_(); */
  step_init_c_order_();
  step_get_commsize_(&commsize);
  step_get_rank_(&rank);
  util_print_rank(0, "Process#%d commsize = %d, rank = %d\n", rank, commsize, rank);

  SET_TRACES("traces", NULL, commsize, rank);

#ifdef TEST_STEP_API_CONSTRUCTS
  util_print_rank(0, "Part I: testing begin and end constructs\n");
  test_constructs();
#endif
#ifdef TEST_STEP_API_1D_EXECUTION
  util_print_rank(0, "\n\n\n");
  test_step_api_1D(commsize, rank);
#endif
#ifdef TEST_STEP_API_2D_EXECUTION
  util_print_rank(0, "\n\n\n");
  test_step_api_2D(commsize, rank);
#endif
  util_print_rank(0, "\n\n\n");
  test_step_api_2D_interlaced(commsize, rank);

  util_print_rank(0, "\nProcess#%d Finalize\n", rank);
  step_finalize_();


   return EXIT_SUCCESS;
}


#ifdef STEP_API_CRITICAL
int main(int argc, char **argv)
{
  int commsize, rank;
  int num_critical = 0;
  int scalar = 0;
  int type = STEP_INTEGER;
  /*  step_init_fortran_order_(); */
  step_init_c_order_();
  step_critical_spawn_();

  step_get_commsize_(&commsize);
  step_get_rank_(&rank);

  SET_TRACES("traces", NULL, commsize, rank);

  construction = STEP_CRITICAL;
  step_construct_begin_(&construction);
  step_critical_request_(&num_critical);
  step_critical_get_currentuptodatescalar_(&scalar, &type);
  
  scalar ++;
  
  step_critical_get_nextprocess_();
  step_critical_set_currentuptodatescalar_(&scalar, &type);
  step_waitall_();

  step_critical_release_();
  step_construct_end_(&construction);

  step_barrier_();
  step_critical_get_finaluptodatescalar)(&scalar, &type);
  
  util_print_rank(0, "\nProcess#%d Finalize\n", rank);
  step_finalize_();


   return EXIT_SUCCESS;
}

#endif
#endif


