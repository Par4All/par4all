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
  steprt_init(STEP_FORTRAN);
}
void STEP_API(step_init_c_order)(void)
{
  steprt_init(STEP_C);
}
void STEP_API(step_finalize)(void)
{
  steprt_finalize();

  STEP_DEBUG(
	     printf("\n##### steprt_finalized #####\n");
	     )
}
void STEP_API(step_get_commsize)(STEP_ARG *commsize)
{
  *commsize = (STEP_ARG)(communications_get_commsize());
}

void STEP_API(step_get_rank)(STEP_ARG *rank)
{
  *rank = (STEP_ARG)(communications_get_rank());
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
    default: assert(0);
    }
}
void STEP_CONSTRUCT_END(const STEP_ARG construction_)
{
  STEP_ARG construction=construction_;
  STEP_API(step_construct_end)(&construction);
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

void STEP_SET_INTERLACED_SENDREGIONS(void *userArray, STEP_ARG nb_workchunks, STEP_ARG *regions)
{
  steprt_set_sharedTable(userArray, (uint32_t)nb_workchunks, NULL, regions, true);

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
void STEP_SET_REDUCTION_SENDREGIONS(void *userArray, STEP_ARG nb_workchunks, STEP_ARG *regions)
{
  Descriptor_reduction *desc_reduction = steprt_find_reduction(userArray);
  assert(desc_reduction);

  steprt_set_reduction_sendregions(desc_reduction, (uint32_t)nb_workchunks, regions);

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

void STEP_SET_RECVREGIONS(void *userArray, STEP_ARG nb_workchunks, STEP_ARG *regions)
{
  steprt_set_sharedTable(userArray, (uint32_t)nb_workchunks, regions, NULL, false);

  STEP_DEBUG({
      Descriptor_shared *desc = steprt_find_in_sharedTable(userArray);

      assert(desc);
      rg_composedRegion_print(&(desc->receiveRegions));
    })
}

void STEP_API(step_register_alltoall_partial)(void *userArray, STEP_ARG *algorithm, STEP_ARG *tag)
{
  bool full_p = false;
  steprt_register_alltoall(userArray, full_p, (uint32_t)*algorithm, (int_MPI)*tag);
}
void STEP_REGISTER_ALLTOALL_PARTIAL(void *userArray, STEP_ARG algorithm, STEP_ARG tag)
{
  bool full_p = false;
  steprt_register_alltoall(userArray, full_p, (uint32_t)algorithm, (int_MPI)tag);
}

void STEP_API(step_alltoall_full)(void *userArray, STEP_ARG *algorithm, STEP_ARG *tag)
{
  bool full_p = true;
  steprt_alltoall(userArray, full_p, (uint32_t)*algorithm, (int_MPI)*tag);
}

void STEP_API(step_alltoall)(STEP_ARG *algorithm, STEP_ARG *tag)
{
  bool full_p = true;
  steprt_alltoall_all(full_p, (uint32_t)*algorithm, (int_MPI)*tag);
}

void STEP_ALLTOALL(STEP_ARG algorithm, STEP_ARG tag)
{
  bool full_p = true;
  steprt_alltoall_all(full_p, (uint32_t)algorithm, (int_MPI)tag);
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
void STEP_ALLTOALL_FULL_INTERLACED(void *userArray, STEP_ARG algorithm, STEP_ARG tag)
{
  bool full_p = true;
  /* interlaced will be determined with stored values in the runtime */
  steprt_alltoall(userArray, full_p, (uint32_t)algorithm, (int_MPI)tag);
}

void STEP_API(step_alltoall_partial)(void *userArray, STEP_ARG *algorithm, STEP_ARG *tag)
{
  bool full_p = false;
  steprt_alltoall(userArray, full_p, (uint32_t)*algorithm, (int_MPI)*tag);
}
void STEP_ALLTOALL_PARTIAL(void *userArray, STEP_ARG algorithm, STEP_ARG tag)
{
  bool full_p = false;
  steprt_alltoall(userArray, full_p, (uint32_t)algorithm, (int_MPI)tag);
}

void STEP_API(step_alltoall_partial_interlaced)(void *userArray, STEP_ARG *algorithm, STEP_ARG *tag)
{
  bool full_p = false;
  steprt_alltoall(userArray, full_p, (uint32_t)*algorithm, (int_MPI)*tag);
}

void STEP_ALLTOALL_PARTIAL_INTERLACED(void *userArray, STEP_ARG algorithm, STEP_ARG tag)
{
  bool full_p = false;
  steprt_alltoall(userArray, full_p, (uint32_t)algorithm, (int_MPI)tag);
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

/* execute every pending communication and wait until
 * all are done.
 */
void STEP_API(step_flush)(void)
{
  assert(CURRENTWORKSHARING);
  steprt_run_registered_alltoall(CURRENTWORKSHARING);
  communications_waitall(&(CURRENTWORKSHARING->communicationsArray));
}

void STEP_API(step_barrier)(void)
{
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
void STEP_INITREDUCTION (void *variable, STEP_ARG op, STEP_ARG type)
{
  assert(CURRENTWORKSHARING);

  steprt_initreduction(variable, (uint32_t)op, (uint32_t)type);
}

void STEP_API(step_reduction)(void *variable)
{
  steprt_reduction(variable);
}
