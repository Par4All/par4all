#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#define INLINE static inline
#include "step_private.h"
#include "communications.h"
#include "regions.h"
#include "trace.h"

/*##############################################################################

  Step_internals

##############################################################################*/
Step_internals steprt_params = {0, 0, 0, 0, 0, NULL};
Array steprt_userArrayTable;
Array steprt_worksharingTable;

static void uptodate_regions_reset_for_communications(Descriptor_userArray *desc_array);
static void steprt_print_worksharing(Descriptor_worksharing *desc_worksharing);

void util_print_rank(int rank, char *input_format, ...)
{
  va_list args;

  va_start (args, input_format);

  if (MYRANK == rank)
  {
    vprintf(input_format, args);
  }

  fflush(stdout);
  va_end (args);
}

void __attribute__((unused)) steprt_print(void)
{
  int i;
  static int id;
  printf("Print#%d : NB_NODES %u rank %u language %u parallel_level %u initialized %u\n", id++, NB_NODES, steprt_params.rank, steprt_params.language, steprt_params.parallel_level, steprt_params.initialized);
  printf("\tSize steprt_userArrayTable = %u\n", (uint32_t)steprt_userArrayTable.len);
  printf("\tSize steprt_worksharingTable = %u\n", (uint32_t)steprt_worksharingTable.len);
  printf("\tCURRENTWORKSHARING=%p\n", CURRENTWORKSHARING);
  for (i = 0; i < steprt_worksharingTable.len; i++)
    {
      Descriptor_worksharing *desc;
      desc = &(array_get_data_from_index(&steprt_worksharingTable, Descriptor_worksharing, i));
      printf("\tWorksharing#%d: ", i);
      steprt_print_worksharing(desc);
    }
}

void steprt_init(int language)
{

  IN_TRACE("language = %d", language);

  communications_init();

  if(!IS_INITIALIZED)
    {
      communications_get_commsize(&steprt_params.commsize);
      communications_get_rank(&steprt_params.rank);
      steprt_params.language = language;
      steprt_params.parallel_level = 0;
      steprt_params.initialized = 1;
      steprt_params.current_worksharing = NULL;
      array_set(&steprt_userArrayTable, Descriptor_userArray);
      array_set(&steprt_worksharingTable, Descriptor_worksharing);
    }

  OUT_TRACE("end");
  return;
}

/*##############################################################################

  Descriptor_userArray

##############################################################################*/
static Descriptor_userArray *steprt_set_userArrayDescriptor(Descriptor_userArray *desc_userArray, void *userArray, uint32_t type, uint nbdims, INDEX_TYPE *bounds)
{
  Array *uptodateArray;
  Array *interlacedArray;
  composedRegion *userArrayBounds;

  IN_TRACE("desc_userArray = %p, userArray = %p, type = %d, nbdims = %d, bounds = %p", desc_userArray, userArray, type, nbdims, bounds);
  assert(desc_userArray && userArray && bounds);

  uptodateArray = &(desc_userArray->uptodateArray);
  interlacedArray = &(desc_userArray->interlacedArray);
  userArrayBounds = &desc_userArray->boundsRegions;

  desc_userArray->userArray = userArray;
  desc_userArray->savedUserArray = NULL;
  desc_userArray->type = type;
  rg_composedRegion_set(userArrayBounds, nbdims);
  rg_composedRegion_reset(userArrayBounds, bounds, 1);
  array_set(uptodateArray, composedRegion);
  array_set(interlacedArray, composedRegion);

  /*
    UPTODATE(id_node) = ALL_ARRAY
    INTERLACED(id_node) = EMPTY
  */
  uptodate_regions_reset_for_communications(desc_userArray);

  OUT_TRACE("desc_userArray = %p", desc_userArray);
  return desc_userArray;
}

Descriptor_userArray *steprt_find_in_userArrayTable(void *userArray)
{
  size_t i;
  Descriptor_userArray *desc_userArray;
  bool stop = false;

  IN_TRACE("userArray = %p", userArray);
  assert(IS_INITIALIZED);
  assert(userArray != NULL);

  STEP_DEBUG({array_print(&steprt_userArrayTable);});

  for(i=0; !stop && i<steprt_userArrayTable.len; i++)
    {
      desc_userArray = &(array_get_data_from_index(&steprt_userArrayTable, Descriptor_userArray, i));
      stop = desc_userArray->userArray == userArray;
    }
  if (!stop)
    desc_userArray = NULL;

  OUT_TRACE("desc_userArray = %p", desc_userArray);
  return desc_userArray;
}
void steprt_userArrayDescriptor_unset(Descriptor_userArray *desc)
{
  uint32_t id_node; 
  Array *uptodateArray = &(desc->uptodateArray);
  Array *interlacedArray = &(desc->interlacedArray);

  rg_composedRegion_unset(&(desc->boundsRegions));

  if (desc->savedUserArray)
    free(desc->savedUserArray);

  for (id_node=0; id_node<NB_NODES; id_node++)
    {
      rg_composedRegion_unset(&(array_get_data_from_index(uptodateArray, composedRegion, id_node)));
      rg_composedRegion_unset(&(array_get_data_from_index(interlacedArray, composedRegion, id_node)));
    }
  array_unset(uptodateArray);
  array_unset(interlacedArray);

}
static void steprt_save_userArray(Descriptor_userArray *desc_userArray)
{
  size_t alloc;

  desc_userArray->savedUserArray = communications_alloc_buffer(desc_userArray, &alloc);
  memcpy(desc_userArray->savedUserArray, desc_userArray->userArray, alloc);
}

/*##############################################################################

  Descriptor_shared

##############################################################################*/
static void steprt_set_sharedDescriptor(Descriptor_shared *desc_shared, Descriptor_userArray *desc_userArray)
{
  uint32_t nbdims;
  
  IN_TRACE("desc_shared = %p, desc_userArray =%p", desc_shared, desc_userArray);

  nbdims = rg_get_userArrayDims(&(desc_userArray->boundsRegions));
  desc_shared->userArray = desc_userArray->userArray;
  rg_composedRegion_set(&(desc_shared->receiveRegions), nbdims);
  rg_composedRegion_set(&(desc_shared->sendRegions), nbdims);
  desc_shared->interlaced_p = false;

  OUT_TRACE("end");
}


Descriptor_shared *steprt_find_in_sharedTable(void *userArray)
{
  size_t i;
  bool stop = false;
  Descriptor_shared *desc;
  Array *sharedTable;

  IN_TRACE("userArray = %p", userArray);

  TRACE_P("CURRENTWORKSHARING = %p", CURRENTWORKSHARING);

  assert(CURRENTWORKSHARING != NULL);

  if (CURRENTWORKSHARING == NULL)
    {
      fprintf(stderr, "CURRENTWORKSHARING == NULL\n");
      exit(EXIT_FAILURE);
    }

  
  sharedTable = &(CURRENTWORKSHARING->sharedArray);

  if (sharedTable == NULL)
    {
      fprintf(stderr, "sharedTable == NULL\n");
      exit(EXIT_FAILURE);
    }

  for(i=0; !stop && i<sharedTable->len; i++)
    {
      desc = &(array_get_data_from_index(sharedTable, Descriptor_shared, i));
      stop = desc->userArray == userArray;
    }
  
  if (!stop)
    desc = NULL;
  
  OUT_TRACE("desc = %p", desc);
  return desc;
}
static void steprt_shared_unset(Descriptor_shared *desc)
{
  rg_composedRegion_unset(&(desc->receiveRegions));
  rg_composedRegion_unset(&(desc->sendRegions));
}


/* FSC a quoi sert le parametre nb_workchunks (a part le assert) ? */

void steprt_set_sharedTable(void *userArray, uint32_t nb_workchunks, STEP_ARG *receiveBounds,
			       STEP_ARG *sendBounds, bool is_interlaced)
{
  Descriptor_userArray *desc_userArray;
  Descriptor_shared *desc_shared;

  IN_TRACE("userArray = %p, nb_workchunks = %d, receiveBounds = %p, sendBounds = %p, is_interlaced = %d", userArray, nb_workchunks, receiveBounds, sendBounds, is_interlaced);
  
  desc_userArray = steprt_find_in_userArrayTable(userArray);
  desc_shared = steprt_find_in_sharedTable(userArray);
  assert(desc_userArray);  

  TRACE_P("desc_shared = %p", desc_shared);
  STEP_DEBUG({steprt_print_worksharing(CURRENTWORKSHARING);});

  TRACE_P("NB_WORKCHUNKS = %d nb_workchunks = %d\n",  NB_WORKCHUNKS, nb_workchunks);

  assert((CURRENTWORKSHARING->type == do_work && nb_workchunks == NB_WORKCHUNKS) ||
	 (CURRENTWORKSHARING->type == parallel_work && nb_workchunks == 0) ||
	 (CURRENTWORKSHARING->type == critical_work && nb_workchunks == 1) ||
	 (CURRENTWORKSHARING->type == master_work && nb_workchunks == 1)); 

  if(!desc_shared)
    {
      Descriptor_shared desc_newshared;
      steprt_set_sharedDescriptor(&desc_newshared, desc_userArray);
      if(receiveBounds)
	{
#if (INDEX_TYPE == STEP_ARG)
	  rg_composedRegion_reset(&(desc_newshared.receiveRegions), receiveBounds, nb_workchunks);
#else
	  INDEX_TYPE tmp[2*nbdims*nb_workchunks];
	  uint32_t i, nbdims = rg_get_userArrayDims(&(desc_userArray->boundsRegions));
	  for(i=0; i<2*nbdims*nb_workchunks; i++)
	    tmp[i]=(INDEX_TYPE)receiveBounds[i];
	  rg_composedRegion_reset(&(desc_newshared.receiveRegions), tmp, nb_workchunks);
#endif
	}
      
      TRACE_P("Set region SEND desc_newshared.sendRegions = %p, sendBounds=%p", desc_newshared.sendRegions, sendBounds);
     
      if(sendBounds)
	{
#if (INDEX_TYPE == STEP_ARG)
	  rg_composedRegion_reset(&(desc_newshared.sendRegions), sendBounds, nb_workchunks);
	  TRACE_P("After RESET region SEND desc_newshared.sendRegions = %p, sendBounds=%p", desc_newshared.sendRegions, sendBounds);
#else
	  INDEX_TYPE tmp[2*nbdims*nb_workchunks];
	  uint32_t i, nbdims = rg_get_userArrayDims(&(desc_userArray->boundsRegions));
	  for(i=0; i<2*nbdims*nb_workchunks; i++)
	    tmp[i]=(INDEX_TYPE)sendBounds[i];
	  rg_composedRegion_reset(&(desc_newshared.sendRegions), tmp, nb_workchunks);
#endif 
	  desc_newshared.interlaced_p = is_interlaced;
	  if (is_interlaced)
	    steprt_save_userArray(desc_userArray);
	}
      TRACE_P("&(CURRENTWORKSHARING->sharedArray) = %p, &desc_newshared = %p", &(CURRENTWORKSHARING->sharedArray), &desc_newshared);
      array_append_vals(&(CURRENTWORKSHARING->sharedArray), &desc_newshared, 1);
    }
  else
    {
      if(receiveBounds)
	{
#if (INDEX_TYPE == STEP_ARG)
	  rg_composedRegion_reset(&(desc_shared->receiveRegions), receiveBounds, nb_workchunks);
#else
	  INDEX_TYPE tmp[2*nbdims*nb_workchunks];
	  uint32_t i, nbdims = rg_get_userArrayDims(&(desc_userArray->boundsRegions));
	  for(i=0; i<2*nbdims*nb_workchunks; i++)
	    tmp[i]=(INDEX_TYPE)receiveBounds[i];
	  rg_composedRegion_reset(&(desc_shared->receiveRegions), tmp, nb_workchunks);
#endif
	  
	}
      if(sendBounds)
	{
#if (INDEX_TYPE == STEP_ARG)
	  rg_composedRegion_reset(&(desc_shared->sendRegions), sendBounds, nb_workchunks);
#else
	  INDEX_TYPE tmp[2*nbdims*nb_workchunks];
	  uint32_t i, nbdims = rg_get_userArrayDims(&(desc_userArray->boundsRegions));
	  for(i=0; i<2*nbdims*nb_workchunks; i++)
	    tmp[i]=(INDEX_TYPE)sendBounds[i];
	  rg_composedRegion_reset(&(desc_shared->sendRegions), tmp, nb_workchunks);
#endif
	  desc_shared->interlaced_p = is_interlaced;
	  if (is_interlaced)
	    steprt_save_userArray(desc_userArray);
	}
    }
  OUT_TRACE("end");
}

/*##############################################################################

  Descriptor_worksharing

##############################################################################*/
static void steprt_print_worksharing(Descriptor_worksharing *desc_worksharing)
{
  printf("Worksharing %p ", desc_worksharing);
  if (desc_worksharing != NULL)
    {
      switch(desc_worksharing->type)
	{
	case parallel_work:
	  printf(": parallel_work\n");
	  break;
	case do_work:
	  printf(": do_work\n");
	  break;
	case master_work:
	  printf(": master_work\n");
	  break;
	case critical_work:
	  printf(": critical_work\n");
	  break;
	default:
	  printf(": UNKNOWN\n");
	  break;
	}
    }
}

static Descriptor_worksharing *steprt_worksharing_current(void)
{
  Descriptor_worksharing * desc_worksharing;

  if (steprt_worksharingTable.len == 0)
    desc_worksharing = NULL;
  else
    desc_worksharing = &(array_get_data_from_index(&steprt_worksharingTable, Descriptor_worksharing, steprt_worksharingTable.len -1));

  return desc_worksharing;
}
void steprt_worksharing_set(worksharing_type type)
{
  Descriptor_worksharing desc_worksharing;
  uint32_t nbdims = 1;

  assert(IS_INITIALIZED);

  desc_worksharing.type = type;
  rg_composedRegion_set(&(desc_worksharing.workchunkRegions), nbdims);
  array_set(&(desc_worksharing.scheduleArray), uint32_t);
  array_set(&(desc_worksharing.sharedArray), Descriptor_shared);
  array_set(&(desc_worksharing.privateArray), void *);
  array_set(&(desc_worksharing.reductionsArray), Descriptor_reduction);
  array_set(&(desc_worksharing.communicationsArray), MPI_Request);

  array_append_vals(&steprt_worksharingTable, &desc_worksharing, 1);
  CURRENTWORKSHARING = steprt_worksharing_current();
}

void steprt_worksharing_unset(void)
{
  size_t i;

  IN_TRACE("begin");
  assert(CURRENTWORKSHARING != NULL);

  for(i=0; i<CURRENTWORKSHARING->sharedArray.len; i++)
    steprt_shared_unset(&(array_get_data_from_index(&(CURRENTWORKSHARING->sharedArray), Descriptor_shared, i)));

  array_unset(&(CURRENTWORKSHARING->sharedArray));
  array_unset(&(CURRENTWORKSHARING->scheduleArray));
  array_unset(&(CURRENTWORKSHARING->privateArray));
  array_unset(&(CURRENTWORKSHARING->reductionsArray));
  array_unset(&(CURRENTWORKSHARING->communicationsArray));
  rg_composedRegion_unset(&(CURRENTWORKSHARING->workchunkRegions));

  array_remove_index_fast(&steprt_worksharingTable, steprt_worksharingTable.len -1);
  CURRENTWORKSHARING = steprt_worksharing_current();
  OUT_TRACE("end");
}

/* Compute workchunks and update the workchunk list of the current worksharing */
composedRegion *steprt_compute_workchunks(INDEX_TYPE begin, INDEX_TYPE end, INDEX_TYPE incr, STEP_ARG nb_workchunks)
{
  INDEX_TYPE workchunks[nb_workchunks][2];
  INDEX_TYPE id_work;
  INDEX_TYPE nb_work;
  STEP_ARG id_workchunk;
    
  IN_TRACE("begin = %d, end = %d, incr = %d, nb_workchunks = %d", begin, end, incr, nb_workchunks);

  assert(nb_workchunks != 0);
  assert(incr != 0);

  id_work =  begin;
  nb_work = (end - begin) / (incr) + 1; /* nb_work < 0 if no work to do*/
  id_workchunk = 0;

  if (nb_work > nb_workchunks) /* at least 1 workchunk with 2 works */
    {
      INDEX_TYPE id_max_work = nb_work % nb_workchunks;
      INDEX_TYPE nb_max_work = (nb_work - id_max_work) / nb_workchunks;
      
      if(id_max_work == 0) /* same workchunk length */
	{
	  id_max_work = nb_workchunks;
	  nb_max_work -= 1;
	}

      /* each first id_max_work workchunks have nb_max_work works */
      for (; id_workchunk < id_max_work; id_workchunk++)
	{
	  workchunks[id_workchunk][LOW(0)] = id_work;
	  id_work += nb_max_work * incr;
	  workchunks[id_workchunk][UP(0)] = id_work;
	  id_work += incr;
	}
      
      /* remaining workchunks have nb_max_work-1 */
      nb_max_work -= 1;
      for (; id_workchunk < nb_workchunks; id_workchunk++)
	{
	  workchunks[id_workchunk][LOW(0)] = id_work;
	  id_work += nb_max_work * incr;
	  workchunks[id_workchunk][UP(0)] = id_work;
	  id_work += incr;
	}
    }
  else /* at most 1 work by workchunk */
    {
      for (; id_workchunk < nb_work; id_workchunk++)
	{
	  workchunks[id_workchunk][LOW(0)] = workchunks[id_workchunk][UP(0)] = id_work;
	  id_work += incr;
	}
    }

  /* fill remaining workchunk with no work */
  for (; id_workchunk < nb_workchunks; id_workchunk++)
    {
      workchunks[id_workchunk][LOW(0)] = incr;
      workchunks[id_workchunk][UP(0)] = 0;
    }

  rg_composedRegion_reset(&(CURRENTWORKSHARING->workchunkRegions), *workchunks, nb_workchunks); 

  OUT_TRACE("&(CURRENTWORKSHARING->workchunkRegions) = %p", &(CURRENTWORKSHARING->workchunkRegions));
  return &(CURRENTWORKSHARING->workchunkRegions);
}

void steprt_set_userArrayTable(void *userArray, uint32_t type, uint32_t nbdims, INDEX_TYPE *bounds)
{
  Descriptor_userArray *desc_userArray;

  IN_TRACE("userArray = %p, type = %d, nbdims = %d, bounds = %p", userArray, type, nbdims, bounds);
  
  STEP_DEBUG({rg_simpleRegion_print(nbdims, bounds);});

  desc_userArray = steprt_find_in_userArrayTable(userArray);

  TRACE_P("desc_userArray = %p", desc_userArray);
  if (!desc_userArray)
    {
      Descriptor_userArray new_desc;

      array_append_vals(&steprt_userArrayTable, &new_desc, 1);
      desc_userArray = &(array_get_data_from_index(&steprt_userArrayTable, Descriptor_userArray, steprt_userArrayTable.len -1));
      steprt_set_userArrayDescriptor(desc_userArray, userArray, type, nbdims, bounds);
    }
  steprt_set_sharedTable(userArray, 0, NULL, NULL, false);

  STEP_DEBUG({
      Descriptor_userArray *desc_userArray = steprt_find_in_userArrayTable(userArray);
      
      assert(desc_userArray);
      rg_composedRegion_print(&(desc_userArray->boundsRegions));
    })

    OUT_TRACE("end");
}

Descriptor_reduction *steprt_find_reduction(void *scalar)
{
  size_t i;
  Descriptor_reduction *desc;
  bool stop = false;

  IN_TRACE("scalar = %p", scalar);

  assert(CURRENTWORKSHARING);

  for(i=0; !stop && i<CURRENTWORKSHARING->reductionsArray.len; i++)
    {
      desc = &(array_get_data_from_index(&(CURRENTWORKSHARING->reductionsArray), Descriptor_reduction, i));
      stop = desc->variable == scalar;
    }
  if (!stop)
    desc = NULL;

  OUT_TRACE("desc = %p", desc);
  return desc;

}
/*##############################################################################

  Compute Region Communicated

##############################################################################*/

/*
  UPTODATE(id_node) = ALL_ARRAY 
  INTERLACED(id_node) = EMPTY
*/
static void uptodate_regions_reset_for_communications(Descriptor_userArray *desc_userArray)
{
  uint32_t id_node;
  INDEX_TYPE *bounds;
  Array *uptodateArray;
  Array *interlacedArray;

  IN_TRACE("desc_userArray = %p", desc_userArray);
  
  assert(rg_get_nb_simpleRegions(&(desc_userArray->boundsRegions)) == 1);

  bounds = rg_get_simpleRegion(&(desc_userArray->boundsRegions), 0);
  uptodateArray = &(desc_userArray->uptodateArray);
  interlacedArray = &(desc_userArray->interlacedArray);

  assert(uptodateArray->len == interlacedArray->len);
  if(uptodateArray->len == 0 || interlacedArray->len == 0)
    {
      uint32_t nbdims;
      composedRegion emptyRegions;

      nbdims = rg_get_userArrayDims(&(desc_userArray->boundsRegions));

      rg_composedRegion_set(&emptyRegions, nbdims);  
      array_reset(uptodateArray, NULL, NB_NODES);
      array_reset(interlacedArray, NULL, NB_NODES);
      for (id_node=0; id_node<NB_NODES; id_node++)
	{
	  array_append_vals(uptodateArray, &emptyRegions, 1);
	  array_append_vals(interlacedArray, &emptyRegions, 1);
	}
    }
  for (id_node=0; id_node<NB_NODES; id_node++)
    {
      composedRegion *uptodateRegions = &(array_get_data_from_index(uptodateArray, composedRegion, id_node));
      composedRegion *interlacedRegions = &(array_get_data_from_index(&(desc_userArray->interlacedArray), composedRegion, id_node));
	
      rg_composedRegion_reset(interlacedRegions, NULL, 0);
      rg_composedRegion_reset(uptodateRegions, bounds, 1);
    }
  OUT_TRACE("end");
}
/*
  LOCAL(id_node) = LOCAL(id_node) union SEND(id_node)
*/
static void local_recompute_regions_for_communications(Array *local, Array *toSend)
{
  uint32_t id_node;
  
  IN_TRACE("local = %p, toSend = %p", local, toSend);

  for (id_node=0; id_node<NB_NODES; id_node++)
    {
      composedRegion *r_local;
      composedRegion *r_send;

      r_local = &(array_get_data_from_index(local, composedRegion, id_node));
      r_send = &(array_get_data_from_index(toSend, composedRegion, id_node));
      rg_composedRegion_union(r_local, r_send);
    }
  OUT_TRACE("end");
}
/*
  UPTODATE(id_node) = UPTODATE(id_node) union SEND(id_node)
  INTERLACED(id_node) = INTERLACED(id_node) union SEND(id_node)
  UPTODATE(id_node) = UPTODATE(id_node) minus { union(n != id_node) SEND(n) }
*/
static void local_uptodate_recompute_for_communications(Descriptor_userArray *desc_userArray, Array *toSend)
{
  composedRegion other_send;
  composedRegion *userArrayBounds;
  Array *uptodateArray;
  Array *interlacedArray;
  uint32_t id_node, id_node_rk;

  IN_TRACE("desc_userArray = %p, toSend = %p", desc_userArray, toSend);

  userArrayBounds = &(desc_userArray->boundsRegions);
  uptodateArray = &(desc_userArray->uptodateArray);
  interlacedArray = &(desc_userArray->interlacedArray);

  /*
    UPTODATE(id_node) = UPTODATE(id_node) union SEND(id_node)
    INTERLACED(id_node) = INTERLACED(id_node) union SEND(id_node)
  */
  local_recompute_regions_for_communications(uptodateArray, toSend);
  local_recompute_regions_for_communications(interlacedArray, toSend);

  /*
    UPTODATE(id_node) = UPTODATE(id_node) minus { union(n != id_node) SEND(n) }
  */
  rg_composedRegion_set(&other_send, rg_get_userArrayDims(userArrayBounds));
  for (id_node_rk=0; id_node_rk<NB_NODES; id_node_rk++)
    {
      composedRegion *uptodateRegions;
      
      uptodateRegions = &(array_get_data_from_index(uptodateArray, composedRegion, id_node_rk));
      
      rg_composedRegion_reset(&other_send, NULL, 0);
      for (id_node=0; id_node<NB_NODES; id_node++)
	if (id_node != id_node_rk)
	  {
	    composedRegion *r_send;
	    
	    r_send = &(array_get_data_from_index(toSend, composedRegion, id_node));

	    rg_composedRegion_union(&other_send, r_send);
	  }
      
      rg_composedRegion_difference(uptodateRegions, &other_send);
      rg_composedRegion_simplify(uptodateRegions, userArrayBounds);
    }
  rg_composedRegion_unset(&other_send);

  OUT_TRACE("end");
}
/*
  INTERLACED(id_node) = INTERLACED(id_node) minus UPTODATE(id_node)
*/
static void communications_regions_interlaced_recompute(Descriptor_userArray *desc_userArray)
{
  composedRegion *userArrayBounds;
  Array *uptodateArray;
  Array *interlacedArray;
  uint32_t id_node;

  IN_TRACE("desc_userArray = %p", desc_userArray);

  userArrayBounds = &(desc_userArray->boundsRegions);
  uptodateArray = &(desc_userArray->uptodateArray);
  interlacedArray = &(desc_userArray->interlacedArray);

  for (id_node=0; id_node<NB_NODES; id_node++)
    {
      composedRegion *interlacedRegions = &(array_get_data_from_index(interlacedArray, composedRegion, id_node));
      composedRegion *uptodateRegions = &(array_get_data_from_index(uptodateArray, composedRegion, id_node));

      rg_composedRegion_difference(interlacedRegions, uptodateRegions);
      rg_composedRegion_simplify(interlacedRegions, userArrayBounds);
    }

  OUT_TRACE("end");
}
/*
  UPTODATE(id_node) = UPTODATE(id_node) union RECV(id_node)
*/
static void communications_regions_partial_uptodate_recompute(Descriptor_userArray *desc_userArray, Array *toReceive)
{ 
  composedRegion *userArrayBounds;
  Array *uptodateArray;
  uint32_t id_node;

  IN_TRACE("desc_userArray = %p, toReceive = %p", desc_userArray, toReceive);

  assert(desc_userArray);
  
  userArrayBounds = &(desc_userArray->boundsRegions);
  uptodateArray = &(desc_userArray->uptodateArray);

  for (id_node=0; id_node<NB_NODES; id_node++)
    {
      composedRegion *uptodateRegions = &(array_get_data_from_index(uptodateArray, composedRegion, id_node));
      composedRegion *r_receive = &(array_get_data_from_index(toReceive, composedRegion, id_node));
      
      rg_composedRegion_union(uptodateRegions, r_receive);
      rg_composedRegion_simplify(uptodateRegions, userArrayBounds);
    }

  OUT_TRACE("end");
}
/*
  Repartition des NB_WORKCHUNKS regions "workchunksRegions" (region SEND ou RECV) entre les NB_NODES regions "regionsToExchange"

  Le schedule par defaut impose NB_WORKCHUNKS==NB_NODES et retourne : regionsToExchange(i)<-workchunksRegions(i)
*/
static void communications_schedule_regions(Descriptor_userArray *desc_userArray, composedRegion *workchunksRegions, Array *regionsToExchange)
{
  uint32_t id_node, nbdims;
  composedRegion emptyRegions;

  IN_TRACE("desc_userArray = %p, workchunksRegions = %p, regionsToExchange = %p", desc_userArray, workchunksRegions, regionsToExchange);

  assert(desc_userArray);

  nbdims = rg_get_userArrayDims(&(desc_userArray->boundsRegions));

  // initialisation : regionsToExchange(id_node) = EMPTY
  rg_composedRegion_set(&emptyRegions, nbdims);
  array_set(regionsToExchange, composedRegion);
  array_reset(regionsToExchange, NULL, NB_NODES);

  for (id_node=0; id_node<NB_NODES; id_node++)
    array_append_vals(regionsToExchange, &emptyRegions, 1);

  // repartition des workchunksRegions selon le scheduling par defaut : le noeud i traite le workchunk i
  for(id_node=0; id_node<rg_get_nb_simpleRegions(workchunksRegions); id_node++)    
    {
      composedRegion *node_send = &(array_get_data_from_index(regionsToExchange, composedRegion, id_node));
      INDEX_TYPE *workchunk_region_bounds = rg_get_simpleRegion(workchunksRegions, id_node);
      array_append_vals(&(node_send->simpleRegionArray), workchunk_region_bounds, 1);
    }

  // workchunksRegions n'est plus utile les regions sont maintenant defini par regionsToExchange
  rg_composedRegion_reset(workchunksRegions, NULL, 0);

  OUT_TRACE("end");
}
static void communications_allToAll_local(Descriptor_userArray *desc_userArray,
					  Descriptor_shared *desc_shared)
{
  Array regionSend;
  uint32_t id_node;

  IN_TRACE("desc_userArray = %p, desc_shared = %p", desc_userArray, desc_shared);

  communications_schedule_regions(desc_userArray, &(desc_shared->sendRegions), &regionSend);

  /*
    UPTODATE(id_node) = UPTODATE(id_node) union SEND(id_node)
    INTERLACED(id_node) = INTERLACED(id_node) union SEND(id_node)
    UPTODATE(id_node) = UPTODATE(id_node) minus { union(n != id_node) SEND(n) }
    
    INTERLACED(id_node) = INTERLACED(id_node) minus UPTODATE(id_node)
  */
  local_uptodate_recompute_for_communications(desc_userArray, &regionSend);
  communications_regions_interlaced_recompute(desc_userArray);
  
  for (id_node=0; id_node<NB_NODES; id_node++)
    rg_composedRegion_unset(&(array_get_data_from_index(&regionSend, composedRegion, id_node)));
  array_unset(&regionSend);

  OUT_TRACE("end");
}
static void communications_allToAll_partial(Descriptor_userArray *desc_userArray,
					    Descriptor_shared *desc_shared,
					    uint32_t algorithm, int_MPI tag)
{
  Array regionReceive; 
  uint32_t id_node;

  IN_TRACE("desc_userArray = %p, desc_shared = %p, algorithm = %d, tag = %d", desc_userArray, desc_shared, algorithm, tag);

  communications_schedule_regions(desc_userArray, &(desc_shared->receiveRegions), &regionReceive);

  /* Communication */
  communications_allToAll(desc_userArray, &regionReceive, algorithm, tag);
  
  /*
    UPTODATE(id_node) = UPTODATE(id_node) union RECV(id_node)
    INTERLACED(id_node) = INTERLACED(id_node) minus UPTODATE(id_node)
  */
  communications_regions_partial_uptodate_recompute(desc_userArray, &regionReceive);
  communications_regions_interlaced_recompute(desc_userArray);
  
  for (id_node=0; id_node<NB_NODES; id_node++)
    rg_composedRegion_unset(&(array_get_data_from_index(&regionReceive, composedRegion, id_node)));
  array_unset(&regionReceive);

  OUT_TRACE("end");
}

static void communications_allToAll_full(Descriptor_userArray *desc_userArray,
					 Descriptor_shared *desc_shared,
					 uint32_t algorithm, int_MPI tag)
{
  IN_TRACE("desc_userArray = %p, desc_shared = %p, algorithm = %d, tag = %d", desc_userArray, desc_shared, algorithm, tag);

  TRACE_P("&(desc_shared->receiveRegions) = %p", &(desc_shared->receiveRegions));
  rg_composedRegion_reset(&(desc_shared->receiveRegions), NULL, 0); // no more nedded.
  communications_allToAll(desc_userArray, NULL, algorithm, tag);
  
  /*
    UPTODATE(id_node) = ALL_ARRAY
    INTERLACED(id_node) = EMPTY
  */
  uptodate_regions_reset_for_communications(desc_userArray);
  OUT_TRACE("end");
}

void steprt_alltoall(void * userArray, bool full_p, uint32_t algorithm, int_MPI tag)
{ 
  Descriptor_shared *desc_shared;
  Descriptor_userArray *desc_userArray;


  IN_TRACE("userArray = %p, full_p = %d, algorithm = %d, tag = %d", userArray, full_p, algorithm, tag);

  desc_shared = steprt_find_in_sharedTable(userArray);
  desc_userArray = steprt_find_in_userArrayTable(userArray);

  assert(desc_shared);
  assert(desc_userArray);

  TRACE_P("userArray nbdims = %d\n", rg_get_userArrayDims(&(desc_userArray->boundsRegions)));
  TRACE_P("shared sendRegions nbdims = %d\n", rg_get_userArrayDims(&(desc_shared->sendRegions)));

  if(rg_get_nb_simpleRegions(&(desc_shared->sendRegions)) != 0 )
    {
      /* Locally no communications, will update uptodateArray */
      communications_allToAll_local(desc_userArray, desc_shared);
    }
  if(full_p)
    {
      communications_allToAll_full(desc_userArray, desc_shared, algorithm, tag);
    }
  else if(rg_get_nb_simpleRegions(&(desc_shared->receiveRegions)) != 0 )
    { 
      communications_allToAll_partial(desc_userArray, desc_shared, algorithm, tag);
    }
  
  OUT_TRACE("end");
}

void steprt_finalize()
{
  size_t i;

  /* FSC creer une fonction communications_critical_stop_pcoord */ 
  communications_critical_stop_pcoord();
  
  communications_finalize();
  
  steprt_params.commsize = 0;
  steprt_params.rank = 0;
  steprt_params.parallel_level = 0;
  steprt_params.initialized = 0;

  for (i=0; i<steprt_userArrayTable.len; i++) 
    steprt_userArrayDescriptor_unset(&(array_get_data_from_index(&steprt_userArrayTable, Descriptor_userArray, i)));
  array_unset(&steprt_userArrayTable);

  for (i=0; i<steprt_worksharingTable.len; i++)
    steprt_worksharing_unset();
  array_unset(&steprt_worksharingTable);
}

void steprt_initreduction(void *variable, uint32_t op, uint32_t type)
{
  Descriptor_reduction desc_reduction;

  desc_reduction.variable = variable;
  desc_reduction.type = type;
  desc_reduction.operator = op;

  Descriptor_userArray *desc_userArray = steprt_find_in_userArrayTable(variable);
  if (desc_userArray == NULL) /* scalar reduction */
    communications_initreduction(&desc_reduction, desc_userArray);
  /* for array reduction : communications_initreduction called from STEP_SET_REDUCTION_SENDREGIONS */

  array_append_vals(&(CURRENTWORKSHARING->reductionsArray), &desc_reduction, 1);

}


/*
  For array reduction, the SEND_REGIONS are ignored : the reduction is performed on the whole array
*/
void steprt_set_reduction_sendregions(Descriptor_reduction *desc_reduction, uint32_t nb_workchunks, STEP_ARG *regions)
{
  void *userArray=desc_reduction->variable;

  /* enregistremant des regions SEND + sauvegarde des valeurs initiales */
  steprt_set_sharedTable(userArray, nb_workchunks, NULL, regions, true);

  /* initialisation à l'element neutre */
  Descriptor_userArray *desc_userArray = steprt_find_in_userArrayTable(userArray);
  communications_initreduction(desc_reduction, desc_userArray);
}

void steprt_reduction(void *variable)
{
  Descriptor_reduction *desc_reduction = steprt_find_reduction(variable);
  communications_reduction(desc_reduction);
  Descriptor_userArray *desc_userArray = steprt_find_in_userArrayTable(variable);
  if (desc_userArray != NULL)
    {
      /*
	UPTODATE(id_node) = ALL_ARRAY
	INTERLACED(id_node) = EMPTY
      */
      uptodate_regions_reset_for_communications(desc_userArray);
    }
}

#ifdef TEST_STEPRT

#include <stdlib.h>
#include <stdio.h>

void test_compute_workchunks()
{
  composedRegion *workchunk_regionlist;
  INDEX_TYPE begin, end, incr;
  STEP_ARG nb_workchunks;

  steprt_worksharing_set(do_work);

  util_print_rank(0, "TEST COMPUTE_WORKCHUNKS\n");
  util_print_rank(0, "-----------------------\n");

  util_print_rank(0, "Creating a loop from begin=1 to end=10 (incr=1)\n");
  begin = 1;
  end = 10;
  incr = 1;

  util_print_rank(0, "Computing the workchunk region list with 5 workchunks\n");
  nb_workchunks = 5;
  workchunk_regionlist = steprt_compute_workchunks(begin, end, incr, nb_workchunks);
  if (MYRANK == 0) rg_composedRegion_print(workchunk_regionlist);
  util_print_rank(0, "\n");

  util_print_rank(0, "Computing the workchunk region list with 4 workchunks\n");
  nb_workchunks = 4;
  workchunk_regionlist = steprt_compute_workchunks(begin, end, incr, nb_workchunks);
  if (MYRANK == 0) rg_composedRegion_print(workchunk_regionlist);
  util_print_rank(0, "\n");

  util_print_rank(0, "Computing the workchunk region list with 10 workchunks\n");
  nb_workchunks = 10;
  workchunk_regionlist = steprt_compute_workchunks(begin, end, incr, nb_workchunks);
  if (MYRANK == 0) rg_composedRegion_print(workchunk_regionlist);
  util_print_rank(0, "\n");

  util_print_rank(0, "Computing the workchunk region list with 11 workchunks\n");
  nb_workchunks = 11;
  workchunk_regionlist = steprt_compute_workchunks(begin, end, incr, nb_workchunks);
  if (MYRANK == 0) rg_composedRegion_print(workchunk_regionlist);
  util_print_rank(0, "\n");

  util_print_rank(0, "Creating a loop from begin=10 to end=1 (incr=1)\n");
  begin = 10;
  end = 1;
  util_print_rank(0, "Computing the workchunk region list with 11 workchunks\n");
  workchunk_regionlist = steprt_compute_workchunks(begin, end, incr, nb_workchunks);
  if (MYRANK == 0) rg_composedRegion_print(workchunk_regionlist);

  util_print_rank(0, "Creating a loop from begin=10 to end=1 (incr=-1)\n");
  incr = -1;
  util_print_rank(0, "Computing the workchunk region list with 11 workchunks\n");
  workchunk_regionlist = steprt_compute_workchunks(begin, end, incr, nb_workchunks);
  if (MYRANK == 0) rg_composedRegion_print(workchunk_regionlist);

  util_print_rank(0, "Computing the workchunk region list with 11 workchunks\n");
  nb_workchunks = 10;
  workchunk_regionlist = steprt_compute_workchunks(begin, end, incr, nb_workchunks);
  if (MYRANK == 0) rg_composedRegion_print(workchunk_regionlist);

  util_print_rank(0, "Computing the workchunk region list with 4 workchunks\n");
  nb_workchunks = 4;
  workchunk_regionlist = steprt_compute_workchunks(begin, end, incr, nb_workchunks);
  if (MYRANK == 0) rg_composedRegion_print(workchunk_regionlist);

  util_print_rank(0, "Computing the workchunk region list with 5 workchunks\n");
  nb_workchunks = 5;
  workchunk_regionlist = steprt_compute_workchunks(begin, end, incr, nb_workchunks);
  if (MYRANK == 0) rg_composedRegion_print(workchunk_regionlist);

  util_print_rank(0, "Creating a loop from begin=10 to end=1 (incr=-2)\n");
  incr = -2;
  util_print_rank(0, "Computing the workchunk region list with 5 workchunks\n");
  nb_workchunks = 5;
  workchunk_regionlist = steprt_compute_workchunks(begin, end, incr, nb_workchunks);
  if (MYRANK == 0) rg_composedRegion_print(workchunk_regionlist);

  util_print_rank(0, "Computing the workchunk region list with 4 workchunks\n");
  nb_workchunks = 4;
  workchunk_regionlist = steprt_compute_workchunks(begin, end, incr, nb_workchunks);
  if (MYRANK == 0) rg_composedRegion_print(workchunk_regionlist);

  util_print_rank(0, "Creating a loop from begin=1 to end=10 (incr=2)\n");
  incr = 2;
  begin = 1;
  end = 10;
  util_print_rank(0, "Computing the workchunk region list with 4 workchunks\n");
  workchunk_regionlist = steprt_compute_workchunks(begin, end, incr, nb_workchunks);
  if (MYRANK == 0) rg_composedRegion_print(workchunk_regionlist);

  util_print_rank(0, "Computing the workchunk region list with 5 workchunks\n");
  nb_workchunks = 5;
  workchunk_regionlist = steprt_compute_workchunks(begin, end, incr, nb_workchunks);
  if (MYRANK == 0) rg_composedRegion_print(workchunk_regionlist);

  util_print_rank(0, "Creating a loop from begin=10 to end=1 (incr=2)\n");
  begin = 10;
  end = 1;
  util_print_rank(0, "Computing the workchunk region list with 5 workchunks\n");
  workchunk_regionlist = steprt_compute_workchunks(begin, end, incr, nb_workchunks);
  if (MYRANK == 0) rg_composedRegion_print(workchunk_regionlist);

  util_print_rank(0, "Creating a loop from begin=1 to end=1 (incr=2)\n");
  begin = 1;
  end = 1;
  util_print_rank(0, "Computing the workchunk region list with 5 workchunks\n");
  workchunk_regionlist = steprt_compute_workchunks(begin, end, incr, nb_workchunks);
  if (MYRANK == 0) rg_composedRegion_print(workchunk_regionlist);

  util_print_rank(0, "Creating a loop from begin=1 to end=1 (incr=2)\n");
  incr = 1;
  util_print_rank(0, "Computing the workchunk region list with 5 workchunks\n");
  workchunk_regionlist = steprt_compute_workchunks(begin, end, incr, nb_workchunks);
  if (MYRANK == 0) rg_composedRegion_print(workchunk_regionlist);

  util_print_rank(0, "Creating a loop from begin=1 to end=10 (incr=1)\n");
  begin = 1;
  end = 10;
  incr = 1;
  util_print_rank(0, "Computing the workchunk region list with 2 workchunks\n");
  nb_workchunks = 2;
  workchunk_regionlist = steprt_compute_workchunks(begin, end, incr, nb_workchunks);
  if (MYRANK == 0) rg_composedRegion_print(workchunk_regionlist);

  steprt_worksharing_unset();
}

#define N 10
#define MAX_NB_WORKCHUNKS 16

#define LOW1 0
#define UP1 N
#define LOW2 0
#define UP2 4

void test_steprt_1D() {

  int p;
  int nbdims;
  STEP_ARG type;
  int A[UP1];
  INDEX_TYPE bounds[BOUNDS(1)];
  Descriptor_userArray *desc;
  int incr, nb_workchunks;
  INDEX_TYPE shared_r[MAX_NB_WORKCHUNKS][1][2];
  Descriptor_shared *shared_desc_list;
  INDEX_TYPE *receiveBounds, *sendBounds;
  int interlaced_p;
  int low1, up1;

  util_print_rank(0, "Testing STEPRT with a 1D array\n");
  util_print_rank(0, "------------------------------\n");
  util_print_rank(0, "1) Creating bounds userArray\n");

  util_print_rank(0, "1) Creating bounds userArray\n");
  low1 = LOW1;
  up1 = UP1 - 1 ;
  bounds[LOW(0)] = low1;
  bounds[UP(0)] = up1;

  util_print_rank(0, "2) Initializing a PARALLEL DO construct\n");
  steprt_worksharing_set(parallel_work);

  steprt_worksharing_set(do_work);

  util_print_rank(0, "3) Initializing array table with A and bounds array, type == STEP_INTEGER, nbdims == 1\n");
  nbdims = 1;
  type = STEP_INTEGER;
  steprt_set_userArrayTable(A, type, nbdims, bounds);

  desc = steprt_find_in_userArrayTable(A);
  util_print_rank(0, "\n3.1) descriptor de A = %p\n",desc);
  if (desc)
    rg_composedRegion_print(&(desc->boundsRegions));

  util_print_rank(0, "\n4) Computing workchunks for the DO loop\n");
  incr = 1;
  nb_workchunks = NB_NODES;
  steprt_compute_workchunks(low1, up1, incr, nb_workchunks);

  util_print_rank(0, "5) Creating shared_r array\n");

  for (p = 0; p < NB_NODES; p++)
    {
      int *bounds;
      int i_low1, i_up1;
      
      bounds = rg_get_simpleRegion(&(CURRENTWORKSHARING->workchunkRegions), p);
      i_low1 = bounds[LOW(0)];
      i_up1 = bounds[UP(0)];

      /* WARNING: FORTRAN and C order of dimensions are different */
      /* p: process rank, 0: first (and only) dimension, 0: low index */
      shared_r[p][0][0]= i_low1;
      /* p: process rank, 0: first (and only) dimension, 1: up index */
      shared_r[p][0][1]= i_up1;
      
      util_print_rank(0, "\tshared_r[%d][0][%d] = %d, shared_r[%d][0][%d] = %d\n", p, 0, shared_r[p][0][0], p, 1, shared_r[p][0][1]);
    }

  nb_workchunks = NB_NODES;
  receiveBounds = NULL;
  sendBounds = (STEP_ARG *)shared_r;
  interlaced_p = true;
  util_print_rank(0, "6) Setting A with %d workchunks as shared SEND interlaced\n", nb_workchunks);
  
  steprt_set_sharedTable(A, nb_workchunks, receiveBounds, sendBounds, interlaced_p);

  util_print_rank(0, "6.1) Retrieve shared_desc_list from A\n");
  shared_desc_list = steprt_find_in_sharedTable(A);
  util_print_rank(0, "6.2) is share interlaced ? interlaced_p=%d\n", shared_desc_list->interlaced_p);
  util_print_rank(0, "6.3) Printing send regions from shared_desc_list:\n");
  if (MYRANK == 0) rg_composedRegion_print(&shared_desc_list->sendRegions);
  util_print_rank(0, "6.4) Printing receive regions from shared_desc_list:\n");
  if (MYRANK == 0) rg_composedRegion_print(&shared_desc_list->receiveRegions);

  steprt_worksharing_unset();
  steprt_worksharing_unset();
}

void test_steprt_2D() {
  int p;
  int nbdims;
  int B[UP1][UP2];
  STEP_ARG type;
  INDEX_TYPE bounds[BOUNDS(2)];
  Descriptor_userArray *desc;
  int incr, nb_workchunks;
  INDEX_TYPE shared_r[MAX_NB_WORKCHUNKS][2][2];
  Descriptor_shared *shared_desc_list;
  INDEX_TYPE *receiveBounds, *sendBounds;
  int interlaced_p;
  int low1, up1, low2, up2;

  util_print_rank(0, "Testing STEPRT with a 2D array\n");
  util_print_rank(0, "------------------------------\n");
  util_print_rank(0, "1) Creating bounds userArray\n");
  low1 = LOW1;
  up1 = UP1 - 1 ;
  low2 = LOW2;
  up2  = UP2 - 1;
  bounds[LOW(0)] = low1;
  bounds[UP(0)] = up1;
  bounds[LOW(1)] = low2;
  bounds[UP(1)] = up2;

  util_print_rank(0, "2) Initializing a PARALLEL DO construct\n");
  steprt_worksharing_set(parallel_work);

  steprt_worksharing_set(do_work);

  util_print_rank(0, "3) Initializing array table with B and bounds array, type == STEP_INTEGER, nbdims == 2\n");
  nbdims = 2;
  type = STEP_INTEGER;
  steprt_set_userArrayTable(B, type, nbdims, bounds);

  util_print_rank(0, "4) Computing workchunks for the DO loop\n");
  incr = 1;
  nb_workchunks = NB_NODES;
  steprt_compute_workchunks(low1, up1, incr, nb_workchunks);

  desc = steprt_find_in_userArrayTable(B);
  util_print_rank(0, "4.1) descriptor de B=%p\n",desc);
  if (desc)
    rg_composedRegion_print(&(desc->boundsRegions));
  steprt_print();


  util_print_rank(0, "5) Creating shared_r array\n");

  for (p = 0; p < NB_NODES; p++)
    {
      int *bounds;
      int i_low, i_up;
      
      bounds = rg_get_simpleRegion(&(CURRENTWORKSHARING->workchunkRegions), p);
      i_low = bounds[LOW(0)];
      i_up = bounds[UP(0)];

      /* WARNING: FORTRAN and C order of dimensions are different */
      /* p: process rank, 0: first dimension, 0: low index */
      shared_r[p][0][0] = i_low;
      /* p: process rank, 0: first dimension, 1: up index */
      shared_r[p][0][1] = i_up;
      /* p: process rank, 1: second dimension, 0: low index */
      shared_r[p][1][0] = low2;
      /* p: process rank, 1: second dimension, 1: up index */
      shared_r[p][1][1] = up2;
      util_print_rank(0, "\tshared_r[%d][0][%d] = %d, shared_r[%d][0][%d] = %d\n", p, 0, shared_r[p][0][0], p, 1, shared_r[p][0][1]);
      util_print_rank(0, "\tshared_r[%d][1][%d] = %d, shared_r[%d][1][%d] = %d\n", p, 0, shared_r[p][1][0], p, 1, shared_r[p][1][1]);
    }

  nb_workchunks = NB_NODES;
  receiveBounds = (STEP_ARG *)shared_r;
  sendBounds = NULL;
  interlaced_p = false;
  util_print_rank(0, "6) Setting B with %d workchunks as shared_desc_list RECEIVE (not interlaced)\n", nb_workchunks);
  steprt_set_sharedTable(B, nb_workchunks, receiveBounds, sendBounds, interlaced_p);
  util_print_rank(0, "6.1)Retrieve shared from B\n");
  shared_desc_list = steprt_find_in_sharedTable(B);
  util_print_rank(0, "6.2) send interlaced : %d\n", shared_desc_list->interlaced_p);
  util_print_rank(0, "6.3) Printing send regions from shared_desc_list:\n");
  rg_composedRegion_print(&shared_desc_list->sendRegions);
  util_print_rank(0, "6.4) Printing receive regions from shared_desc_list:\n");
  if (MYRANK == 0) rg_composedRegion_print(&shared_desc_list->receiveRegions);
  if (MYRANK == 0) steprt_print();

  steprt_worksharing_unset();
  steprt_worksharing_unset();
}

#define TEST_STEPRT_COMPUTE_WORKCHUNKS

#define TEST_STEPRT_1D

#define TEST_STEPRT_2D

int main(int argc, char **argv)
{
  steprt_init(STEP_FORTRAN);

  SET_TRACES("traces", NULL, 1, 0);

#ifdef TEST_STEPRT_COMPUTE_WORKCHUNKS
  test_compute_workchunks();
#endif

#ifdef TEST_STEPRT_1D
  util_print_rank(0, "\n\n");
  test_steprt_1D();
#endif

#ifdef TEST_STEPRT_2D
  util_print_rank(0, "\n\n");
  test_steprt_2D();
#endif

  steprt_finalize();
  return EXIT_SUCCESS;
}
#endif
