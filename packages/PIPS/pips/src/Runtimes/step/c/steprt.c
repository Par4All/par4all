#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#define INLINE static inline
#include "steprt.h"
#include "step_private.h"
#include "communications.h"
#include "regions.h"
#include "trace.h"

/*##############################################################################

  Step_internals

##############################################################################*/

Step_internals steprt_params = {0, 0, NULL};

Array steprt_userArrayTable = {NULL, 0, 0, 0};
Array steprt_worksharingTable = {NULL, 0, 0, 0};

static void uptodate_regions_reset_for_communications(Descriptor_userArray *desc_array);
static void steprt_print_worksharing(Descriptor_worksharing *desc_worksharing);

void __attribute__((unused)) steprt_print(void)
{
  int i;
  static int id;
  printf("Print#%d : NB_NODES %u rank %u language %u parallel_level %u initialized %u\n", id++, NB_NODES, MYRANK, LANGUAGE_ORDER, steprt_params.parallel_level, steprt_params.initialized);
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
  STEP_DEBUG({printf("?/?) <%s>: begin language = %d\n", __func__, language);});

  if(steprt_params.initialized == 0)
    {
      array_set(&steprt_userArrayTable, Descriptor_userArray);
      array_set(&steprt_worksharingTable, Descriptor_worksharing);
      steprt_params.current_worksharing = NULL;
      steprt_params.parallel_level = 0;
      steprt_params.initialized = 1;

      communications_init(language);
    }

  STEP_DEBUG({printf("%d/%d) <%s/>: end\n", MYRANK, NB_NODES, __func__);});
  return;
}

/*##############################################################################

  Descriptor_userArray

##############################################################################*/
Descriptor_userArray *steprt_find_in_userArrayTable(void *userArray)
{
  size_t i;
  Descriptor_userArray *desc_userArray;
  bool stop = false;

  IN_TRACE("begin : userArray = %p", userArray);
  assert(array_ready_p(&steprt_userArrayTable));
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

  IN_TRACE("begin : desc_shared = %p, desc_userArray =%p", desc_shared, desc_userArray);

  nbdims = rg_get_userArrayDims(&(desc_userArray->boundsRegions));
  desc_shared->userArray = desc_userArray->userArray;
  rg_composedRegion_set(&(desc_shared->receiveRegions), nbdims);
  rg_composedRegion_set(&(desc_shared->sendRegions), nbdims);
  desc_shared->interlaced_p = false;
  array_set(&(desc_shared->pending_alltoall), Alltoall_descriptor);

  OUT_TRACE("end");
}


Descriptor_shared *steprt_find_in_sharedTable(void *userArray)
{
  size_t i;
  bool stop = false;
  Descriptor_shared *desc;
  Array *sharedTable;

  IN_TRACE("begin : userArray = %p", userArray);

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

  IN_TRACE("begin : userArray = %p, nb_workchunks = %d, receiveBounds = %p, sendBounds = %p, is_interlaced = %d", userArray, nb_workchunks, receiveBounds, sendBounds, is_interlaced);

  desc_userArray = steprt_find_in_userArrayTable(userArray);
  if(!desc_userArray)
    {
      STEP_COMMUNICATIONS_VERBOSE(printf("uninitialized array %p. Skip %s\n", userArray, sendBounds?"SET_SEND_REGIONS":"SET_RECV_REGIONS"););
    }
  else
    {
      desc_shared = steprt_find_in_sharedTable(userArray);
      assert(desc_userArray);

      TRACE_P("desc_shared = %p", desc_shared);
      STEP_DEBUG({steprt_print_worksharing(CURRENTWORKSHARING);});

      TRACE_P("NB_WORKCHUNKS = %d nb_workchunks = %d\n",  NB_WORKCHUNKS, nb_workchunks);

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
    }
  OUT_TRACE("end");
}

/*##############################################################################

  Descriptor_worksharing

##############################################################################*/
static char* steprt_worksharing_txt[undefined_work+1]={"parallel_work", "do_work", "master_work", "UNKNOWN_work"};

static void steprt_print_worksharing(Descriptor_worksharing *desc_worksharing)
{
  printf("Worksharing %s (%p)\n", desc_worksharing?steprt_worksharing_txt[desc_worksharing->type]:"", desc_worksharing);
}

static Descriptor_worksharing *steprt_worksharing_current(void)
{
  Descriptor_worksharing * desc_worksharing;
  assert(array_ready_p(&steprt_worksharingTable));

  if (steprt_worksharingTable.len == 0)
    desc_worksharing = NULL;
  else
    desc_worksharing = &(array_get_data_from_index(&steprt_worksharingTable, Descriptor_worksharing, steprt_worksharingTable.len -1));

  return desc_worksharing;
}

/* move a worksharing sharedArray into a new one */
void copy_sharedArray(Descriptor_worksharing* _old, Descriptor_worksharing* _new)
{
  /* make sure the elt_size wasn't overriden */
  array_set(&_new->sharedArray, Descriptor_shared);

  /* add the item to the upper worksharing */
  array_append_vals(&(_new->sharedArray), _old->sharedArray.data, _old->sharedArray.len);

  /* free the old worksharing array */
  array_unset(&(_old->sharedArray));
}

void steprt_worksharing_set(worksharing_type type)
{
  Descriptor_worksharing desc_worksharing;
  uint32_t nbdims = 1;

  IN_TRACE("begin : type : %s", steprt_worksharing_txt[type]);

  assert(array_ready_p(&steprt_worksharingTable));

  if(type==parallel_work)
    steprt_params.parallel_level++;

  desc_worksharing.type = type;
  rg_composedRegion_set(&(desc_worksharing.workchunkRegions), nbdims);
  array_set(&(desc_worksharing.scheduleArray), uint32_t);
  array_set(&(desc_worksharing.sharedArray), Descriptor_shared);
  array_set(&(desc_worksharing.privateArray), void *);
  array_set(&(desc_worksharing.reductionsArray), Descriptor_reduction);
  array_set(&(desc_worksharing.communicationsArray), MPI_Request);

  if(CURRENTWORKSHARING) {
    copy_sharedArray(CURRENTWORKSHARING, &desc_worksharing);
  }

  array_append_vals(&steprt_worksharingTable, &desc_worksharing, 1);
  CURRENTWORKSHARING = steprt_worksharing_current();
  OUT_TRACE("end");
}

void steprt_worksharing_unset(void)
{
  size_t i;

  IN_TRACE("begin");
  assert(CURRENTWORKSHARING != NULL);

  if(CURRENTWORKSHARING->type==parallel_work)
    steprt_params.parallel_level--;

  Descriptor_worksharing *upper_worksharing = NULL;

  if(steprt_worksharingTable.len - 1 >0) {
    /* there's an upper worksharing */
    upper_worksharing = &(array_get_data_from_index(&steprt_worksharingTable,
						    Descriptor_worksharing,
						    (steprt_worksharingTable.len-2) ));
    copy_sharedArray(CURRENTWORKSHARING, upper_worksharing);

  } else {
    for(i=0; i<CURRENTWORKSHARING->sharedArray.len; i++) {
      /* remove reference to item in the current worksharing */
      steprt_shared_unset(&(array_get_data_from_index(&(CURRENTWORKSHARING->sharedArray), Descriptor_shared, i)));
    }
  }

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

  IN_TRACE("begin : userArray = %p, type = %d, nbdims = %d, bounds = %p", userArray, type, nbdims, bounds);

  STEP_DEBUG({rg_simpleRegion_print(nbdims, bounds);});

  if (userArray == NULL)
    {

      fprintf(stderr, "ERROR: userArray is not allocated. This error might occur when allocation is done by master only.\n");

      exit(EXIT_FAILURE);
    }
 
  desc_userArray = steprt_find_in_userArrayTable(userArray);

  TRACE_P("desc_userArray = %p", desc_userArray);
  if (!desc_userArray)
    {
      Descriptor_userArray new_desc;

      array_append_vals(&steprt_userArrayTable, &new_desc, 1);
      desc_userArray = &(array_get_data_from_index(&steprt_userArrayTable, Descriptor_userArray, steprt_userArrayTable.len -1));
      communications_set_userArrayDescriptor(desc_userArray, userArray, type, nbdims, bounds);

      /*
	UPTODATE(id_node) = ALL_ARRAY
	INTERLACED(id_node) = EMPTY
      */
      uptodate_regions_reset_for_communications(desc_userArray);
    }
  steprt_set_sharedTable(userArray, 0, NULL, NULL, false);

  STEP_DEBUG({
      Descriptor_userArray *desc_userArray = steprt_find_in_userArrayTable(userArray);

      assert(desc_userArray);
      rg_composedRegion_print(&(desc_userArray->boundsRegions));
    });

  OUT_TRACE("end");
}

Descriptor_reduction *steprt_find_reduction(void *scalar)
{
  size_t i;
  Descriptor_reduction *desc;
  bool stop = false;

  IN_TRACE("begin : scalar = %p", scalar);

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

  IN_TRACE("begin : desc_userArray = %p", desc_userArray);

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

  IN_TRACE("begin : local = %p, toSend = %p", local, toSend);

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

  IN_TRACE("begine : desc_userArray = %p, toSend = %p", desc_userArray, toSend);

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
static void steprt_communications_regions_interlaced_recompute(Descriptor_userArray *desc_userArray)
{
  composedRegion *userArrayBounds;
  Array *uptodateArray;
  Array *interlacedArray;
  uint32_t id_node;

  IN_TRACE("begin : desc_userArray = %p", desc_userArray);

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
static void steprt_communications_regions_partial_uptodate_recompute(Descriptor_userArray *desc_userArray, Array *toReceive)
{
  composedRegion *userArrayBounds;
  Array *uptodateArray;
  uint32_t id_node;

  IN_TRACE("begin : desc_userArray = %p, toReceive = %p", desc_userArray, toReceive);

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
static void steprt_communications_schedule_regions(Descriptor_userArray *desc_userArray, composedRegion *workchunksRegions, Array *regionsToExchange)
{
  uint32_t id_node, nbdims;
  composedRegion emptyRegions;

  IN_TRACE("begin : desc_userArray = %p, workchunksRegions = %p, regionsToExchange = %p", desc_userArray, workchunksRegions, regionsToExchange);

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
static void steprt_communications_allToAll_local(Descriptor_userArray *desc_userArray,
					  Descriptor_shared *desc_shared)
{
  Array regionSend;
  uint32_t id_node;

  IN_TRACE("begin : desc_userArray = %p, desc_shared = %p", desc_userArray, desc_shared);
  assert(desc_userArray);
  assert(desc_shared);

  if(rg_get_nb_simpleRegions(&(desc_shared->sendRegions)) == 0 ) /* sans region SEND, il n'y a rien a faire */
    return;

  steprt_communications_schedule_regions(desc_userArray, &(desc_shared->sendRegions), &regionSend);

  STEP_DEBUG({ printf("COMPUTE LOCAL UPTODATE\n"); });

  /*
    UPTODATE(id_node) = UPTODATE(id_node) union SEND(id_node)
    INTERLACED(id_node) = INTERLACED(id_node) union SEND(id_node)
    UPTODATE(id_node) = UPTODATE(id_node) minus { union(n != id_node) SEND(n) }

    INTERLACED(id_node) = INTERLACED(id_node) minus UPTODATE(id_node)
  */
  local_uptodate_recompute_for_communications(desc_userArray, &regionSend);
  steprt_communications_regions_interlaced_recompute(desc_userArray);

  for (id_node=0; id_node<NB_NODES; id_node++)
    rg_composedRegion_unset(&(array_get_data_from_index(&regionSend, composedRegion, id_node)));
  array_unset(&regionSend);

  OUT_TRACE("end");
}

/*
  Communication from_node -> to_node

  regions_toExchanged =  [regions_localSender(from_node) inter regions_neededReceiver(to_node) ] \ regions_localReceiver(to_node)
*/
static void steprt_communications_compute_region(Descriptor_userArray *desc_userArray, Array *neededReceiver,
					  uint32_t from_node, uint32_t to_node, bool interlaced_p,
					  composedRegion *toExchange)
{
  Array *localSender, *localReceiver;
  composedRegion *regions_localSender, *regions_localReceiver;
  composedRegion *regions_neededReceiver;
  composedRegion *allArrayBounds;
  uint32_t nbdims;

  IN_TRACE("begin : desc_userArray = %p, neededReceiver = %p, from_node = %d, to_node = %d, interlaced_p = %d, toExchange = %p", desc_userArray, neededReceiver, from_node, to_node, interlaced_p, toExchange);

  localSender = interlaced_p?&(desc_userArray->interlacedArray):&(desc_userArray->uptodateArray);
  localReceiver = &(desc_userArray->uptodateArray);
  regions_localSender = &(array_get_data_from_index(localSender, composedRegion, from_node));
  regions_neededReceiver = neededReceiver ? &(array_get_data_from_index(neededReceiver, composedRegion, to_node)) : NULL;
  regions_localReceiver = &(array_get_data_from_index(localReceiver, composedRegion, to_node));
  allArrayBounds = &(desc_userArray->boundsRegions);
  nbdims = rg_get_userArrayDims(allArrayBounds);

  rg_composedRegion_set(toExchange, nbdims);

  if(from_node != to_node)
    {
      rg_composedRegion_union(toExchange, regions_localSender);

      if (regions_neededReceiver != NULL) // partial_communication case
	rg_composedRegion_intersection(toExchange, regions_neededReceiver);

      if (!rg_composedRegion_empty_p(toExchange))
	rg_composedRegion_difference(toExchange, regions_localReceiver);

      rg_composedRegion_simplify(toExchange, &(desc_userArray->boundsRegions));
    }

  OUT_TRACE("End");
}

/*
  For the given node 'local_node', compute regions to send and receive with other nodes
  Communication local_node -> id_node
  toSend(id_node) =  [local_sender(local_node) inter needed_receiver(id_node) ] \ local_receiver(id_node)

  Communication id_node -> local_node
  toRecv(id_node) =  [local_sender(id_node) inter needed_receiver(local_node) ] \ local_receiver(local_node)

  if neededReceiver is NULL a full communication would be performed
  else a partial communication would be perfomed according the neededReceiver regions
*/
static bool steprt_communications_regions_exchanged_set(Descriptor_userArray *desc_userArray, bool interlaced_p, Array *neededReceiver, Array *toSend, Array* toReceive)
{
  uint32_t id_node;
  composedRegion regions_toSend[NB_NODES];
  composedRegion regions_toReceive[NB_NODES];
  bool somethink_p = false;

  IN_TRACE("begin : local_node = %d, desc_userArray = %p, interlaced_p = %d, neededReceiver = %p, toSend = %p, toReceive = %p", MYRANK, desc_userArray, interlaced_p, neededReceiver, toSend, toReceive);

  array_set(toSend, composedRegion);
  array_set(toReceive, composedRegion);
  for (id_node=0; id_node<NB_NODES; id_node++)
    {
      steprt_communications_compute_region(desc_userArray, neededReceiver, MYRANK, id_node, interlaced_p, &(regions_toSend[id_node]));
      steprt_communications_compute_region(desc_userArray, neededReceiver, id_node, MYRANK, interlaced_p, &(regions_toReceive[id_node]));

      if (!(rg_composedRegion_empty_p(&(regions_toSend[id_node])) &&
	    rg_composedRegion_empty_p(&(regions_toReceive[id_node]))))
	somethink_p |= true;
    }
  array_reset(toSend, regions_toSend, NB_NODES);
  array_reset(toReceive, regions_toReceive, NB_NODES);

  OUT_TRACE("end");
  return somethink_p;
}

static void steprt_communications_regions_exchanged_unset(Array *toSend, Array* toReceive)
{
  uint32_t id_node;

  IN_TRACE("begin : toSend = %p, toReceive = %p", toSend, toReceive);

  for (id_node = 0; id_node < NB_NODES; id_node++)
    {
      rg_composedRegion_unset(&(array_get_data_from_index(toSend, composedRegion, id_node)));
      rg_composedRegion_unset(&(array_get_data_from_index(toReceive, composedRegion, id_node)));
    }
  array_unset(toSend);
  array_unset(toReceive);
  OUT_TRACE("End");
}


static void steprt_communications_allToAll_unlocal(Descriptor_userArray *desc_userArray,
					    Descriptor_shared *desc_shared,
					    uint32_t algorithm, int_MPI tag, bool is_full)
{
  Array regionReceive;
  Array *pRegionReceive=&regionReceive;
  uint32_t id_node;
  bool is_interlaced, something_p;
  Array toSend, toReceive;

  IN_TRACE("begin : desc_userArray = %p, desc_shared = %p, algorithm = %d, tag = %d", desc_userArray, desc_shared, algorithm, tag);
  assert(desc_userArray);
  assert(desc_shared);

  if (!is_full && rg_get_nb_simpleRegions(&(desc_shared->receiveRegions)) == 0) /* si on est en comm partielle et qu'il n'y a pas de region RECV, il n'y pas rien a communiquer */
    return;

  if(is_full)
    {
      rg_composedRegion_reset(&(desc_shared->receiveRegions), NULL, 0); // no more nedded.
      pRegionReceive = NULL;
    }
  else
    steprt_communications_schedule_regions(desc_userArray, &(desc_shared->receiveRegions), &regionReceive);

  STEP_DEBUG({printf("COMPUTE COMM(Pi->Pj)\n"); });
  /*
     if (is_full) // pRegionReceive = NULL
        toSend(id_node) = [UPTODATE(local_node)] \ UPTODATE(id_node)
	toReceive(id_node) = [UPTODATE(id_node)] \ UPTODATE(local_node)
     else
        toSend(id_node) = [UPTODATE(local_node) inter RECV(id_node) ] \ UPTODATE(id_node)
        toReceive(id_node) = [UPTODATE(id_node) inter RECV(local_node) ] \ UPTODATE(local_node)
   */
  is_interlaced = false;
  something_p = steprt_communications_regions_exchanged_set(desc_userArray, is_interlaced, pRegionReceive, &toSend, &toReceive);
  STEP_COMMUNICATIONS_VERBOSE(printf("%u) COMMUNICATION %s %s (%p%s)\n", MYRANK,  is_full?"FULL":"PARTIAL", something_p?"DETAIL":"EMPTY", desc_userArray->userArray, is_interlaced?" interlaced":""););

  /* Communication */
  if (something_p)
    communications_allToAll(desc_userArray, &toSend, &toReceive, is_interlaced, algorithm, tag, &(CURRENTWORKSHARING->communicationsArray));

  steprt_communications_regions_exchanged_unset(&toSend, &toReceive);

  STEP_DEBUG({ printf("COMPUTE COMM(Pi->Pj) INTERLACED\n"); });
  /*
    if (is_full) // pRegionReceive = NULL
        toSendInterlaced(id_node) = INTERLACED(local_node)\ UPTODATE(id_node)
	toReceiveInterlaced(id_node) = INTERLACED(id_node) \ UPTODATE(local_node)
     else
        toSendInterlaced(id_node) = [INTERLACED(local_node) inter RECV(id_node) ] \ UPTODATE(id_node)
	toReceiveInterlaced(id_node) = [INTERLACED(id_node) inter RECV(local_node) ] \ UPTODATE(local_node)
   */
  is_interlaced = true;
  something_p = steprt_communications_regions_exchanged_set(desc_userArray, is_interlaced, pRegionReceive, &toSend, &toReceive);
  STEP_COMMUNICATIONS_VERBOSE(printf("%u) COMMUNICATION %s %s (%p%s)\n", MYRANK,  is_full?"FULL":"PARTIAL", something_p?"DETAIL":"EMPTY", desc_userArray->userArray, is_interlaced?" interlaced":""););

  /* Communication */
  if (something_p)
    communications_allToAll(desc_userArray, &toSend, &toReceive, is_interlaced, algorithm, tag, &(CURRENTWORKSHARING->communicationsArray));

  steprt_communications_regions_exchanged_unset(&toSend, &toReceive);

  STEP_DEBUG({printf("COMPUTE UPTODATE AFTER COMM(Pi->Pj)\n"); });
  if(is_full)
    {
      /*
	UPTODATE(id_node) = ALL_ARRAY
	INTERLACED(id_node) = EMPTY
      */
      uptodate_regions_reset_for_communications(desc_userArray);
    }
  else
    {
      /*
	UPTODATE(id_node) = UPTODATE(id_node) union RECV(id_node)
	INTERLACED(id_node) = INTERLACED(id_node) minus UPTODATE(id_node)
      */
      steprt_communications_regions_partial_uptodate_recompute(desc_userArray, &regionReceive);
      steprt_communications_regions_interlaced_recompute(desc_userArray);

      for (id_node=0; id_node<NB_NODES; id_node++)
	rg_composedRegion_unset(&(array_get_data_from_index(&regionReceive, composedRegion, id_node)));
      array_unset(&regionReceive);
    }
  OUT_TRACE("end");
}

static void steprt_alltoall_generic(Descriptor_shared *desc_shared,
				    Descriptor_userArray* desc_userArray,
				    bool full_p,
				    uint32_t algorithm,
				    int_MPI tag)
{
  IN_TRACE("begin : desc_shared = %p, desc_userArray = %p, full_p = %d, algorithm = %d, tag = %d", desc_shared, desc_userArray, full_p, algorithm, tag);

  steprt_communications_allToAll_local(desc_userArray, desc_shared);

  steprt_communications_allToAll_unlocal(desc_userArray, desc_shared, algorithm, tag, full_p);

  OUT_TRACE("end");
}

void steprt_run_registered_alltoall(Descriptor_worksharing *worksharing)
{
  int i;
  int nb_sharedArray = worksharing->sharedArray.len;

  IN_TRACE("begin");
  for(i=0; i<nb_sharedArray; i++) {
    Descriptor_shared *desc_shared;
    desc_shared = array_sized_index(&(worksharing->sharedArray), i);

    while(desc_shared->pending_alltoall.len) {
      Alltoall_descriptor *cur_alltoall;
      cur_alltoall = array_sized_index(&(desc_shared->pending_alltoall), 0);
      assert(cur_alltoall);
      steprt_alltoall_generic(cur_alltoall->desc_shared,
			      cur_alltoall->desc_userArray,
			      cur_alltoall->full_p,
			      cur_alltoall->algorithm,
			      cur_alltoall->tag);

      /* request processed, remove it from the list */
      array_remove_index_fast(&(desc_shared->pending_alltoall), 0);
    }
  }
  OUT_TRACE("end");
}

void steprt_register_alltoall(void * userArray, bool full_p, uint32_t algorithm, int_MPI tag)
{
  Descriptor_shared *desc_shared;
  Descriptor_userArray *desc_userArray;
  Alltoall_descriptor new_item;

  IN_TRACE("begin : userArray = %p, full_p = %d, algorithm = %d, tag = %d", userArray, full_p, algorithm, tag);

  desc_shared = steprt_find_in_sharedTable(userArray);
  desc_userArray = steprt_find_in_userArrayTable(userArray);
  if(!desc_userArray)
    {
      STEP_COMMUNICATIONS_VERBOSE(printf("uninitialized array %p. Skip register_alltoall\n", userArray););
    }
  else
    {
      new_item.desc_shared = desc_shared;
      new_item.desc_userArray = desc_userArray;
      new_item.full_p = full_p;
      new_item.algorithm = algorithm;
      new_item.tag = tag;

      /* add the descriptor to the list */
      array_append_vals(&(desc_shared->pending_alltoall), &new_item, 1);
    }
  OUT_TRACE("end");
}

void steprt_alltoall(void * userArray, bool full_p, uint32_t algorithm, int_MPI tag)
{
  Descriptor_shared *desc_shared;
  Descriptor_userArray *desc_userArray;

  IN_TRACE("begin : userArray = %p, full_p = %d, algorithm = %d, tag = %d", userArray, full_p, algorithm, tag);

  desc_shared = steprt_find_in_sharedTable(userArray);
  desc_userArray = steprt_find_in_userArrayTable(userArray);
  if(!desc_userArray)
    {
      STEP_COMMUNICATIONS_VERBOSE(printf("uninitialized array %p. Skip alltoall\n", userArray););
    }
  else
    {
      TRACE_P("userArray nbdims = %d\n", rg_get_userArrayDims(&(desc_userArray->boundsRegions)));
      TRACE_P("shared sendRegions nbdims = %d\n", rg_get_userArrayDims(&(desc_shared->sendRegions)));

      steprt_alltoall_generic(desc_shared, desc_userArray, full_p, algorithm, tag);
    }
  OUT_TRACE("end");
}

void steprt_alltoall_all(bool full_p, uint32_t algorithm, int_MPI tag)
{
  int i;
  for(i=0; i<CURRENTWORKSHARING->sharedArray.len; i++) {
    Descriptor_shared* new_desc = &(array_get_data_from_index(&CURRENTWORKSHARING->sharedArray,
							      Descriptor_shared,
							      i));
    steprt_alltoall(new_desc->userArray, full_p, algorithm, tag);
  }
}

void steprt_finalize()
{
  size_t i;
  IN_TRACE("begin");

  communications_finalize();

  steprt_params.parallel_level = 0;
  steprt_params.initialized = 0;

  for (i=0; i<steprt_userArrayTable.len; i++)
    communications_unset_userArrayDescriptor(&(array_get_data_from_index(&steprt_userArrayTable, Descriptor_userArray, i)));
  array_unset(&steprt_userArrayTable);

  for (i=0; i<steprt_worksharingTable.len; i++)
    steprt_worksharing_unset();
  array_unset(&steprt_worksharingTable);
  OUT_TRACE("end");
}

void steprt_initreduction(void *variable, uint32_t op, uint32_t type)
{
  Descriptor_reduction desc_reduction;
  IN_TRACE("begin : variable = %p, op = %d, type = %d", variable, op, type);

  desc_reduction.variable = variable;
  desc_reduction.type = type;
  desc_reduction.operator = op;

  Descriptor_userArray *desc_userArray = steprt_find_in_userArrayTable(variable);
  if (desc_userArray == NULL) /* scalar reduction */
    communications_initreduction(&desc_reduction, desc_userArray);
  /* for array reduction : communications_initreduction called from STEP_SET_REDUCTION_SENDREGIONS */

  array_append_vals(&(CURRENTWORKSHARING->reductionsArray), &desc_reduction, 1);
  OUT_TRACE("end");
}


/*
  For array reduction, the SEND_REGIONS are ignored : the reduction is performed on the whole array
*/
void steprt_set_reduction_sendregions(Descriptor_reduction *desc_reduction, uint32_t nb_workchunks, STEP_ARG *regions)
{
  void *userArray;
  Descriptor_userArray *desc_userArray;

  IN_TRACE("begin : desc_reduction = %p, nb_workchunks = %d, regions =%p", desc_reduction, nb_workchunks, regions);

  userArray = desc_reduction->variable;

  /* enregistremant des regions SEND + sauvegarde des valeurs initiales */
  steprt_set_sharedTable(userArray, nb_workchunks, NULL, regions, true);

  /* initialisation à l'element neutre */
  desc_userArray = steprt_find_in_userArrayTable(userArray);
  communications_initreduction(desc_reduction, desc_userArray);
  OUT_TRACE("end");
}

void steprt_reduction(void *variable)
{
  Descriptor_reduction *desc_reduction;
  Descriptor_userArray *desc_userArray;
  IN_TRACE("begin : variable = %p", variable);

  desc_userArray = steprt_find_in_userArrayTable(variable);
  desc_reduction = steprt_find_reduction(variable);
  communications_reduction(desc_reduction, desc_userArray);

  if (desc_userArray != NULL)
    {
      /*
	UPTODATE(id_node) = ALL_ARRAY
	INTERLACED(id_node) = EMPTY
      */
      uptodate_regions_reset_for_communications(desc_userArray);
    }
  OUT_TRACE("end");
}
