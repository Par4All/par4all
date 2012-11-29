#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <float.h>

#define INLINE static inline
#include "communications.h"
#include "regions.h"
#include "step_private.h"
#include "trace.h"
#include "step_common.h"

#ifndef  MPI_REAL16
#define  MPI_REAL16      MPI_LONG_DOUBLE
#endif

uint32_t communications_NB_NODES = 0;
uint32_t communications_MY_RANK = 0;
uint32_t communications_LANGUAGE_ORDER = -1;


/*##############################################################################

  Descriptor_userArray

##############################################################################*/
Descriptor_userArray *communications_set_userArrayDescriptor(Descriptor_userArray *desc_userArray, void *userArray, uint32_t type, uint nbdims, INDEX_TYPE *bounds)
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

  OUT_TRACE("desc_userArray = %p", desc_userArray);
  return desc_userArray;
}

void communications_unset_userArrayDescriptor(Descriptor_userArray *desc)
{
  uint32_t id_node;
  Array *uptodateArray = &(desc->uptodateArray);
  Array *interlacedArray = &(desc->interlacedArray);

  rg_composedRegion_unset(&(desc->boundsRegions));

  if (desc->savedUserArray)
    free(desc->savedUserArray);

  assert(uptodateArray->len == interlacedArray->len);
  for (id_node=0; id_node< uptodateArray->len; id_node++)
    {
      rg_composedRegion_unset(&(array_get_data_from_index(uptodateArray, composedRegion, id_node)));
      rg_composedRegion_unset(&(array_get_data_from_index(interlacedArray, composedRegion, id_node)));
    }
  array_unset(uptodateArray);
  array_unset(interlacedArray);

}

/*##############################################################################

  Data type

##############################################################################*/
typedef struct
{
  uint32_t type_id;
  int32_t type_size;
  MPI_Datatype type_mpi;
}Descriptor_type;

/*
  Default standard sizes are unknown at compile time because Fortran
  allows to define the size of an integer, for instance, at
  compilation time with option -i4 (with Intel compiler).

  This is the reason why sizes equal 0 for STEP_INTEGER, STEP_REAL,
  STEP_DOUBLE_PRECISION, STEP_COMPLEX. They will be initialized with
  communications_type_init().

 */

Descriptor_type step_types_table[STEP_TYPE_UNDEFINED+1] ={
  {STEP_INTEGER,           0, MPI_INTEGER},          /* INTEGER */
  {STEP_REAL,              0, MPI_REAL},             /* REAL */
  {STEP_DOUBLE_PRECISION,  0, MPI_DOUBLE_PRECISION}, /* DOUBLE PRECISION */
  {STEP_COMPLEX,           0, MPI_COMPLEX},          /* COMPLEX */

  {STEP_INTEGER1,          1, MPI_INTEGER1},         /* INTEGER*1 */
  {STEP_INTEGER2,          2, MPI_INTEGER2},         /* INTEGER*2 */
  {STEP_INTEGER4,          4, MPI_INTEGER4},         /* INTEGER*4 */
  {STEP_INTEGER8,          8, MPI_INTEGER8},         /* INTEGER*8 */
  {STEP_REAL4,             4, MPI_REAL4},            /* REAL*4 */
  {STEP_REAL8,             8, MPI_REAL8},            /* REAL*8 */
  {STEP_REAL16,           16, MPI_REAL16},           /* REAL*16 */
  {STEP_COMPLEX8,          8, MPI_COMPLEX8},         /* COMPLEX*8 */
  {STEP_COMPLEX16,        16, MPI_COMPLEX16},        /* COMPLEX*16 */

  {STEP_TYPE_UNDEFINED,    0, MPI_DATATYPE_NULL}     /* Must be the last */
};

static char *communication_type_name(uint32_t operator)
{
  switch(operator)
    {
    case STEP_INTEGER: return "STEP_INTEGER";
    case STEP_REAL : return "STEP_REAL";
    case STEP_DOUBLE_PRECISION: return "STEP_DOUBLE_PRECISION";
    case STEP_COMPLEX: return "STEP_COMPLEX";
    case STEP_INTEGER1: return "STEP_INTEGER1";
    case STEP_INTEGER2: return "STEP_INTEGER2";
    case STEP_INTEGER4: return "STEP_INTEGER4";
    case STEP_INTEGER8: return "STEP_INTEGER8";
    case STEP_REAL4: return "STEP_REAL4";
    case STEP_REAL8: return "STEP_REAL8";
    case STEP_REAL16: return "STEP_REAL16";
    case STEP_COMPLEX8: return "STEP_COMPLEX8";
    case STEP_COMPLEX16: return "STEP_COMPLEX16";
    case STEP_TYPE_UNDEFINED: return "STEP_TYPE_UNDEFINED";
    default: return "?";
    }
}

static void communications_type_init(void)
{
  Descriptor_type descriptor;
  int32_t step_type;
  uint32_t id_type;

  IN_TRACE("Begin");

  step_type = STEP_INTEGER;
  step_sizetype_(&step_type, &(step_types_table[step_type].type_size));
  switch(step_types_table[step_type].type_size)
    {
    case 1: step_types_table[step_type].type_mpi = MPI_INTEGER1; break;
    case 2: step_types_table[step_type].type_mpi = MPI_INTEGER2; break;
    case 4: step_types_table[step_type].type_mpi = MPI_INTEGER4; break;
    case 8: step_types_table[step_type].type_mpi = MPI_INTEGER8; break;
    default: assert(0);
    }

  step_type = STEP_REAL;
  step_sizetype_(&step_type, &(step_types_table[step_type].type_size));
  switch(step_types_table[step_type].type_size)
    {
    case 4: step_types_table[step_type].type_mpi = MPI_REAL4; break;
    case 8: step_types_table[step_type].type_mpi = MPI_REAL8; break;
    case 16: step_types_table[step_type].type_mpi = MPI_REAL16; break;
    default: assert(0);
    }

  step_type = STEP_DOUBLE_PRECISION;
  step_sizetype_(&step_type, &(step_types_table[step_type].type_size));
  switch(step_types_table[step_type].type_size)
    {
    case 8: step_types_table[step_type].type_mpi = MPI_REAL8; break;
    case 16: step_types_table[step_type].type_mpi = MPI_REAL16; break;
    default: assert(0);
    }

  step_type = STEP_COMPLEX;
  step_sizetype_(&step_type, &(step_types_table[step_type].type_size));
  switch(step_types_table[step_type].type_size)
    {
    case 8: step_types_table[step_type].type_mpi = MPI_COMPLEX8; break;
    case 16: step_types_table[step_type].type_mpi = MPI_COMPLEX16; break;
    default: assert(0);
    }

  for(id_type=0; id_type<STEP_TYPE_UNDEFINED; id_type++)
    {
      descriptor = step_types_table[id_type];
      assert(descriptor.type_id ==  id_type);
      assert(descriptor.type_mpi != MPI_DATATYPE_NULL);
      assert(descriptor.type_size != 0);
    }
  OUT_TRACE("End");
}

static int32_t communications_get_type_size(uint32_t type_id)
{
  int32_t type_size;

  IN_TRACE("type_id = %d", type_id);

  assert(type_id <= STEP_TYPE_UNDEFINED);
  type_size = step_types_table[type_id].type_size;

  OUT_TRACE("type_size = %d", type_size);
  return type_size;
}

static MPI_Datatype communications_get_type_mpi(uint32_t type_id)
{
  MPI_Datatype mpi_type;

  IN_TRACE("type_id = %d", type_id);
  assert(type_id < STEP_TYPE_UNDEFINED);
  mpi_type = step_types_table[type_id].type_mpi;

  OUT_TRACE("mpi_type = %d", mpi_type);
  return mpi_type;
}

/*##############################################################################

  Communications

##############################################################################*/

void communications_init(int language)
{
  int_MPI is_initialized;
  int_MPI is_finalized;

  STEP_DEBUG({printf("?/?) <%s>: begin\n", __func__);});

  assert(MPI_Finalized(&is_finalized) == MPI_SUCCESS);

  if(is_finalized)
    {
      fprintf(stderr, "MPI already finalized\n");
      assert(0);
    }

  assert(MPI_Initialized(&is_initialized) == MPI_SUCCESS);
  if(!is_initialized)
    {
      int_MPI provided;

      assert(MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided) == MPI_SUCCESS);
      if(!(provided == MPI_THREAD_FUNNELED ||
	   provided == MPI_THREAD_MULTIPLE))
	fprintf (stderr, "Warning in communication_init() : FUNNELED thread support not available\n");
    }
  communications_type_init();
  NB_NODES = communications_get_commsize();
  MYRANK = communications_get_rank();
  LANGUAGE_ORDER = language;

  STEP_DEBUG({printf("%d/%d) <%s/>: end\n", MYRANK, NB_NODES, __func__);});
}

void communications_finalize(void)
{
  IN_TRACE("begin");

  assert(MPI_Finalize() == MPI_SUCCESS);
  NB_NODES = 0;
  MYRANK = 0;
  LANGUAGE_ORDER = -1;

  OUT_TRACE("end");
}

uint32_t communications_get_commsize(void)
{
  int_MPI size_mpi = 1;

  assert(MPI_Comm_size(MPI_COMM_WORLD, &size_mpi) == MPI_SUCCESS);
  return (uint32_t)size_mpi;
}

uint32_t communications_get_rank(void)
{
  int_MPI rank_mpi = 0;

  assert(MPI_Comm_rank(MPI_COMM_WORLD, &rank_mpi) == MPI_SUCCESS);
  return (uint32_t)rank_mpi;
}

void *communications_alloc_buffer(Descriptor_userArray *descriptor, size_t *alloc)
{
  uint32_t d, size;
  uint32_t nbdims;
  INDEX_TYPE *bounds;
  void *buffer;
  int_MPI type_size;

  IN_TRACE("descriptor = %p, alloc = %p", descriptor, alloc);
  size = 1;
  nbdims = rg_get_userArrayDims(&(descriptor->boundsRegions));
  bounds = rg_get_simpleRegion(&(descriptor->boundsRegions), 0);

  type_size = (int_MPI)communications_get_type_size(descriptor->type);
  for (d = 0; d < nbdims; d++)
    size *= 1 + bounds[UP(d)] - bounds[LOW(d)];

  buffer = malloc(type_size * size);
  assert(buffer);

  *alloc = type_size * size;

  OUT_TRACE("buffer = %p", buffer);
  return buffer;
}

/*
  Build new MPI type from regions to exchange
*/
static void communication_type_mpi_set(composedRegion *compReg, MPI_Datatype type,
				       INDEX_TYPE *allArraybounds, int_MPI *allArray_sizes,
				       MPI_Datatype *type_mpi)
{
  IN_TRACE("compReg = %p, type = %p, allArraybounds = %p, allArray_sizes = %p, type_mpi = %p", compReg, type, allArraybounds, allArray_sizes, type_mpi);

  if (rg_composedRegion_empty_p(compReg))
    {
      *type_mpi = MPI_DATATYPE_NULL;
    }
  else
    {
      uint32_t nbdims = rg_get_userArrayDims(compReg);
      int_MPI subArray_start[nbdims];
      int_MPI subArray_sizes[nbdims];
      uint32_t id_region, nbRegions = rg_get_nb_simpleRegions(compReg);
      int_MPI array_of_blocklengths[nbRegions];
      MPI_Aint array_of_displacements[nbRegions];
      MPI_Datatype regions_type[nbRegions];
      int_MPI order;


      switch (LANGUAGE_ORDER)
	{
	case STEP_FORTRAN:
	  order = MPI_ORDER_FORTRAN;
	  break;
	case STEP_C:
	  order = MPI_ORDER_C;
	  break;
	default: assert(0);
	}

      for (id_region=0; id_region<nbRegions; id_region++)
	{
	  INDEX_TYPE *bounds_r = rg_get_simpleRegion(compReg, id_region);

	  array_of_blocklengths[id_region] = 1;
	  array_of_displacements[id_region] = 0;
	  BOUNDS_2_START(nbdims, allArraybounds, bounds_r, int_MPI, subArray_start);
	  BOUNDS_2_SIZES(nbdims, allArraybounds, bounds_r, int_MPI, subArray_sizes);
	  MPI_Type_create_subarray((int_MPI)nbdims, allArray_sizes, subArray_sizes, subArray_start,
				   order, type, &(regions_type[id_region]));
	}
      MPI_Type_create_struct((int_MPI)nbRegions, array_of_blocklengths, array_of_displacements,
			     regions_type, type_mpi);
      MPI_Type_commit(type_mpi);
    }

  OUT_TRACE("end");
  return ;
}

static void communications_types_mpi_set(Descriptor_userArray *desc_userArray, Array *toExchange_regions, Array *toExchange_mpi)
{
  uint32_t id_node;
  INDEX_TYPE *userArrayBounds;
  MPI_Datatype type;
  composedRegion *bounds = &(desc_userArray->boundsRegions);
  uint32_t nbdims = rg_get_userArrayDims(bounds);
  int_MPI allArray_sizes[nbdims];
  MPI_Datatype types_mpi[NB_NODES];

  IN_TRACE("desc_userArray = %p, toExchange_regions = %p, toExchange_mpi = %p", desc_userArray, toExchange_regions, toExchange_mpi);


  userArrayBounds = rg_get_simpleRegion(bounds,0);
  type = communications_get_type_mpi(desc_userArray->type);
  array_set(toExchange_mpi, MPI_Datatype);

  BOUNDS_2_SIZES(nbdims, userArrayBounds , userArrayBounds, int_MPI, allArray_sizes);
  for (id_node=0; id_node<NB_NODES; id_node++)
    {
      composedRegion *r = &(array_get_data_from_index(toExchange_regions, composedRegion, id_node));

      communication_type_mpi_set(r, type, userArrayBounds, allArray_sizes, &types_mpi[id_node]);
    }
  array_append_vals(toExchange_mpi, types_mpi, NB_NODES);

  OUT_TRACE("end");
  return ;
}

/*
  Free MPI type
*/
static void communication_type_mpi_unset(MPI_Datatype *type_mpi)
{
  if(*type_mpi != MPI_DATATYPE_NULL)
    MPI_Type_free(type_mpi);
}
static void communications_types_mpi_unset(Array *toExchange_mpi)
{
  uint32_t id_node;

  for (id_node=0; id_node<NB_NODES; id_node++)
    communication_type_mpi_unset(&(array_get_data_from_index(toExchange_mpi, MPI_Datatype, id_node)));

  array_unset(toExchange_mpi);
}


#define DIFF_UPDATE(datatype, saved, new, current, index)	\
  if (((datatype*) new)[index] != ((datatype*) saved)[index])	\
    ((datatype*) current)[index] = ((datatype*) new)[index];

static void communications_diff(Descriptor_userArray *desc_userArray, void *buffer, composedRegion *toReceiveInterlaced)
{
  assert(desc_userArray && buffer && toReceiveInterlaced);

  Descriptor_type d_type = step_types_table[desc_userArray->type];
  composedRegion *bounds = &(desc_userArray->boundsRegions);
  INDEX_TYPE *userArrayBounds = rg_get_simpleRegion(bounds,0);
  uint32_t nbdims = rg_get_userArrayDims(bounds);
  INDEX_TYPE allArray_sizes[nbdims];
  size_t id_region;

  BOUNDS_2_SIZES(nbdims, userArrayBounds , userArrayBounds, INDEX_TYPE, allArray_sizes);

  for (id_region=0; id_region<rg_get_nb_simpleRegions(toReceiveInterlaced); id_region++)
    {
      uint32_t d;
      INDEX_TYPE id_value, nb_value;
      INDEX_TYPE *bounds_r = rg_get_simpleRegion(toReceiveInterlaced, id_region);
      INDEX_TYPE subArray_start[nbdims], subArray_end[nbdims], index[nbdims];

      // initialisation
      BOUNDS_2_START(nbdims, userArrayBounds, bounds_r, INDEX_TYPE, subArray_start);
      BOUNDS_2_END(nbdims, userArrayBounds, bounds_r, INDEX_TYPE, subArray_end);
      memcpy(index, subArray_start, sizeof(INDEX_TYPE) * nbdims);
      nb_value = 1;
      for (d=0; d<nbdims; d++)
	nb_value = nb_value * (1 + subArray_end[d] - subArray_start[d]);

      for (id_value=0; id_value<nb_value; id_value++)
	{
	  // compute offset (COL_MAJOR)
	  INDEX_TYPE offset = 0;
	  for (d=nbdims-1; d==0; d--)
	    offset = index[d] + allArray_sizes[d]*offset;

	  // where diff is performed
	  switch(d_type.type_id)
	    {
	    case STEP_INTEGER: case STEP_INTEGER1: case STEP_INTEGER2: case STEP_INTEGER4: case STEP_INTEGER8:
	      switch(d_type.type_size)
		{
		case 1: DIFF_UPDATE(int8_t, desc_userArray->savedUserArray, buffer, desc_userArray->userArray, offset); break;
		case 2: DIFF_UPDATE(int16_t, desc_userArray->savedUserArray, buffer, desc_userArray->userArray, offset); break;
		case 4: DIFF_UPDATE(int32_t, desc_userArray->savedUserArray, buffer, desc_userArray->userArray, offset); break;
		case 8: DIFF_UPDATE(int64_t, desc_userArray->savedUserArray, buffer, desc_userArray->userArray, offset); break;
		default : assert(0);
		}
	      break;
	    case STEP_REAL: case STEP_REAL4: case STEP_REAL8: case STEP_REAL16:
	      switch(d_type.type_size)
		{
		case 4: DIFF_UPDATE(float, desc_userArray->savedUserArray, buffer, desc_userArray->userArray, offset); break;
		case 8: DIFF_UPDATE(double, desc_userArray->savedUserArray, buffer, desc_userArray->userArray, offset); break;
		case 16: DIFF_UPDATE(long double, desc_userArray->savedUserArray, buffer, desc_userArray->userArray, offset); break;
		default : assert(0);
		}
	      break;
	    case STEP_COMPLEX: case STEP_COMPLEX8: case STEP_COMPLEX16:
	      switch(d_type.type_size)
		{
		case 8: DIFF_UPDATE(float complex, desc_userArray->savedUserArray, buffer, desc_userArray->userArray, offset); break;
		case 16: DIFF_UPDATE(double complex, desc_userArray->savedUserArray, buffer, desc_userArray->userArray, offset); break;
		default: assert(0);
		}
	      break;
	    default:
	      assert(0);
	    }

	  //index increment
	  for (d=0; d<nbdims; )
	    {
	      if (index[d] < subArray_end[d])
		{
		  index[d]++;
		  d= nbdims;
		}
	      else
		{
		  index[d] = subArray_start[d];
		  d++;
		}
	    }
	}
    }
}

static void communications_alltoall_NBlocking_1(Descriptor_userArray *desc_userArray,
						Array *toSend_mpi, Array *toReceive_mpi, Array *toReceiveInterlaced,
						int_MPI tag, uint *nb_requests, MPI_Request *requests)
{
  int_MPI id_node;
  MPI_Datatype mpi_type;

  IN_TRACE("local_node = %d, desc_userArray = %p, toSend_mpi = %p, toReceive_mpi = %p, toReceiveInterlaced = %p, tag = %d, nb_requests = %d, requests = %p", MYRANK, desc_userArray, toSend_mpi, toReceive_mpi, toReceiveInterlaced, tag, nb_requests, requests)
  /*
    SEND
  */
  for (id_node=0; id_node<NB_NODES; id_node++)
    {
      mpi_type = array_get_data_from_index(toSend_mpi, MPI_Datatype, id_node);
      if ((id_node != MYRANK) && (mpi_type != MPI_DATATYPE_NULL))
	MPI_Isend(desc_userArray->userArray, 1, mpi_type, id_node, tag, MPI_COMM_WORLD, &requests[(*nb_requests)++]);
    }

  /*
    RECEIVE
  */
  if (toReceiveInterlaced == NULL)
    for (id_node=0; id_node<NB_NODES; id_node++)
      {
	mpi_type = array_get_data_from_index(toReceive_mpi, MPI_Datatype, id_node);
	if ((id_node != MYRANK) && (mpi_type != MPI_DATATYPE_NULL))
	  MPI_Irecv(desc_userArray->userArray, 1, mpi_type, id_node, tag, MPI_COMM_WORLD, &requests[(*nb_requests)++]);
      }
  else // interlaced regions
    {
      size_t alloc;
      void *buffer = communications_alloc_buffer(desc_userArray, &alloc);

      for (id_node=0; id_node<NB_NODES; id_node++)
	{
	  mpi_type = array_get_data_from_index(toReceive_mpi, MPI_Datatype, id_node);
	  if ((id_node != MYRANK) && (mpi_type != MPI_DATATYPE_NULL))
	    {
	      MPI_Recv(buffer, 1, mpi_type, id_node, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	      communications_diff(desc_userArray, buffer,
				  &(array_get_data_from_index(toReceiveInterlaced, composedRegion, id_node)));
	    }
	}
      free(buffer);
    }

  OUT_TRACE("End");
}

static void communication_display(bool is_interlaced, Descriptor_userArray *desc_userArray, Array *toSend, Array *toReceive)
{
  uint32_t id_node;
  composedRegion *rg;

  for (id_node=0; id_node<NB_NODES; id_node++)
    {
      rg = &(array_get_data_from_index(toSend, composedRegion, id_node));
      if (!rg_composedRegion_empty_p(rg))
	{
	  printf("node %u send to %u (%p%s):", MYRANK, id_node, desc_userArray->userArray, is_interlaced?" interlaced":"");
	  rg_composedRegion_print(rg);
	}
      rg = &(array_get_data_from_index(toReceive, composedRegion, id_node));
      if (!rg_composedRegion_empty_p(rg))
	{
	  printf("node %u receive from %u (%p%s):", MYRANK, id_node, desc_userArray->userArray, is_interlaced?" interlaced":"");
	  rg_composedRegion_print(rg);
	}
    }
}

void communications_allToAll(Descriptor_userArray *desc_userArray,  Array *toSend, Array *toReceive, bool is_interlaced, uint32_t algorithm, int_MPI tag, Array *pending_communications)
{
  Array toSend_mpi, toReceive_mpi;
  IN_TRACE("begin");
  STEP_COMMUNICATIONS_VERBOSE(communication_display(is_interlaced, desc_userArray, toSend, toReceive););

  communications_types_mpi_set(desc_userArray, toSend, &toSend_mpi);
  communications_types_mpi_set(desc_userArray, toReceive, &toReceive_mpi);

  STEP_DEBUG({printf("COMMUNICATION MPI\n"); });
  switch (algorithm)
    {
    case STEP_NBLOCKING_ALG :
      {
	MPI_Request requests[2*NB_NODES]; // SEND RECEIVE
	uint nb_requests = 0;

	communications_alltoall_NBlocking_1(desc_userArray, &toSend_mpi, &toReceive_mpi,
					    is_interlaced?toReceive:NULL, tag, &nb_requests, requests);
	array_append_vals(pending_communications, requests, nb_requests);
      }
      break;
    default : assert(0);
    }
  communications_types_mpi_unset(&toSend_mpi);
  communications_types_mpi_unset(&toReceive_mpi);
  OUT_TRACE("end");
}

void communications_waitall(Array *communicationsArray)
{
  IN_TRACE("begin");
  MPI_Waitall ((int_MPI)communicationsArray->len, (MPI_Request*)communicationsArray->data, MPI_STATUS_IGNORE);
  array_reset(communicationsArray, NULL, 0);
  OUT_TRACE("end");
}

void communications_barrier(void)
{
  int_MPI is_initialized;
  assert(MPI_Initialized(&is_initialized) == MPI_SUCCESS);

  MPI_Barrier(MPI_COMM_WORLD);
}

void communications_oneToAll_Scalar(void *scalar, uint32_t type, uint32_t algorithm)
{
  switch (algorithm)
    {
    default:
      MPI_Bcast (scalar, 1, communications_get_type_mpi(type), 0, MPI_COMM_WORLD);
    }
}

void communications_oneToAll_Array(Descriptor_userArray *desc_userArray, uint32_t algorithm)
{
  assert(desc_userArray);
  MPI_Datatype type_mpi, type = communications_get_type_mpi(desc_userArray->type);
  INDEX_TYPE *userArrayBounds = rg_get_simpleRegion(&(desc_userArray->boundsRegions), 0);
  uint32_t nbdims = rg_get_userArrayDims(&(desc_userArray->boundsRegions));
  int_MPI allArray_sizes[nbdims];

  BOUNDS_2_SIZES(nbdims, userArrayBounds , userArrayBounds, int_MPI, allArray_sizes);
  communication_type_mpi_set(&(desc_userArray->boundsRegions), type, userArrayBounds, allArray_sizes, &type_mpi);

  switch (algorithm)
    {
    default:
      MPI_Bcast (desc_userArray->userArray, 1, type_mpi, 0, MPI_COMM_WORLD);
    }

  communication_type_mpi_unset(&type_mpi);
}


#define REDUCTIONS_INIT_NEUTRAL_ELT(c_type, neutral_elt, nb_element)	\
  {									\
    uint32_t h;								\
    for(h=0; h<nb_element; h++)						\
      ((c_type *)(reduction->variable))[h]=neutral_elt;			\
  }

#define REDUCTIONS_INIT(c_type, union_type, v_max, v_min)		\
  {									\
    if(nb_element == 1)							\
      reduction->saved.union_type = *((c_type*)reduction->variable);	\
									\
    switch(reduction->operator)						\
      {									\
      case STEP_PROD_REDUCE: REDUCTIONS_INIT_NEUTRAL_ELT(c_type, 1, nb_element); break; \
      case STEP_SUM_REDUCE: REDUCTIONS_INIT_NEUTRAL_ELT(c_type, 0, nb_element); break;	\
      case STEP_MAX_REDUCE: REDUCTIONS_INIT_NEUTRAL_ELT(c_type, v_max, nb_element); break; \
      case STEP_MIN_REDUCE: REDUCTIONS_INIT_NEUTRAL_ELT(c_type, v_min, nb_element); break; \
      default: assert(0);						\
      }									\
  }
#define REDUCTIONS_COMPLEX_INIT(c_type, union_type)			\
  {									\
    if(nb_element == 1)							\
      reduction->saved.union_type = *((c_type*)reduction->variable);	\
									\
    switch(reduction->operator)						\
      {									\
      case STEP_PROD_REDUCE: REDUCTIONS_INIT_NEUTRAL_ELT(c_type,1, nb_element); break;	\
      case STEP_SUM_REDUCE: REDUCTIONS_INIT_NEUTRAL_ELT(c_type,0, nb_element); break;	\
      default: assert(0);						\
      }									\
  }

void communications_initreduction(Descriptor_reduction *reduction, Descriptor_userArray *desc_userArray)
{
  uint32_t nb_element = 1;
  if (desc_userArray!=NULL)
    {
      uint32_t nbdims, d;
      INDEX_TYPE *bounds;

      nbdims = rg_get_userArrayDims(&(desc_userArray->boundsRegions));
      bounds = rg_get_simpleRegion(&(desc_userArray->boundsRegions), 0);
      for (d = 0; d < nbdims; d++)
	nb_element *= 1 + bounds[UP(d)] - bounds[LOW(d)];
    }

  switch(reduction->type)
    {
    case STEP_INTEGER: case STEP_INTEGER1: case STEP_INTEGER2: case STEP_INTEGER4: case STEP_INTEGER8:
      switch(step_types_table[reduction->type].type_size)
	{
	case 1: REDUCTIONS_INIT(int8_t, integer1, INT8_MAX, INT8_MIN); break;
	case 2: REDUCTIONS_INIT(int16_t, integer2, INT16_MAX, INT16_MIN); break;
	case 4: REDUCTIONS_INIT(int32_t, integer4, INT32_MAX, INT32_MIN); break;
	case 8: REDUCTIONS_INIT(int64_t, integer8, INT64_MAX, INT64_MIN); break;
	default: assert(0);
	}
      break;
    case STEP_REAL: case STEP_REAL4: case STEP_REAL8: case STEP_REAL16:
      switch(step_types_table[reduction->type].type_size)
	{
	case 4: REDUCTIONS_INIT(float, real4, FLT_MAX, FLT_MIN); break;
	case 8: REDUCTIONS_INIT(double, real8, DBL_MAX, DBL_MIN); break;
	case 16: REDUCTIONS_INIT(long double, real16, LDBL_MAX, LDBL_MIN); break;
	default: assert(0);
	}
      break;
    case STEP_COMPLEX: case STEP_COMPLEX8: case STEP_COMPLEX16:
      switch(step_types_table[reduction->type].type_size)
	{
	case 8: REDUCTIONS_COMPLEX_INIT(float complex, compl8); break;
	case 16: REDUCTIONS_COMPLEX_INIT(double complex, compl16); break;
	default: assert(0);
	}
      break;
    default: assert(0);
    }
}

/*
FSC
- initialiser les tableaux avec l'élément neutre
- supprimer la variable taille (correspond en fait à la valeur maximum de l'index de la première dimension)
- créer un type MPI correspondant au tableau (potentiellement multidimensionnel)
- supprimer le deuxieme allreduce sur le premier element et debugger le probleme
- revoir la premiere boucle sur h
   * savedUserArray est utilisé pour gérer l'entrelacement
   * ne pas utiliser savedUserArray dans le cas de la réduction
- revoir la sauvegarde
   * saved fonctionne pour un scalaire (voir l'union Value dans step_private.h)
*/

/*
   c_type: type in C
   u_type: type in union Value
   type_mpi:  MPI type
*/

static char *communication_operator_name(uint32_t operator)
{
  switch(operator)
    {
    case STEP_PROD_REDUCE: return "*";
    case STEP_MAX_REDUCE: return "MAX";
    case STEP_MIN_REDUCE: return "MIN";
    case STEP_SUM_REDUCE: return "+";
    default: return "?";
    }
}

#define OP_PRODUIT(a,b) ((a) * (b))
#define OP_SUM(a,b) ((a) + (b))

#define REDUCTIONS_OP(c_type, type_mpi, op, mpi_op)			\
  {									\
    MPI_Allreduce(reduction->variable, &buffer, nb_element, type_mpi, mpi_op, MPI_COMM_WORLD); \
    for(h=0;h<nb_element;h++)						\
      ((c_type *)(reduction->variable))[h] = op(((c_type *)initial_value)[h], ((c_type *)buffer)[h]); \
  }
#define REDUCTIONS(c_type, u_type, type_mpi)				\
  {									\
    c_type buffer[nb_element];						\
    c_type *initial_value;						\
    if (nb_element==1)							\
      initial_value= &(reduction->saved.u_type);			\
    else								\
      initial_value= desc_userArray->savedUserArray;			\
    switch(reduction->operator)						\
      {									\
      case STEP_PROD_REDUCE: REDUCTIONS_OP(c_type, type_mpi, OP_PRODUIT, MPI_PROD); break; \
      case STEP_SUM_REDUCE: REDUCTIONS_OP(c_type, type_mpi, OP_SUM, MPI_SUM); break; \
      case STEP_MAX_REDUCE: REDUCTIONS_OP(c_type, type_mpi, MAX, MPI_MAX); break; \
      case STEP_MIN_REDUCE: REDUCTIONS_OP(c_type, type_mpi, MIN, MPI_MIN); break; \
      default:								\
	assert(0);							\
      }									\
  }
#define REDUCTIONS_COMPLEX(c_type, u_type, type_mpi)			\
  {									\
    c_type buffer[nb_element];						\
    c_type *initial_value;						\
    if (nb_element==1)							\
      initial_value= &(reduction->saved.u_type);			\
    else								\
      initial_value= desc_userArray->savedUserArray;			\
    switch(reduction->operator)						\
      {									\
      case STEP_PROD_REDUCE: REDUCTIONS_OP(c_type, type_mpi, OP_PRODUIT, MPI_PROD); break; \
      case STEP_SUM_REDUCE: REDUCTIONS_OP(c_type, type_mpi, OP_SUM, MPI_SUM); break; \
      default:								\
	assert(0);							\
      }									\
  }


/* FSC
   voir l'enregistrement de reduction->type
   dans STEP_BEGIN_CONSTRUCT(parallel_construct)
   pour mettre le type MPI composé pour le tableau

   ou step_types_table[reduction->type].type_mpi?

   FSC voir s'il faut mettre en type_mpi la description du tableau
   initial ou la description de la région de tableau écrite

*/
void communications_reduction(Descriptor_reduction *reduction, Descriptor_userArray *desc_userArray)
{
  Descriptor_type d_type;
  IN_TRACE("reduction = %p, array = %p", reduction, desc_userArray);

  uint32_t h, nb_element = 1;
  if (desc_userArray!=NULL)
    {
      uint32_t nbdims, d;
      INDEX_TYPE *bounds;

      nbdims = rg_get_userArrayDims(&(desc_userArray->boundsRegions));
      bounds = rg_get_simpleRegion(&(desc_userArray->boundsRegions), 0);
      for (d = 0; d < nbdims; d++)
	nb_element *= 1 + bounds[UP(d)] - bounds[LOW(d)];
    }

  assert(reduction);

  STEP_COMMUNICATIONS_VERBOSE(printf("%u) REDUCTION %s type=%s OP=%s (%p)\n", MYRANK,  nb_element==1?"SCALAR":"ARRAY", communication_type_name(reduction->type), communication_operator_name(reduction->operator), reduction->variable););
  d_type = step_types_table[reduction->type];
  switch(reduction->type)
    {
    case STEP_INTEGER:
    case STEP_INTEGER1:
    case STEP_INTEGER2:
    case STEP_INTEGER4:
    case STEP_INTEGER8:
      switch(d_type.type_size)
	{
	case 1: REDUCTIONS(int8_t, integer1, d_type.type_mpi); break;
	case 2: REDUCTIONS(int16_t, integer2, d_type.type_mpi); break;
	case 4: REDUCTIONS(int32_t, integer4, d_type.type_mpi); break;
	case 8: REDUCTIONS(int64_t, integer8, d_type.type_mpi); break;
	default : assert(0);
	}
      break;
    case STEP_REAL:
    case STEP_REAL4:
    case STEP_REAL8:
    case STEP_REAL16:
      switch(d_type.type_size)
	{
	case 4:
	  REDUCTIONS(float, real4, d_type.type_mpi);
	  break;
	case 8:
	  REDUCTIONS(double, real8, d_type.type_mpi);
	  break;
	case 16:
	  REDUCTIONS(long double, real16, d_type.type_mpi);
	  break;
	default : assert(0);
	}
      break;
    case STEP_COMPLEX:
    case STEP_COMPLEX8:
    case STEP_COMPLEX16:
      switch(d_type.type_size)
	{
	case 8:
	  REDUCTIONS_COMPLEX(float complex, compl8, d_type.type_mpi);
	  break;
	case 16:
	  REDUCTIONS_COMPLEX(double complex, compl16, d_type.type_mpi);
	  break;
	default: assert(0);
	}
      break;
    default:
      assert(0);
    }

  if (desc_userArray!=NULL)
    {
      /* Mise a jour des données UPTODATE */
    }

  OUT_TRACE("end");
}

#ifdef TEST_COMMUNICATIONS
#include "regions.h"
int main(int argc, char **argv)
{
  uint32_t commsize;
  uint32_t myrank;
  MPI_Init(&argc,&argv);
  printf("Initializing communications...\n");
  communications_init();
  commsize = communications_get_commsize();
  myrank = communications_get_rank();

  printf("commsize = %d, myrank =%d\n", commsize, myrank);*/

  printf("Finalizing communications...\n");
  communications_finalize();
  return EXIT_SUCCESS;
}
#endif
