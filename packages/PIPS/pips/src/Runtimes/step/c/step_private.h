#ifndef __STEP_LOCAL_H__
#define __STEP_LOCAL_H__

#include <stdbool.h>
#include <stdint.h>
#include <complex.h>

#include "step_api.h"
#include "regions.h"
#include "array.h"


typedef struct
{
  void *userArray;            // @ of the array (used as key)
  void *savedUserArray;            // copy of array in case of interlaced communications

  uint32_t type;          // type of the array's data 
  composedRegion boundsRegions;         // index bounds of the array which dimension is rg_get_userArrayDims(boundsRegions) 

  /* size of Array: NB_NODES
     an element of Array has the type composedRegion representing a union of "regions of userArray"
   */
  Array uptodateArray;         
  /* size of Array: NB_NODES

     an element of Array has the type composedRegion representing a union of
     "regions of userArray". In this case a diff will be necessary
     between union of regions of each node.
   */
  Array interlacedArray;
}Descriptor_userArray;


typedef enum {parallel_work, do_work, master_work,critical_work} worksharing_type;

typedef struct
{
  worksharing_type type;  // parallel, do, master, ...
  composedRegion workchunkRegions;    // index bounds for each workchunk. The number of workchunks is regions_nb(workchunks)
  Array sharedArray;           // set of Descriptor_shared (identified by @ of the array)
  Array communicationsArray;   // set of pending communications (MPI_Request)
  Array privateArray;          // NOT USED YET set of private array (identified by @ of the array)
  Array reductionsArray;       // NOT USED YET set of Descriptor_reduction (identified by @ of the variable)
  Array scheduleArray;         // NOT USED YET mapping between workchunk and node (schedule.len == regions_dim(workchunks))
}Descriptor_worksharing;

typedef union
{
  int8_t integer1;
  int16_t integer2;
  int32_t integer4;
  int64_t integer8;
  float real4;
  double real8;
  long double real16;
  float complex compl8;
  double complex compl16;
}Value;


typedef struct
{
  void *variable;         // @ of the variable
  Value saved;            // original value
  uint32_t type;          // type of the variable
  uint32_t operator;      // reduction operator
}Descriptor_reduction;


typedef struct
{
  void *userArray;          // @ of the array used as key
  /*
    Region needed for each workchunk (receiveRegions.len == nb_workchunk, sendRegions.len == nb_workchunk)
    Each simpleRegion will match a workchunk. Order is significant.
   */
  composedRegion receiveRegions;   
  composedRegion sendRegions;      
  bool interlaced_p;        // true if send regions are interlaced
}Descriptor_shared;

typedef struct
{
  uint32_t commsize;          // The number of processes
  uint32_t rank;          // The rank of current process
  uint32_t language;      // The calling language (C or Fortran)
  uint32_t parallel_level;// The current nesting parallel level
  uint32_t initialized;    // Did the current process called step_init
  Descriptor_worksharing *current_worksharing;
}Step_internals;

extern Step_internals steprt_params;

#define IS_INITIALIZED (steprt_params.initialized)
#define CURRENTWORKSHARING (steprt_params.current_worksharing)
#define NB_WORKCHUNKS (rg_get_nb_simpleRegions(&(CURRENTWORKSHARING->workchunkRegions)))
#define NB_NODES (steprt_params.commsize)
#define MYRANK (steprt_params.rank)

#ifdef STEP_DEBUG
#undef STEP_DEBUG
#define STEP_DEBUG(code) {code}
#else
#define STEP_DEBUG(code) {}
#endif

#endif //__STEP_LOCAL_H__
