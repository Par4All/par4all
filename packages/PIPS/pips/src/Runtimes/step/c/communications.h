#ifndef __COMMUNICATIONS_H__
#define __COMMUNICATIONS_H__

#include "step_private.h"
#include "mpi.h"

#ifndef INLINE
#define INLINE extern inline
#endif

#ifndef int_MPI
#define int_MPI int32_t
#endif

extern void step_sizetype_(int32_t *step_type, int32_t *type_size);
extern void communications_init(void);
extern void communications_finalize(void);

extern void communications_get_commsize(uint32_t *size);
extern void communications_get_rank(uint32_t *rank);

extern void *communications_alloc_buffer(Descriptor_userArray *desc_userArray, size_t *alloc);

extern void communications_allToAll(Descriptor_userArray *desc_userArray, Array *regionReceive, uint32_t algorithm, int_MPI tag);
extern void communications_waitall(Descriptor_worksharing *worksharing);

extern void communications_barrier(void);

extern void communications_oneToAll_Scalar(void *scalar, uint32_t type, uint32_t algorithm);
extern void communications_oneToAll_Array(Descriptor_userArray *desc_userArray, uint32_t algorithm);

extern void communications_initreduction(Descriptor_reduction *reduction, Descriptor_userArray *desc_userArray);
extern void communications_reduction(Descriptor_reduction *desc_reduction);

extern void communications_set_currentuptodate_scalar(void *scalar, uint32_t type);
extern void communications_get_currentuptodate_scalar(void *scalar, uint32_t type);
extern void communications_finaluptodate_scalar(void *scalar, uint32_t type);
extern void communications_finaluptodate_array(Descriptor_userArray *desc_array);
extern void communications_set_currentuptodate_array(Descriptor_userArray *desc_array);
extern void communications_get_currentuptodate_array(Descriptor_userArray *desc_array);
extern void communications_critical_spawn ();
extern void communications_critical_request (int num_current_critical);
extern void communications_critical_get_nextprocess ();
extern void communications_critical_release ();
extern void communications_critical_stop_pcoord ();

#endif /* __COMMUNICATIONS_H__ */
