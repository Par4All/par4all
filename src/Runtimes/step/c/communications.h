#ifndef __COMMUNICATIONS_H__
#define __COMMUNICATIONS_H__

#include "step_private.h"
#include "array.h"
#include "mpi.h"

#ifndef INLINE
#define INLINE extern inline
#endif

#ifndef int_MPI
#define int_MPI int32_t
#endif

extern uint32_t communications_NB_NODES;
extern uint32_t communications_MY_RANK;
extern uint32_t communications_LANGUAGE_ORDER;

extern Descriptor_userArray *communications_set_userArrayDescriptor(Descriptor_userArray *desc_userArray, void *userArray, uint32_t type, uint nbdims, INDEX_TYPE *bounds);
extern void communications_unset_userArrayDescriptor(Descriptor_userArray *desc);

extern void step_sizetype_(int32_t *step_type, int32_t *type_size);
extern void communications_init(int language);
extern void communications_finalize(void);

extern uint32_t communications_get_commsize(void);
extern uint32_t communications_get_rank(void);

extern void *communications_alloc_buffer(Descriptor_userArray *desc_userArray, size_t *alloc);
extern void communications_allToAll(Descriptor_userArray *desc_userArray,  Array *toSend, Array *toReceive, bool is_interlaced, uint32_t algorithm, int_MPI tag, Array *pending_communications);
extern void communications_waitall(Array *communicationsArray);

extern void communications_barrier(void);

extern void communications_oneToAll_Scalar(void *scalar, uint32_t type, uint32_t algorithm);
extern void communications_oneToAll_Array(Descriptor_userArray *desc_userArray, uint32_t algorithm);

extern void communications_initreduction(Descriptor_reduction *reduction, Descriptor_userArray *desc_userArray);
extern void communications_reduction(Descriptor_reduction *desc_reduction, Descriptor_userArray *desc_userArray);

#endif /* __COMMUNICATIONS_H__ */
