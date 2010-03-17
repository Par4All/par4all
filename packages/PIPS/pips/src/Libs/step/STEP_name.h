/* STEP variable name
   fichier inclus dans : ri-utils-local.h
*/

// STEP API Runtime base name
/* 
!!!!! Les noms des intrinseques doivent etre en majuscule !!!!!
*/
#define RT_STEP_get_myloopslice "STEP_GET_MYLOOPSLICE"
#define RT_STEP_get_i_low "STEP_GET_I_LOW"
#define RT_STEP_get_i_up "STEP_GET_I_UP"
#define RT_STEP_Init "STEP_INIT"
#define RT_STEP_Finalize "STEP_FINALIZE"
#define RT_STEP_Barrier "STEP_BARRIER"
#define RT_STEP_Get_size "STEP_GET_SIZE"
#define RT_STEP_Get_rank "STEP_GET_RANK"
#define RT_STEP_Get_thread_num "STEP_GET_THREAD_NUM"
#define RT_STEP_ComputeLoopSlices "STEP_COMPUTELOOPSLICES"
#define RT_STEP_SizeRegion "STEP_SIZEREGION"
#define RT_STEP_WaitAll "STEP_WAITALL"

#define RT_STEP_InitReduction "STEP_INITREDUCTION"
#define RT_STEP_Reduction "STEP_REDUCTION"
#define RT_STEP_MasterToAllScalar "STEP_MASTERTOALLSCALAR"
#define RT_STEP_MasterToAllRegion "STEP_MASTERTOALLREGION"
#define RT_STEP_AlltoAllRegion "STEP_ALLTOALLREGION"
#define RT_STEP_InitInterlaced "STEP_INITINTERLACED"
#define RT_STEP_AlltoAllRegion_Merge "STEP_ALLTOALLREGION_MERGE"

// in STEP.h
#define STEP_MAX_NBNODE_NAME "STEP_MAX_NBNODE" // ne devrait plus etre necessaire avec la RT en C
#define STEP_MAX_NB_LOOPSLICES_NAME "MAX_NB_LOOPSLICES"  // ne devrait plus etre necessaire avec la RT en C
#define STEP_MAX_DIM_NAME "STEP_MAX_DIM"  // ne devrait plus etre necessaire avec la RT en C
#define STEP_MAX_NBREQ_NAME "STEP_MAX_NBREQ"  // ne devrait plus etre necessaire avec la RT en C
#define STEP_SIZE_NAME "STEP_Size"
#define STEP_RANK_NAME "STEP_Rank"
#define STEP_SIZE_REGION_NAME RT_STEP_SizeRegion
#define STEP_STATUS_NAME "STEP_Status"
#define STEP_NBREQUEST_NAME "STEP_NbRequest"
#define STEP_INDEX_SLICE_LOW_NAME "IDX_SLICE_LOW"
#define STEP_INDEX_SLICE_UP_NAME "IDX_SLICE_UP"
#define STEP_IDX_NAME "STEP_IDX"
#define STEP_NONBLOCKING_NAME "STEP_NONBLOCKING"
#define STEP_BLOCKING1_NAME "STEP_BLOCKING1"
#define STEP_BLOCKING2_NAME "STEP_BLOCKING2"

#define STEP_SUM_NAME "STEP_SUM"
#define STEP_PROD_NAME "STEP_PROD"
#define STEP_MINUS_NAME "STEP_MINUS"
#define STEP_AND_NAME "STEP_AND"
#define STEP_OR_NAME "STEP_OR"
#define STEP_EQV_NAME "STEP_EQV"
#define STEP_NEQV_NAME "STEP_NEQV"
#define STEP_MAX_NAME "STEP_MAX"
#define STEP_MIN_NAME "STEP_MIN"
#define STEP_IAND_NAME "STEP_IAND"
#define STEP_IOR_NAME "STEP_IOR"
#define STEP_IEOR_NAME "STEP_IEOR"

#define STEP_MPI_STATUS_SIZE_NAME "MPI_STATUS_SIZE"


// in _MPI.f
#define STEP_MAX_NB_REQUEST_NAME "MAX_NB_REQUEST"
#define STEP_REQUEST_NAME "STEP_REQUESTS"
#define STEP_INDEX_NAME "IDX"
#define STEP_INDEX_LOW_NAME(index) concatenate(entity_user_name(index),"_LOW",NULL)
#define STEP_INDEX_UP_NAME(index) concatenate(entity_user_name(index),"_UP",NULL)
#define STEP_BOUNDS_LOW(index) concatenate("STEP_", entity_user_name(index),"_LOW", NULL)
#define STEP_BOUNDS_UP(index) concatenate("STEP_", entity_user_name(index),"_UP", NULL)
#define STEP_LOOPSLICES_NAME(index) concatenate("STEP_",entity_user_name(index),"_LOOPSLICES",NULL)
#define STEP_SR_NAME(array) concatenate("STEP_SR_",entity_user_name(array),NULL)
#define STEP_INITIAL_NAME(array) concatenate("STEP_INITIAL_",entity_user_name(array),NULL)
#define STEP_BUFFER_NAME(array) concatenate("STEP_BUFFER_", entity_user_name(array),NULL)

#define STEP_REDUC_NAME(variable) concatenate("STEP_",entity_user_name(variable),"_REDUC",NULL)
