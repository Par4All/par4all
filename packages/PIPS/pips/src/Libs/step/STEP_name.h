/* STEP variable name
   fichier inclus dans : ri-utils-local.h
*/

#define DIR_CALL "STEP_DIRECTIVES_"

// STEP API Runtime base name
/* 
!!!!! Les noms des intrinseques doivent etre en majuscule !!!!!
*/

#define RT_STEP_get_myloopslice "STEP_GET_MYLOOPSLICE"
#define RT_STEP_get_i_low "STEP_GET_I_LOW"
#define RT_STEP_get_i_up "STEP_GET_I_UP"
#define RT_STEP_Init_Fortran_Order  "STEP_INIT_FORTRAN_ORDER"
#define RT_STEP_Init_C_Order  "STEP_INIT_C_ORDER"
#define RT_STEP_Critical_Spawn  "STEP_CRITICAL_SPAWN"
#define RT_STEP_Finalize "STEP_FINALIZE"
#define RT_STEP_Barrier "STEP_BARRIER"
#define RT_STEP_Get_Commsize "STEP_GET_COMMSIZE"
#define RT_STEP_Get_rank "STEP_GET_RANK"
#define RT_STEP_Get_thread_num "STEP_GET_THREAD_NUM"
#define RT_STEP_ComputeLoopSlices "STEP_COMPUTE_LOOPSLICES"
#define RT_STEP_GetLoopBounds "STEP_GET_LOOPBOUNDS"
#define RT_STEP_SizeRegion "STEP_SIZEREGION"
#define RT_STEP_WaitAll "STEP_WAITALL"
#define RT_STEP_Begin "STEP_CONSTRUCT_BEGIN"
#define RT_STEP_End "STEP_CONSTRUCT_END"
#define RT_STEP_Request "STEP_CRITICAL_REQUEST"
#define RT_STEP_Release "STEP_CRITICAL_RELEASE"
#define RT_STEP_Get_Nextprocess "STEP_CRITICAL_GET_NEXTPROCESS"

#define RT_STEP_InitReduction "STEP_INITREDUCTION"
#define RT_STEP_Reduction "STEP_REDUCTION"
#define RT_STEP_MasterToAllScalar "STEP_MASTERTOALLSCALAR"
#define RT_STEP_MasterToAllRegion "STEP_MASTERTOALLREGION"
#define RT_STEP_AllToMasterRegion "STEP_ALLTOMASTERREGION"
#define RT_STEP_Critical_set_CurrentUptodateScalar "STEP_CRITICAL_SET_CURRENTUPTODATESCALAR"
#define RT_STEP_Critical_set_CurrentUptodateRegion "STEP_CRITICAL_SET_CURRENTUPTODATEREGION"
#define RT_STEP_Critical_get_CurrentUptodateScalar "STEP_CRITICAL_GET_CURRENTUPTODATESCALAR"
#define RT_STEP_Critical_get_CurrentUptodateRegion "STEP_CRITICAL_GET_CURRENTUPTODATEREGION"
#define RT_STEP_Critical_FinalUptodateScalar "STEP_CRITICAL_FINALUPTODATESCALAR"
#define RT_STEP_Critical_FinalUptodateRegion "STEP_CRITICAL_FINALUPTODATEREGION"

#define RT_STEP_Critical_Check_n_threads "STEP_CRITICAL_CHECK_N_THREADS"


#define RT_STEP_AlltoAllRegion "STEP_ALLTOALLREGION"
#define RT_STEP_InitInterlaced "STEP_INITINTERLACED"
#define RT_STEP_AlltoAllRegion_Merge "STEP_ALLTOALLREGION_MERGE"

#define RT_STEP_InitArrayRegions "STEP_INIT_ARRAYREGIONS"
#define RT_STEP_Set_SendRegions "STEP_SET_SENDREGIONS"
#define RT_STEP_Set_InterlacedSendRegions "STEP_SET_INTERLACED_SENDREGIONS"
#define RT_STEP_Set_ReductionSendRegions "STEP_SET_REDUCTION_SENDREGIONS"
#define RT_STEP_Set_RecvRegions "STEP_SET_RECVREGIONS"
#define RT_STEP_AllToAll_Full "STEP_ALLTOALL_FULL"
#define RT_STEP_AllToAll_FullInterlaced "STEP_ALLTOALL_FULL_INTERLACED"
#define RT_STEP_AllToAll_Partial "STEP_ALLTOALL_PARTIAL"
#define RT_STEP_AllToAll_PartialInterlaced "STEP_ALLTOALL_PARTIAL_INTERLACED"


// in STEP.h
#define STEP_MAX_NBNODE_NAME "STEP_MAX_NBNODE" // ne devrait plus etre necessaire avec la RT en C
#define STEP_MAX_NB_LOOPSLICES_NAME "MAX_NB_LOOPSLICES"  // ne devrait plus etre necessaire avec la RT en C
#define STEP_MAX_DIM_NAME "STEP_MAX_DIM"  // ne devrait plus etre necessaire avec la RT en C
#define STEP_MAX_NBREQ_NAME "STEP_MAX_NBREQ"  // ne devrait plus etre necessaire avec la RT en C
#define STEP_SIZE_REGION_NAME RT_STEP_SizeRegion
#define STEP_STATUS_NAME "STEP_Status"
#define STEP_NBREQUEST_NAME "STEP_NbRequest"
#define STEP_INDEX_SLICE_LOW_NAME "IDX_SLICE_LOW"
#define STEP_INDEX_SLICE_UP_NAME "IDX_SLICE_UP"
#define STEP_IDX_NAME "STEP_IDX"
#define STEP_NONBLOCKING_NAME "STEP_NBLOCKING_ALG"
#define STEP_BLOCKING1_NAME "STEP_BLOCKING1"
#define STEP_BLOCKING2_NAME "STEP_BLOCKING2"
#define STEP_TAG_DEFAULT_NAME "STEP_TAG_DEFAULT"

#define STEP_SUM_NAME "STEP_SUM"
#define STEP_PROD_NAME "STEP_PROD"
#define STEP_MINUS_NAME "STEP_MINUS"
#define STEP_AND_NAME "STEP_AND"
#define STEP_OR_NAME "STEP_OR"
#define STEP_EQV_NAME "STEP_EQV"
#define STEP_NEQV_NAME "STEP_NEQV"
#define STEP_MAX_NAME "STEP_MAX_"
#define STEP_MIN_NAME "STEP_MIN_"
#define STEP_IAND_NAME "STEP_IAND"
#define STEP_IOR_NAME "STEP_IOR"
#define STEP_IEOR_NAME "STEP_IEOR"

/* In steprt_common.h*/
#define STEP_INT_1_NAME                 "STEP_INTEGER1"
#define STEP_INT_2_NAME                 "STEP_INTEGER2"
#define STEP_INT_4_NAME                 "STEP_INTEGER4"
#define STEP_INT_8_NAME                 "STEP_INTEGER8"
#define STEP_REAL4_NAME                 "STEP_REAL4"
#define STEP_REAL8_NAME                 "STEP_REAL8"
#define STEP_COMPLEX8_NAME              "STEP_COMPLEX8"
#define STEP_COMPLEX16_NAME             "STEP_COMPLEX16"


#define STEP_COMM_SIZE_NAME     	"STEP_COMM_SIZE"
#define STEP_COMM_RANK_NAME	        "STEP_COMM_RANK"
#define STEP_FORTRAN_NAME		"STEP_FORTRAN"

#define STEP_MPI_STATUS_SIZE_NAME "MPI_STATUS_SIZE"

#define STEP_PARALLEL_NAME "STEP_PARALLEL"
#define STEP_DO_NAME "STEP_DO"
#define STEP_PARALLELDO_NAME "STEP_PARALLEL_DO"
#define STEP_MASTER_NAME "STEP_MASTER"
#define STEP_CRITICAL_NAME "STEP_CRITICAL"

// in _MPI.f
#define STEP_MAX_NB_REQUEST_NAME "MAX_NB_REQUEST"
#define STEP_REQUEST_NAME "STEP_REQUESTS"
#define STEP_SLICE_INDEX_NAME "IDX"
#define STEP_INDEX_LOW_NAME(index) concatenate(entity_user_name(index),"_LOW",NULL)
#define STEP_INDEX_UP_NAME(index) concatenate(entity_user_name(index),"_UP",NULL)
#define STEP_BOUNDS_LOW(index) concatenate("STEP_", entity_user_name(index),"_LOW", NULL)
#define STEP_BOUNDS_UP(index) concatenate("STEP_", entity_user_name(index),"_UP", NULL)
#define STEP_LOOPSLICES_NAME(index) concatenate("STEP_",entity_user_name(index),"_LOOPSLICES",NULL)
#define STEP_SR_NAME(array) concatenate("STEP_SR_",entity_user_name(array),NULL)
#define STEP_RR_NAME(array) concatenate("STEP_RR_",entity_user_name(array),NULL)
#define STEP_INITIAL_NAME(array) concatenate("STEP_INITIAL_",entity_user_name(array),NULL)
#define STEP_BUFFER_NAME(array) concatenate("STEP_BUFFER_", entity_user_name(array),NULL)

#define STEP_REDUC_NAME(variable) concatenate("STEP_",entity_user_name(variable),"_REDUC",NULL)
