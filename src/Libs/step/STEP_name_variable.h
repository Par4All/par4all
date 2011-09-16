/*
  Variable name in genereted file
*/

#define STEP_SLICE_INDEX_NAME "IDX"
#define STEP_INDEX_LOW_NAME(index) concatenate(entity_user_name(index),"_LOW",NULL)
#define STEP_INDEX_UP_NAME(index) concatenate(entity_user_name(index),"_UP",NULL)
#define STEP_BOUNDS_LOW(index) concatenate("STEP_", entity_user_name(index),"_LOW", NULL)
#define STEP_BOUNDS_UP(index) concatenate("STEP_", entity_user_name(index),"_UP", NULL)
#define STEP_LOOPSLICES_NAME(index) concatenate("STEP_",entity_user_name(index),"_LOOPSLICES",NULL)
#define STEP_SR_NAME(array) concatenate("STEP_SR_",entity_user_name(array),NULL)
#define STEP_RR_NAME(array) concatenate("STEP_RR_",entity_user_name(array),NULL)

/* In Runtime/step/STEP.h */
#define STEP_COMM_SIZE_NAME         "STEP_COMM_SIZE"    /* variable */
#define STEP_COMM_RANK_NAME	    "STEP_COMM_RANK"    /* variable */
