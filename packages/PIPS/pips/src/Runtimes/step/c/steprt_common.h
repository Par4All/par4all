/**
*                                                                             
*   \file             steprt_common.h
*   \author           Abdellah Kouadri.
*                     Daniel Millot.
*                     Frédérique Silber-Chaussumier.
*   \date             22/10/2009
*   \version          1.1
*   \brief            Some useful declarations and definitions 
*
*/

#ifndef STEP_COMMON_H_
#define STEP_COMMON_H_


/**
* Definitions used to size various arrays used in the runtime
*/
#define MAX_DIMS 						16
#define MAX_PROCESSES						16
#define MAX_REQUESTS						MAX_PROCESSES*2
#define	MAX_SIZE						2048*1024
#define STEP_MAX_TYPES                                          16
#define STEP_TAG_DEFAULT                                         0
/**
* Data types actually handled by the runtime 
*/

#define STEP_Datatype						int
#define STEP_TYPE_UNDIFINED                                     0
#define STEP_INTEGER1			        		1
#define STEP_INTEGER2						2
#define STEP_INTEGER4						3
#define STEP_INTEGER8						4
#define STEP_REAL4						5
#define STEP_REAL8						6
#define STEP_REAL16						7
#define STEP_COMPLEX8						8
#define STEP_COMPLEX16						9
#define STEP_COMPLEX32                                          10
#define STEP_INTEGER						11
#define STEP_REAL                                               12
#define STEP_COMPLEX					        13
#define STEP_DOUBLE_PRECISION                                   14

/**
* Supported communication algorithms 
*/

#define STEP_NBLOCKING_ALG 					0
#define	STEP_BLOCKING_ALG_1					1
#define	STEP_BLOCKING_ALG_2					2
#define	STEP_BLOCKING_ALG_3					3
#define	STEP_BLOCKING_ALG_4					4
#define	STEP_ONETOALL_BCAST					0


/**
* Supported reduction operators 
*/
#define STEP_SUM				 	        3
#define STEP_MAX_						1
#define STEP_MIN_						2
#define STEP_PROD						0
#define STEP_LAND						4
#define STEP_BAND						5
#define STEP_LOR						6
#define STEP_BOR						7
#define STEP_LXOR						8
#define STEP_BXOR						9
#define STEP_MINLOC						10
#define STEP_MAXLOC						11

/**
* Supported languages 
*/

#define STEP_FORTRAN					MPI_ORDER_FORTRAN
#define STEP_C						MPI_ORDER_C



/**
* Helpful macros 
*/

#undef 	STEP_DEBUG_REGIONS
#undef 	STEP_DEBUG_LOOP
#define MAX(X,Y) ((X)>(Y)) ? X:Y
#define MIN(X,Y) ((X)<(Y)) ? X:Y


/**
* Error status
*/
#define STEP_OK							 0
#define STEP_ERROR						-1
#define STEP_FATAL_ERROR					-2
#define STEP_CAN_NOT_PROCEED					-3
#define STEP_ALREADY_INITIALIZED				-4



#endif
