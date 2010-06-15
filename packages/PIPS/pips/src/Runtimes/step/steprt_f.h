!*******************************************************************************
!*                                                                             *
!*   Authors 	        :       Abdellah. Kouadri                              *
!*				Daniel Millot                          	       *
!*                              Frederique Silber-Chaussumier                  *
!*				                                               *
!*   Date		:       25/06/2009                                     *
!*						                               *
!*   File		:	steprt_f.h                                     *
!*							                       *
!*   Version		:       1.1				               *
!*   Description	:	Fortran runtime interface                      *
! ******************************************************************************



! Datatype
      INTEGER STEP_INTEGER1
      INTEGER STEP_INTEGER2
      INTEGER STEP_INTEGER4
      INTEGER STEP_INTEGER8
      INTEGER STEP_REAL4
      INTEGER STEP_REAL8
      INTEGER STEP_REAL16
      INTEGER STEP_COMPLEX8
      INTEGER STEP_COMPLEX16
      INTEGER STEP_COMPLEX32
      INTEGER STEP_INTEGER
      INTEGER STEP_REAL
      INTEGER STEP_COMPLEX
      INTEGER STEP_DOUBLE_PRECISION
!
      PARAMETER (STEP_INTEGER1 = 1)
      PARAMETER (STEP_INTEGER2 = 2)
      PARAMETER (STEP_INTEGER4 = 3)
      PARAMETER (STEP_INTEGER8 = 4)
      PARAMETER (STEP_REAL4 = 5)
      PARAMETER (STEP_REAL8 = 6)
      PARAMETER (STEP_REAL16 = 7)
      PARAMETER (STEP_COMPLEX8 = 8)
      PARAMETER (STEP_COMPLEX16 = 9)
      PARAMETER (STEP_COMPLEX32 = 10)
      PARAMETER (STEP_INTEGER = 11)
      PARAMETER (STEP_REAL = 12)
      PARAMETER (STEP_COMPLEX = 13)
      PARAMETER (STEP_DOUBLE_PRECISION = 14)

! Communication tag
      INTEGER   STEP_TAG_DEFAULT
!
      PARAMETER (STEP_TAG_DEFAULT = 0)

! Communication algorithms
      INTEGER 	STEP_NBLOCKING_ALG 
      INTEGER	STEP_BLOCKING_ALG_1
      INTEGER	STEP_BLOCKING_ALG_2
      INTEGER	STEP_BLOCKING_ALG_3
      INTEGER	STEP_BLOCKING_ALG_4
! reduction operators 
      INTEGER STEP_SUM	
      INTEGER STEP_MAX_
      INTEGER STEP_MIN_
      INTEGER STEP_PROD
      INTEGER STEP_LAND	
      INTEGER STEP_BAND
      INTEGER STEP_LOR	
      INTEGER STEP_BOR	
      INTEGER STEP_LXOR	
      INTEGER STEP_BXOR	
      INTEGER STEP_MINLOC
      INTEGER STEP_MAXLOC
!
      PARAMETER (STEP_NBLOCKING_ALG  = 0)
      PARAMETER (STEP_BLOCKING_ALG_1 = 1)
      PARAMETER (STEP_BLOCKING_ALG_2 = 2)
      PARAMETER (STEP_BLOCKING_ALG_3 = 3)
      PARAMETER (STEP_BLOCKING_ALG_4 = 4)
!
      PARAMETER (STEP_SUM  =	3)
      PARAMETER (STEP_MAX_ =	1)
      PARAMETER (STEP_MIN_ =	2)
      PARAMETER (STEP_PROD =	0)
      PARAMETER (STEP_LAND =	4)
      PARAMETER (STEP_BAND =	5)
      PARAMETER (STEP_LOR  =	6)
      PARAMETER (STEP_BOR  =	7)
      PARAMETER (STEP_LXOR =	8)
      PARAMETER (STEP_BXOR =	9)
      PARAMETER (STEP_MINLOC =	10)
      PARAMETER (STEP_MAXLOC =	11)

