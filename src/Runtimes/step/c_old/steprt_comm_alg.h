/**
*                                                                             
*   \file             steprt_comm_alg.h
*   \author           Abdellah Kouadri.
*                     Daniel Millot.
*                     Frédérique Silber-Chaussumier.
*                     Alain Muller
*   \date             04/02/2010
*   \version          2.0
*   \brief            Interface file for steprt_comm_alg.c file
*
*/

#ifndef STEP_COMM_ALG_H_
#define STEP_COMM_ALG_H_


#include "steprt_private.h"


void step_alltoall(array_identifier_t *array_desciptor, int algorithm, int tag);

#endif
