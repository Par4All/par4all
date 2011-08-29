/***************************************************************
 * Author : Christophe Clienti
 * Date : 10/22/2008
 * Copyright (c) 2008 Thales Research & Technologies (FR)
 ***************************************************************/

#ifdef __cplusplus
extern "C" {
#endif


#ifndef __FREIA_COMPLEX_OP_MEASURE_H__
#define __FREIA_COMPLEX_OP_MEASURE_H__

#include <freiaCommonTypes.h>

  /*!
   * \defgroup freia_cipo_measure  Measure complex processing
   * \ingroup freia_cipo
   * @{
   */

  /*!  
    \brief Sum of Absolute Differences sum(|I1-I2|)
    
    \param[in] imin1 source 1
    \param[in] imin2 source 2
    \param[out] sad result of sum of absolute differences
    \return SAD
  */
  extern freia_status freia_cipo_global_sad(freia_data2d *imin1, freia_data2d *imin2, uint32_t *sad);




  /*!@}*/

#endif

#ifdef __cplusplus
}
#endif
