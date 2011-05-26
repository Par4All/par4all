/***************************************************************
 * Author : Christophe Clienti
 * Date : 10/22/2008
 * Copyright (c) 2008 Thales Research & Technologies (FR)
 ***************************************************************/

#ifdef __cplusplus
extern "C" {
#endif


#ifndef __FREIA_COMPLEX_OP_LINEAR_H__
#define __FREIA_COMPLEX_OP_LINEAR_H__

#include <freiaCommonTypes.h>

  /*!
   * \defgroup freia_cipo_linear  Linear complex processing
   * \ingroup freia_cipo
   * @{
   */

  /*!  
    \brief Compute a fast correlation using sad and store the result in imout

    \warning SPOCSIM USE EXPLICITALLY FULGURO

    Some important properties about image sizes must be respected :
    - imref size : NxM (Height,Width)
    - imin size : (N-2*horizon) x (M-2*horizon)
    - imout size : (2*horizon) x (2*horizon)

    Two possibilities to manage correctly image sizes :
    - Use images with correct size
    - Set working area for each image with correct sizes

    \param[out] imout destination image 
    \param[in] imin source image
    \param[in] imref reference image
    \param[in] horizon correlation horizon
    \return error code
  */
   freia_status freia_cipo_fast_correlation(freia_data2d *imout, freia_data2d *imin, freia_data2d *imref,  uint32_t horizon);

  /*!@}*/

#endif

#ifdef __cplusplus
}
#endif
