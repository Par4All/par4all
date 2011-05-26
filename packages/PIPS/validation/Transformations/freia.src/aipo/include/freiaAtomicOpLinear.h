/***************************************************************
 * Author : Christophe Clienti
 * Date : 10/22/2008
 * Copyright (c) 2008 Thales Research & Technologies (FR)
 ***************************************************************/

#ifdef __cplusplus
extern "C" {
#endif


#ifndef __FREIA_ATOMIC_OP_LINEAR_H__
#define __FREIA_ATOMIC_OP_LINEAR_H__

#include <freiaCommonTypes.h>


  /*!
   * \defgroup freia_aipo_linear Linear image processing operations
   * \ingroup freia_aipo
   * @{
   */


  /*!  
    \brief Compute a convolution imin using kernel declared in 8 connexity and
    store the result in imout

    The convolution will be done in-place if pointers to imin and imout
    are equals. 

    \param[out] imout destination image 
    \param[in] imin source image
    \param[in] kernel kernel declaration for the convolution.    
    \param[in] kernelwidth width of the kernel
    \param[in] kernelheight height of the kernel
    \return error code
  */
  extern freia_status freia_aipo_convolution(freia_data2d *imout, freia_data2d *imin, 
					    int32_t *kernel, uint32_t kernelwidth, uint32_t kernelheight);

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

    \warning SPOCSIM USE EXPLICITALLY FULGURO
  */
  extern freia_status freia_aipo_fast_correlation(freia_data2d *imout, freia_data2d *imin, freia_data2d *imref, uint32_t horizon);
  



  /*!@}*/


#endif

#ifdef __cplusplus
}
#endif
