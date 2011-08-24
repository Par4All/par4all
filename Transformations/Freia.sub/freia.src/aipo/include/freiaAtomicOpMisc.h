/***************************************************************
 * Author : Christophe Clienti
 * Date : 10/22/2008
 * Copyright (c) 2008 Thales Research & Technologies (FR)
 ***************************************************************/

#ifdef __cplusplus
extern "C" {
#endif


#ifndef __FREIA_ATOMIC_OP_MISC_H__
#define __FREIA_ATOMIC_OP_MISC_H__

#include <freiaCommonTypes.h>

  /*!
   * \defgroup freia_aipo_misc Miscellaneous operations
   * \ingroup freia_aipo
   * @{
   */


  /*!  
    \brief Copy imin to imout. Take into account working area and cast if necessary.

    Only image with same type could be copied

    Working areas are taken into account, so images must have only the
    same working area in terms of width and height, the working area
    position in the image has no importance.

    This function can be used to extract region of interest.

    \param[out] imout
    \param[in] imin
    \return error code
  */    
  extern freia_status freia_aipo_copy(freia_data2d *imout, freia_data2d *imin);


  /*!  
    \brief Copy Cast imin to imout. Take into account working area and cast if necessary.

    Valid cases are : 
    - 8 bits image to 16 bits image
    - 16 bits image to 8 bits image
    - 16 bits image to 16 bits image
    - 8 bits image to 8 bits image

    Working areas are taken into account, so images must have only the
    same working area in terms of width and height, the working area
    position in the image has no importance.

    This function can be used to extract region of interest.

    \param[out] imout
    \param[in] imin
    \return error code
  */    
  extern freia_status freia_aipo_cast(freia_data2d *imout, freia_data2d *imin);



  /*!  
    \brief Set all pixels to a constant value

    \param[out] imout
    \param[in] constant
    \return error code
  */    
  extern freia_status freia_aipo_set_constant(freia_data2d *imout, int32_t constant);



  /*!  
    \brief Apply a threshold to input image. 

    Pixels between boundinf and boundsup are conserved. others are replaced by zero :
    - imout(p) = ( boundinf <= imin(p) <= boundsup  ? imin(p) : 0 )

    if binarize is set to true, conserved pixels will be replaced by the maximum value regarding the image type :
    - imout(p) = ( boundinf <= imin(p) <= boundsup  ? (binarize==true ? MAX : imin(p)) : 0 )


    \param[out] imout
    \param[in] imin
    \param[in] boundinf lower bound for the threshold
    \param[in] boundsup upper bound for the threshold
    \param[in] binarize must be set to true for binarization
    \return error code
  */    
  extern freia_status freia_aipo_threshold(freia_data2d *imout, freia_data2d *imin, int32_t boundinf, int32_t boundsup, bool binarize);


  /*!  
    \brief copy pixel of input to output when immask pixels equal to constant

    Pixels between boundinf and boundsup are conserved. others are replaced by zero :
    - imout(p) = ( immask(p) == constant) ? imin(p) : 0 )

    \param[out] imout
    \param[in] imask
    \param[in] imin
    \param[in] constant
    \return error code
  */    
  extern freia_status freia_aipo_replace_const(freia_data2d *imout, freia_data2d *immask, freia_data2d *imin, int32_t constant);


  /*!@}*/


#endif

#ifdef __cplusplus
}
#endif
