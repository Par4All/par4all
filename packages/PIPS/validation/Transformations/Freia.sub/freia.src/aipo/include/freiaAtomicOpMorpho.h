/***************************************************************
 * Author : Christophe Clienti
 * Date : 10/22/2008
 * Copyright (c) 2008 Thales Research & Technologies (FR)
 ***************************************************************/

#ifdef __cplusplus
extern "C" {
#endif


#ifndef __FREIA_ATOMIC_OP_MORPHO_H__
#define __FREIA_ATOMIC_OP_MORPHO_H__

#include <freiaCommonTypes.h>


  /*!
   * \defgroup freia_aipo_morpho Basic mathematical morphology operations
   * \ingroup freia_aipo
   * @{
   */



  /*!  
    \brief Median imin using kernel declared in 8 connexity and
    store the result in imout

    The erosion will be done in-place if pointers to imin and imout
    are equals. The kernel must be an array of nine values
    [a,b,c,d,e,f,g,h,i] corresponding to the following shape :

    a b c

    d e f

    g h i

    The kernel is centered on "e".

    In order to take into account a specific neighbor during
    computations, the value at the corresponding index in the kernel
    must be set to something different of 0. To desactivate a specific
    neighbor, the corresponding value in the kernel must be set to 0.
    

    \param[out] imout destination image 
    \param[in] imin source image
    \param[in] kernel kernel declaration for the erosion.    
  */
  extern freia_status freia_aipo_median_8c(freia_data2d *imout, freia_data2d *imin, const int32_t *kernel);
 

  /*!  
    \brief Erode imin using kernel declared in 8 connexity and
    store the result in imout

    The erosion will be done in-place if pointers to imin and imout
    are equals. The kernel must be an array of nine values
    [a,b,c,d,e,f,g,h,i] corresponding to the following shape :

    a b c

    d e f

    g h i

    The kernel is centered on "e".

    In order to take into account a specific neighbor during
    computations, the value at the corresponding index in the kernel
    must be set to something different of 0. To desactivate a specific
    neighbor, the corresponding value in the kernel must be set to 0.
    

    \param[out] imout destination image 
    \param[in] imin source image
    \param[in] kernel kernel declaration for the erosion.    
  */
  extern freia_status freia_aipo_erode_8c(freia_data2d *imout, freia_data2d *imin, const int32_t *kernel);
 


  /*!  
    \brief Dilate imin using kernel declared in 8 connexity and
    store the result in imout
    
    \sa freia_aipo_erode_8c

    \param[out] imout destination image
    \param[in] imin source image
    \param[in] kernel kernel declaration for the erosion.
  */
  extern freia_status freia_aipo_dilate_8c(freia_data2d *imout, freia_data2d *imin, const int32_t *kernel);



  /*!  
    \brief Erode imin using kernel declared in 6 connexity and
    store the result in imout

    The erosion will be done in-place if pointers to imin and imout
    are equals. The kernel must be an array of seven values
    [a,b,c,d,e,f,g,0,0] corresponding to the following shape :

     a b

    c d e

     f g 

    The kernel is centered on "d".

    In order to take into account a specific neighbor during
    computations, the value at the corresponding index in the kernel
    must be set to something different of 0. To desactivate a specific
    neighbor, the corresponding value in the kernel must be set to 0.
    

    \param[out] imout destination image 
    \param[in] imin source image
    \param[in] kernel kernel declaration for the erosion.    
  */
  extern freia_status freia_aipo_erode_6c(freia_data2d *imout, freia_data2d *imin, const int32_t *kernel);



  /*!  
    \brief Dilate imin using kernel declared in 6 connexity and
    store the result in imout
    
    \sa freia_aipo_erode_6c

    \param[out] imout destination image
    \param[in] imin source image
    \param[in] kernel kernel declaration for the erosion.
  */
  extern freia_status freia_aipo_dilate_6c(freia_data2d *imout, freia_data2d *imin, const int32_t *kernel);




  /*!@}*/



#endif

#ifdef __cplusplus
}
#endif
