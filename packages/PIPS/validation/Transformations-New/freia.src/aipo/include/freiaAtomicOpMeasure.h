/***************************************************************
 * Author : Christophe Clienti
 * Date : 10/22/2008
 * Copyright (c) 2008 Thales Research & Technologies (FR)
 ***************************************************************/

#ifdef __cplusplus
extern "C" {
#endif


#ifndef __FREIA_ATOMIC_OP_MEASURE_H__
#define __FREIA_ATOMIC_OP_MEASURE_H__

#include <freiaCommonTypes.h>



  /*!
   * \defgroup freia_aipo_measure Measure basic parameters
   * \ingroup freia_aipo
   * @{
   */

  /*!  
    \brief Measure global minimum of an image
    
    \param[in] image source image
    \param[out] min global min
    \return freia_status
  */
  extern freia_status freia_aipo_global_min(freia_data2d *image, int32_t *min);

  /*!  
    \brief Measure global coordinates of minimum of an image
    
    \param[in] image source image
    \param[out] min global min
    \param[out] xmin x coordinate of the minimum
    \param[out] ymin y coordinate of the minimum
    \return freia_status
  */
  extern freia_status freia_aipo_global_min_coord(freia_data2d *image, int32_t *min, uint32_t *xmin, uint32_t *ymin);

  /*!  
    \brief Measure global maximum of an image
    
    \param[in] image source image
    \param[out] max global max
    \return freia_status
  */
  extern freia_status freia_aipo_global_max(freia_data2d *image, int32_t *max);

  /*!  
    \brief Measure global coordinates of maximum of an image
    
    \param[in] image source image
    \param[out] max global max
    \param[out] xmax x coordinate of the maximum
    \param[out] ymax y coordinate of the maximum
    \return freia_status
  */
  extern freia_status freia_aipo_global_max_coord(freia_data2d *image, int32_t *max, uint32_t *xmax, uint32_t *ymax);

  /*!  
    \brief Measure global volume of an image
    
    \param[in] image source image
    \param[out] vol global volume
    \return freia_status
  */
  extern freia_status freia_aipo_global_vol(freia_data2d *image, int32_t *vol);


  /*!  
    \brief Sum of Absolute Differences sum(|I1-I2|)
    
    \param[in] imin1 source 1
    \param[in] imin2 source 2
    \param[out] sad result of the sum of absolute differences
    \return SAD
  */
  extern freia_status freia_aipo_global_sad(freia_data2d *imin1, freia_data2d *imin2, uint32_t *sad);




  /*!@}*/


#endif

#ifdef __cplusplus
}
#endif
