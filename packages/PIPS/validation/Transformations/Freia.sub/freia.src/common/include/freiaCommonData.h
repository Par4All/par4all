/***************************************************************
 * Author : Christophe Clienti
 * Date : 10/22/2008
 * Copyright (c) 2008 Thales Research & Technologies (FR)
 ***************************************************************/

#ifdef __cplusplus
extern "C" {
#endif


#ifndef __FREIA_COMMON_DATA_H__
#define __FREIA_COMMON_DATA_H__

#include <freiaCommonTypes.h>
  /**
   * @defgroup freia_common_data Bidimensional creation and management functions
   * @ingroup freia_common
   * @{
   */


  /**
     @brief Init fields of freia_data2d instance structure

     Example:
     @code
     freia_data2d imin;
     freia_common_init_data(&imin,
     (freia_ptr) freia_common_alloc(sizeof(uint8_t)*size_x*size_y),
     (freia_ptr*) freia_common_alloc(sizeof(freia_ptr)*size_y),
     sizeof(uint8_t)*8, size_x, size_y);
     (...)
     freia_common_free(imin.raw);
     freia_common_free(imin.row);
     @endcode

     @sa freia_common_create_data

     @param[out] data pointer to a valid instance of freia_data2d
     @param[in] addressBuffer Address of a 1D buffer which can manage all image pixels
     @param[in] rowPtrBuffer Array of row pointers. The array size corresponds to the image height.
     @param[in] bpp number of bit per pixel
     @param[in] width
     @param[in] height 
     @return error code
  */
  extern freia_status freia_common_init_data(freia_data2d *data, freia_ptr addressBuffer, freia_ptr *rowPtrBuffer, 
					     uint32_t bpp, uint32_t width, uint32_t height);


 /**
     @brief Create an instance of freia_data2d structure

     Example:
     @code
     freia_data2d *imin;
     imin = freia_common_create_data(sizeof(uint8_t)*8, size_x, size_y);
     (...)
     freia_common_desctruc_data(imin);
     @endcode

     @sa freia_common_init_data

     @param[in] bpp number of bit per pixel
     @param[in] width
     @param[in] height 
     @return pointer to freia_data2d instance
  */
  extern freia_data2d *freia_common_create_data(uint32_t bpp, uint32_t width, uint32_t height);



  /**
     @brief Initialize an existing instance of freia_data2d structure and share the
     payload with the given freia_data2d
     
     @param[out] data pointer to a valid instance of freia_data2d to be initialized
     @param[in] data_to_share_payload
     @return error code 
  */
  freia_status freia_common_init_data_link(freia_data2d *data, freia_data2d *data_to_share_payload);

  /**
     @brief Create an instance of freia_data2d structure and share the
     payload with the given freia_data2d
     
     @param[in] data_to_share_payload
     @return pointer to freia_data2d instance
  */
   extern freia_data2d* freia_common_create_data_link(freia_data2d *data_to_share_payload);


  
  /**
     @brief it frees allocated buffer in a freia_data2d instance
     @param[in] data pointer to a valid instance of freia_data2d
     @return error code
  */
  extern freia_status freia_common_destruct_buffer(freia_data2d *data);


  /**
     @brief it frees allocated buffer and the freia_data2d instance
     @param[in] data pointer to a valid instance of freia_data2d
     @return error code
  */
  extern freia_status freia_common_destruct_data(freia_data2d *data);


  /**
     @brief it frees freia_data2d instance but not allocated buffers
     @param[in] data pointer to a valid instance of freia_data2d
     @return error code
  */
  freia_status freia_common_destruct_data_link(freia_data2d *image);

  /**
     @brief Set the working area
     @param[in, out] data: pointer to a valid instance of freia_data2d
     @param[in] xStartWa X coordinate of the working area (upper left)
     @param[in] yStartWa Y coordinate of the working area (upper left)
     @param[in] widthWa Working area width
     @param[in] heightWa Working area height
     @return error code
  */
  extern freia_status freia_common_set_wa(freia_data2d *data, uint32_t xStartWa, uint32_t yStartWa, 
					  uint32_t widthWa, uint32_t heightWa);

  /**
     @brief get the working area
     @param[in] data: pointer to a valid instance of freia_data2d
     @param[out] xStartWa X coordinate of the working area (upper left)
     @param[out] yStartWa Y coordinate of the working area (upper left)
     @param[out] widthWa Working area width
     @param[out] heightWa Working area height
     @return error code
  */
  extern freia_status freia_common_get_wa(freia_data2d *data, uint32_t *xStartWa, uint32_t *yStartWa, 
					  uint32_t *widthWa, uint32_t *heightWa);

  /**
     @brief Reset to maximum the working area
     @param[in, out] data: pointer to a valid instance of freia_data2d
     @return error code
  */
  extern freia_status freia_common_reset_wa(freia_data2d *data);


  /**
     @brief Check if given images are window compatible (same working area size but offsets could be different)
    
     No check regarding other aspects of images (size, bpp, ...)
     Each parametrer of the variadic list must be of type freia_data2d. The list must be NULL-terminated

     example : freia_common_check_image_window_compat(imin1, imin2, imin3, NULL);

     @param[in] imin1 first image to test with the variable list of parameters
     @return true or false
  */
  extern bool freia_common_check_image_window_compat(freia_data2d *imin1, ...);


  /**
     @brief Check if given images are size compatible.
    
     No check regarding other aspects of images (working area, bpp, ...)
     Each parametrer of the variadic list must be of type freia_data2d. The list must be NULL-terminated

     example : freia_common_check_image_size_compat(imin1, imin2, imin3, NULL);

     @param[in] imin1 first image to test with the variable list of parameters
     @return true or false
  */
  extern bool freia_common_check_image_size_compat(freia_data2d *imin1, ...);


  /**
     @brief Check if given images are bpp compatible
    
     No check regarding other aspects of images (working area, size, ...)
     Each parametrer of the variadic list must be of type freia_data2d. The list must be NULL-terminated

     example : freia_common_check_image_bpp_compat(imin1, imin2, imin3, NULL);

     @param[in] imin1 first image to test with the variable list of freia_data2d parameters
     @return true or false
  */
  extern bool freia_common_check_image_bpp_compat(freia_data2d *imin1, ...);


  /*
    @brief check if to integer arguments are equals
   */
  extern bool freia_common_check_value_compat(int, int);

  /*
    @brief check that there is indeed an image
   */
  extern bool freia_common_check_image_not_null(freia_data2d *);

  /**
     
     @brief Return a coordinate between [0..dimsize-1]

     If the given coordinate is out of bound, this function will return
     a coordinate as the dimension considered was unfolded
     
     @param[in] coord coordinate value of a specific dimension
     @param[in] dimsize number of element in the dimension
     @return new coordinate between [0..dimsize-1]
  */
  int32_t freia_common_protect_coord(int32_t coord, uint32_t dimsize);


  /**
     @}
  */

  /* temporary(?) fix to compile code assuming that it is correct,
   * so that common checks are always true.
   */
#ifdef FREIA_ASSUME_CORRECT_CODE
#define freia_common_check_image_window_compat(args...) (true)
#define freia_common_check_image_size_compat(args...) (true)
#define freia_common_check_image_bpp_compat(args...) (true)
#define freia_common_check_value_compat(args...) (true)
#define freia_common_check_image_not_null(args...) (true)
#endif /* FREIA_ASSUME_CORRECT_CODE */

#endif



#ifdef __cplusplus
}
#endif
