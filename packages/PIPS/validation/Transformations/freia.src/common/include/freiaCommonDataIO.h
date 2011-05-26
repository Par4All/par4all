/***************************************************************
 * Author : Christophe Clienti
 * Date : 10/22/2008
 * Copyright (c) 2008 Thales Research & Technologies (FR)
 ***************************************************************/

#ifdef __cplusplus
extern "C" {
#endif


#ifndef __FREIA_COMMON_DATA_IO_H__
#define __FREIA_COMMON_DATA_IO_H__

#include <freiaCommonTypes.h>
#include <freiaCommonData.h>

  /*!
   * \ingroup freia_common_data
   * @{
   */


  /*!
    \brief Pick a specific uint8_t value in a freia_data2d
    
    You must take care to the bounds of coordinates. No specific
    protection is set up and a core dump could happen with bad
    coordinates

    \param[in] data pointer to a valid instance of freia_data2d
    \param[in] x x coordinate of the value to pick
    \param[in] y y coordinate of the value to pick
    \return value picked
  */
  static __inline__ uint8_t freia_common_get_uint8_t(freia_data2d *data, uint32_t x, uint32_t y) {
    uint8_t *row;
    row = (uint8_t *) (data->row[y+data->yStartWa]);
    return row[x+data->xStartWa];
  }

  /*!
    \brief Pick a specific int16_t value in a freia_data2d
    
    You must take care to the bounds of coordinates. No specific
    protection is set up and a core dump could happen with bad
    coordinates

    \param[in] data pointer to a valid instance of freia_data2d
    \param[in] x x coordinate of the value to pick
    \param[in] y y coordinate of the value to pick
    \return value picked
  */
  static __inline__ int16_t freia_common_get_int16_t(freia_data2d *data, uint32_t x, uint32_t y) {
    int16_t *row;
    row = (int16_t *) (data->row[y+data->yStartWa]);
    return row[x+data->xStartWa];
  }

  /*!
    \brief Pick a specific value in a freia_data2d
    
    You must take care to the bounds of coordinates. No specific
    protection is set up and a core dump could happen with bad
    coordinates

    \param[in] data pointer to a valid instance of freia_data2d
    \param[in] x x coordinate of the value to pick
    \param[in] y y coordinate of the value to pick
    \return value picked
  */
  static __inline__ int32_t freia_common_get(freia_data2d *data, uint32_t x, uint32_t y) {
    if(data->bpp==8) return freia_common_get_uint8_t(data,x,y);
    else return freia_common_get_int16_t(data,x,y);
  }

  /*!
    \brief Set a specific uint8_t value in a freia_data2d
    
    You must take care to the bounds of coordinates. No specific
    protection is set up and a core dump could happen with bad
    coordinates

    \param[in,out] data: pointer to a valid instance of freia_data2d
    \param[in] x x coordinate of the value to pick
    \param[in] y y coordinate of the value to pick
    \param[in] value uint8_t value to set in the data
    \return void
  */
  static __inline__ void freia_common_set_uint8_t(freia_data2d *data, uint32_t x, uint32_t y, uint8_t value) {
    uint8_t *row;
    row = (uint8_t *) (data->row[y+data->yStartWa]);
    row[x+data->xStartWa] = value;
  }

  /*!
    \brief Set a specific int16_t value in a freia_data2d

    You must take care to the bounds of coordinates. No specific
    protection is set up and a core dump could happen with bad
    coordinates

    \param[in,out] data: pointer to a valid instance of freia_data2d
    \param[in] x x coordinate of the value to pick
    \param[in] y y coordinate of the value to pick
    \param[in] value int16_t value to set in the data
    \return void
  */
  static __inline__ void freia_common_set_int16_t(freia_data2d *data, uint32_t x, uint32_t y, int16_t value) {
    int16_t *row;
    row = (int16_t *) (data->row[y+data->yStartWa]);
    row[x+data->xStartWa] = value;
  }


  /*!
    \brief Set a specific value in a freia_data2d
    
    You must take care to the bounds of coordinates. No specific
    protection is set up and a core dump could happen with bad
    coordinates

    \param[in,out] data: pointer to a valid instance of freia_data2d
    \param[in] x x coordinate of the value to pick
    \param[in] y y coordinate of the value to pick
    \param[in] value uint8_t value to set in the data
    \return void
  */
  static __inline__ void freia_common_set(freia_data2d *data, uint32_t x, uint32_t y, int32_t value) {
    if(data->bpp==8) freia_common_set_uint8_t(data,x,y,(uint8_t) value);
    else freia_common_set_int16_t(data,x,y,(int16_t) value);
  }

  /*!
    \brief Set a specific uint8_t value in a freia_data2d
    
    If coordinates are out of the current working area, the function
    will not write any values

    \param[in,out] data: pointer to a valid instance of freia_data2d
    \param[in] x x coordinate of the value to pick
    \param[in] y y coordinate of the value to pick
    \param[in] value uint8_t value to set in the data
    \return void
  */
  static __inline__ void freia_common_setb_uint8_t(freia_data2d *data, int32_t x, int32_t y, uint8_t value) {
    if(y>=data->heightWa) return;
    if(x>=data->widthWa) return;
    if(y<0) return;
    if(x<0) return;
    freia_common_set_uint8_t(data,x,y,value);
  }

  /*!
    \brief Set a specific int16_t value in a freia_data2d
    
    The access out of bounds of coordinates is managed by simulating an
    unfolding of the data using some symmetry properties
 
    \param[in,out] data: pointer to a valid instance of freia_data2d
    \param[in] x x coordinate of the value to pick
    \param[in] y y coordinate of the value to pick
    \param[in] value int16_t value to set in the data
    \return void
  */
  static __inline__ void freia_common_setb_int16_t(freia_data2d *data, int32_t x, int32_t y, int16_t value) {
    if(y>=data->heightWa) return;
    if(x>=data->widthWa) return;
    if(y<0) return;
    if(x<0) return;    
    freia_common_set_int16_t(data,x,y,value);
  }


  /*!
    \brief Set a specific value in a freia_data2d
    
    The access out of bounds of coordinates is managed by simulating an
    unfolding of the data using some symmetry properties
 
    \param[in,out] data: pointer to a valid instance of freia_data2d
    \param[in] x x coordinate of the value to pick
    \param[in] y y coordinate of the value to pick
    \param[in] value int16_t value to set in the data
    \return void
  */
  static __inline__ void freia_common_setb(freia_data2d *data, int32_t x, int32_t y, int16_t value) {
    if(data->bpp==8) freia_common_setb_uint8_t(data,x,y,value);
    else freia_common_setb_int16_t(data,x,y,value);
  }





  /*!
    \brief Pick a specific uint8_t value in a freia_data2d
    
    The access out of bounds of coordinates is managed by simulating an
    unfolding of the data using some symmetry properties

    \param[in] data pointer to a valid instance of freia_data2d
    \param[in] x x coordinate of the value to pick
    \param[in] y y coordinate of the value to pick
    \return value picked
  */
  extern uint8_t freia_common_getu_uint8_t(freia_data2d *data, int32_t x, int32_t y);

  /*!
    \brief Pick a specific int16_t value in a freia_data2d
    
    The access out of bounds of coordinates is managed by simulating an
    unfolding of the data using some symmetry properties
 
    \param[in] data pointer to a valid instance of freia_data2d
    \param[in] x x coordinate of the value to pick
    \param[in] y y coordinate of the value to pick
    \return value picked
  */
  extern int16_t freia_common_getu_int16_t(freia_data2d *data, int32_t x, int32_t y);

  /*!
    \brief Pick a specific value in a freia_data2d
    
    The access out of bounds of coordinates is managed by simulating an
    unfolding of the data using some symmetry properties
 
    \param[in] data pointer to a valid instance of freia_data2d
    \param[in] x x coordinate of the value to pick
    \param[in] y y coordinate of the value to pick
    \return value picked
  */
  extern int32_t freia_common_getu(freia_data2d *data, int32_t x, int32_t y);

  /*!
    \brief Set a specific uint8_t value in a freia_data2d
    
    The access out of bounds of coordinates is managed by simulating an
    unfolding of the data using some symmetry properties
 
    \param[in,out] data: pointer to a valid instance of freia_data2d
    \param[in] x x coordinate of the value to pick
    \param[in] y y coordinate of the value to pick
    \param[in] value uint8_t value to set in the data
    \return void
  */
  extern void freia_common_setu_uint8_t(freia_data2d *data, int32_t x, int32_t y, uint8_t value);

  /*!
    \brief Set a specific int16_t value in a freia_data2d
    
    The access out of bounds of coordinates is managed by simulating an
    unfolding of the data using some symmetry properties
 
    \param[in,out] data: pointer to a valid instance of freia_data2d
    \param[in] x x coordinate of the value to pick
    \param[in] y y coordinate of the value to pick
    \param[in] value int16_t value to set in the data
    \return void
  */
  extern void freia_common_setu_int16_t(freia_data2d *data, int32_t x, int32_t y, int16_t value);

  /*!
    \brief Set a specific value in a freia_data2d
    
    The access out of bounds of coordinates is managed by simulating an
    unfolding of the data using some symmetry properties
 
    \param[in,out] data: pointer to a valid instance of freia_data2d
    \param[in] x x coordinate of the value to pick
    \param[in] y y coordinate of the value to pick
    \param[in] value int16_t value to set in the data
    \return void
  */
  extern void freia_common_setu(freia_data2d *data, int32_t x, int32_t y, int32_t value);



  /*!
    \brief Pick a specific uint8_t value in a freia_data2d
    
    If coordinates are out of the current working area, the function
    will return the given border value

    \param[in] data pointer to a valid instance of freia_data2d
    \param[in] x x coordinate of the value to pick
    \param[in] y y coordinate of the value to pick
    \param[in] border border value to return when a bad access is done
    \return value picked or border
  */
  extern uint8_t freia_common_getb_uint8_t(freia_data2d *data, int32_t x, int32_t y, uint8_t border);

  /*!
    \brief Pick a specific int16_t value in a freia_data2d
    
    If coordinates are out of the current working area, the function
    will return the given border value

    \param[in] data pointer to a valid instance of freia_data2d
    \param[in] x x coordinate of the value to pick
    \param[in] y y coordinate of the value to pick
    \param[in] border border value to return when a bad access is done
    \return value picked or border
  */
  extern int16_t freia_common_getb_int16_t(freia_data2d *data, int32_t x, int32_t y, int16_t border);

 
  /*!
    \brief Pick a specific value in a freia_data2d
    
    If coordinates are out of the current working area, the function
    will return the given border value

    \param[in] data pointer to a valid instance of freia_data2d
    \param[in] x x coordinate of the value to pick
    \param[in] y y coordinate of the value to pick
    \param[in] border border value to return when a bad access is done
    \return value picked or border
  */
  extern int32_t freia_common_getb(freia_data2d *data, int32_t x, int32_t y, int32_t border);

 
  /*!
    \brief Export freia_data2d content to a uint8_t buffer.
    
    Working area is taken into account

    \param[out] memarea
    \param[in] data 
    \return freia_status
  */
  extern freia_status freia_common_export_uint8_t(uint8_t *memarea, freia_data2d *data);



  /*!
    \brief Import uint8_t buffer to freia_data2d
    
    Working area is taken into account

    \param[out] data 
    \param[in] memarea
    \return freia_status
  */
  extern freia_status freia_common_import_uint8_t(freia_data2d *data, uint8_t *memarea);

  /*!
    \brief Export freia_data2d content to a int16_t buffer.
    
    Working area is taken into account

    \param[out] memarea
    \param[in] data 
    \return freia_status
  */
  extern freia_status freia_common_export_int16_t(int16_t *memarea, freia_data2d *data);

  /*!
    \brief Export freia_data2d content to a buffer.
    
    Working area is taken into account

    \param[out] memarea
    \param[in] data 
    \return freia_status
  */
  extern freia_status freia_common_export(freia_ptr memarea, freia_data2d *data);



  /*!
    \brief Import int16_t buffer to freia_data2d
    
    Working area is taken into account

    \param[out] data 
    \param[in] memarea
    \return freia_status
  */
  extern freia_status freia_common_import_int16_t(freia_data2d *data, int16_t *memarea);


  /*!
    \brief Import buffer to freia_data2d
    
    Working area is taken into account

    \param[out] data 
    \param[in] memarea
    \return freia_status
  */
  extern freia_status freia_common_import(freia_data2d *data, freia_ptr memarea);


  /*!
    \brief Print the image on stdio

    \param[in] image
    \return freia_status
  */
  extern freia_status freia_common_print_data(freia_data2d *image);


  /*!@}*/


#endif



#ifdef __cplusplus
}
#endif
