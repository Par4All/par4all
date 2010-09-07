/***************************************************************
 * Author : Christophe Clienti
 * Date : 10/22/2008
 * Copyright (c) 2008 Thales Research & Technologies (FR)
 ***************************************************************/

#ifdef __cplusplus
extern "C" {
#endif


#ifndef __FREIA_COMMON_DRAW_H__
#define __FREIA_COMMON_DRAW_H__

#include <freiaCommonTypes.h>


  /*!
   * \defgroup freia_common_draw Functions to draw shapes
   * \ingroup freia_common
   * @{
   */



  /*!  
    \brief Draw a line with an arbitrary angle using the bresenham algorithm

    The line is drown between two points which coordinates are P1(x1,y1) P2(x2,y2)

    \param[in,out] image
    \param[in] x1 x coordinate of the first point
    \param[in] y1 y coordinate of the first point
    \param[in] x2 x coordinate of the second point
    \param[in] y2 y coordinate of the second point
    \param[in] color color of the line
    \return error code
  */
  extern freia_status freia_common_draw_line(freia_data2d *image, int32_t x1, int32_t y1, int32_t x2, int32_t y2, int32_t color);

  /*!  
    \brief Draw a rectangle

    The rectangle is drown between two points which coordinates are P1(x1,y1) P2(x2,y2)

    \param[in,out] image
    \param[in] x1 x coordinate of the first point
    \param[in] y1 y coordinate of the first point
    \param[in] x2 x coordinate of the second point
    \param[in] y2 y coordinate of the second point
    \param[in] color color of the rectangle
   \return error code
  */
  extern freia_status freia_common_draw_rect(freia_data2d *image, int32_t x1, int32_t y1, int32_t x2, int32_t y2, int32_t color);

  /*!  
    \brief Draw a filled rectangle

    The box is drown between two points which coordinates are P1(x1,y1) P2(x2,y2)

    \param[in,out] image
    \param[in] x1 x coordinate of the first point
    \param[in] y1 y coordinate of the first point
    \param[in] x2 x coordinate of the second point
    \param[in] y2 y coordinate of the second point
    \param[in] color color of the box
   \return error code
  */
  extern freia_status freia_common_draw_filled_rect(freia_data2d *image, int32_t x1, int32_t y1, int32_t x2, int32_t y2, int32_t color);

  /*!  
    \brief Draw a circle

    The circle is drown by specifying a center (xc,yc) and a radius

    \param[in,out] image
    \param[in] xc x coordinate of the circle center
    \param[in] yc y coordinate of the circle center
    \param[in] radius circle radius
   \param[in] color color of the circle
   \return error code
  */
  extern freia_status freia_common_draw_circle(freia_data2d *image, int32_t xc, int32_t yc, int32_t radius, int32_t color);

  /*!  
    \brief Draw a disc

    The disc is drown by specifying a center (xc,yc) and a radius

    \param[in,out] image
    \param[in] xc x coordinate of the disc center
    \param[in] yc y coordinate of the disc center
    \param[in] radius disc radius
   \param[in] color color of the disc
   \return error code
  */
  extern freia_status freia_common_draw_disc(freia_data2d *image, int32_t xc, int32_t yc, int32_t radius, int32_t color);



  /*!@}*/


#endif

#ifdef __cplusplus
}
#endif
