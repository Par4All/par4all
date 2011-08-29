/***************************************************************
 * Author : Christophe Clienti
 * Date : 10/22/2008
 * Copyright (c) 2008 Thales Research & Technologies (FR)
 ***************************************************************/

#ifdef __cplusplus
extern "C" {
#endif

#ifndef __FREIA_COMMON_TYPES_H__
#define __FREIA_COMMON_TYPES_H__

#include <sys/types.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stddef.h>
#include <stdarg.h>




  /*!
   * \defgroup freia_common_type Specific declaration of types
   * \ingroup freia_common
   * @{
   */

  /*!
   * freia pointer type
   */
  typedef void *    freia_ptr;

  /*!
   * freia error type
   */
  typedef int32_t   freia_status;

  /*!
   * error code enum
   */
  typedef enum {
    FREIA_OK = 0,
    FREIA_INVALID_PARAM = -1,
    FREIA_TRANSMIT_ERROR = -2,
    FREIA_TIMEOUT = -4,
    FREIA_SIZE_ERROR = -8,
    FREIA_NULL_PARAM = -16,
    FREIA_UNDEF_ERROR = -32,
    FREIA_NOT_IMPLEMENTED = -64,
    FREIA_POOL_EXHAUSTED = -128
  }freia_status_enum;


  /*!
   * Bidimensional data manipulated by the user
   */
  typedef struct{
    freia_ptr raw;      /*!< 1D buffer address */
    freia_ptr *row;     /*!< Row pointers */
    uint32_t bpp;       /*!< Number of bit per pixel */
    uint32_t width;     /*!< Data width */
    uint32_t height;    /*!< Data height */
    uint32_t xStartWa;  /*!< X coordinate of the working area (upper left) */
    uint32_t yStartWa;  /*!< Y coordinate of the working area (upper left) */
    uint32_t widthWa;   /*!< Working area width */
    uint32_t heightWa;  /*!< Working area height */
    uint32_t originalWidth ;    /*!< for openCL. original width */
    uint32_t clId ; /*!<OpenCL target extension structure */

  }freia_data2d;


  /*!
   * Structure to manage input or ouput video flow
   */
  typedef struct {
    uint32_t frameindex;   /*!< index of the current frame */
    uint32_t framewidth;   /*!< frames width*/
    uint32_t frameheight;  /*!< frames height */
    uint32_t framebpp;     /*!< frames bit per pixel */
    int32_t  framecount;   /*!< max number of frames, -1 for no limit */
    uint32_t vidchan;      /*!< video channel index */
    bool isinput;          /*!< boolean set to true if instance will be dedicated to video input flow  */
    freia_ptr extend;      /*!< pointer to anything (spare) */
  }freia_dataio;


  /*!@}*/


#endif


#ifdef __cplusplus
}
#endif
