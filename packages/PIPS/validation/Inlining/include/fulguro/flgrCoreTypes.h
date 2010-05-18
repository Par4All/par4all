/****************************************************************
 * Fulguro
 * Copyright (C) 2004 Christophe Clienti
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 ***************************************************************/

#ifdef __cplusplus
extern "C" {
#endif


#ifndef __FLGR_CORE_TYPES_H
#define __FLGR_CORE_TYPES_H

#include <flgrCoreDll.h>
#include <stdint.h>

  /*!
   * \addtogroup group_fulguro_core
   * @{
   */

  /*!
   *  Errors flags definition
   */
  typedef enum {
    /*! Function worked correctly */
    FLGR_RET_OK                    = 0,
    /*! Data type specified is not recoginzed in the function */
    FLGR_RET_TYPE_UNKNOWN          = -1,
    /*! Function could not work with images with different data types */
    FLGR_RET_TYPE_DIFFERENT        = -2,
    /*! Function could not work with images with different sizes */
    FLGR_RET_SIZE_ERROR            = -3,
    /*! Memory could not be allocated */
    FLGR_RET_ALLOCATION_ERROR      = -4,
    /*! Connexity specified is not recognized by the function */
    FLGR_RET_CONNEXITY_UNKNOWN     = -5,
    /*! The parameter is not known */
    FLGR_RET_PARAM_UNKNOWN         = -6,
    /*! The function is not implemented */
    FLGR_RET_NOT_IMPLEMENTED       = -7,
    /*! For all others*/
    FLGR_RET_UNDEFINED_ERROR       = -8,
    /*  Non-initialized objects are used */
    FLGR_RET_NULL_OBJECT           = -9,
    /*  Sample per pixel (spp) different */
    FLGR_RET_VECTOR_SIZE_DIFFERENT = -10,
    /*  Vector size is not valid */
    FLGR_RET_VECTOR_SIZE_ERROR     = -11,
    /*  function parameter is not valid */
    FLGR_RET_PARAM_ERROR           = -12,

  }FLGR_Ret;

  /*!
   *  Shape flags definition
   */
  typedef enum {
    FLGR_LINE,       /*!< 1D Shape*/
    FLGR_RECT,       /*!< 2D, Filled Rectangle */
    FLGR_HEX,        /*!< 2D, Hexagon */
    FLGR_DISC,       /*!< 2D, Disc */
    FLGR_ELLIPSE,    /*!< 2D, Filled Ellipse */
    FLGR_CROSS,      /*!< 2D, Cross + */
    FLGR_CROSSX,     /*!< 2D, Cross X */
    FLGR_SLASH,      /*!< 2D, line / */
    FLGR_BSLASH,     /*!< 2D, line \\ */
    FLGR_DIAMOND,    /*!< 2D, Diamond */
    FLGR_OCTAGON,    /*!< 2D, Octagon */
    FLGR_USER_SHAPE, /*!< User shape */
    FLGR_NO_SHAPE,   /*!< Shape not applicable */
  }FLGR_Shape;

  /*!
   *  Connexity Mesh flags definition
   */
  typedef enum {
    FLGR_4_CONNEX  = 4, /*!< Flag for 4-connex operation (2D) */
    FLGR_6_CONNEX  = 6, /*!< Flag for 6-connex operation (2D) */
    FLGR_8_CONNEX  = 8, /*!< Flag for 8-connex operation (2D) */
    FLGR_NO_CONNEX = 0 /*!< Flag No Connexity applicable */
  }FLGR_Connexity;

  /*!
   *  Data Types flags definition
   */
  typedef enum {
    FLGR_BIT,         /*!< Flag for BIT type (one value is stored in one bit) */
    FLGR_UINT8,       /*!< Flag for UINT8 type */
    FLGR_UINT16,      /*!< Flag for UINT16 type */
    FLGR_UINT32,      /*!< Flag for UINT32 type */
    FLGR_UINT64,      /*!< Flag for UINT64 type */
    FLGR_INT8,        /*!< Flag for INT8 type */
    FLGR_INT16,       /*!< Flag for INT16 type */
    FLGR_INT32,       /*!< Flag for INT32 type */
    FLGR_INT64,       /*!< Flag for INT64 type */
    FLGR_FLOAT32,     /*!< Flag for FLOAT32 type */
    FLGR_FLOAT64     /*!< Flag for FLOAT64 type */
  }FLGR_Type;


  typedef enum {
    FLGR_TEST_INF     = 0,
    FLGR_TEST_INFEQ   = 1,
    FLGR_TEST_EQ      = 2,
    FLGR_TEST_SUPEQ   = 3,
    FLGR_TEST_SUP     = 4,
    FLGR_TEST_DIFF    = 5,
    FLGR_TEST_UNKNOWN = 6
  }FLGR_Test;

  /*!
   *  Symetrize Neighborhood flags definition
   */
  typedef enum {
    /*! Neihborhood Definition Symetrization Flag */
    FLGR_NHB_SYM              = 0,
    /*! Neihborhood Definition No Symetrization Flag */
    FLGR_NHB_NO_SYM           = 1,
  }FLGR_Sym;



  /*!
   *  Optimization availables for kernel-based operations
   */
  typedef enum {
    FLGR_MORPHO_OPTIM_SQUARE,     /*!< Flag for square kernel optimization (8-connex) */
    FLGR_MORPHO_OPTIM_CROSS,      /*!< Flag for cross kernel optimization (4-connex) */
    FLGR_MORPHO_OPTIM_HEXAGON,    /*!< Flag for hexagonal kernel optimization (6-connex) */
    FLGR_MORPHO_OPTIM_RECTANGLE,  /*!< Flag for rectangle kernel optimization (8-connex) */
    FLGR_MORPHO_OPTIM_OCTAGON,    /*!< Flag for octagon kernel optimization (8-connex) */
    FLGR_MORPHO_OPTIM_DIAMOND,    /*!< Flag for rhombus kernel optimization (8-connex) */
    FLGR_MORPHO_OPTIM_NONE        /*!< Flag for no specific kernel optimization */
  }FLGR_MorphoOptim;

  //type for returning errors, ok



  //type for some measure method ...
#if defined(_MSC_VER)
  typedef unsigned __int64      fgUINT64
  typedef __int64               fgINT64
#else
  typedef unsigned long long    fgUINT64;
  typedef long long             fgINT64;
#endif

  //types for tdata2d values
  typedef unsigned char         fgUINT8;   //!< 8 bits unsigned
  typedef unsigned short int    fgUINT16;  //!< 16 bits unsigned
  typedef unsigned int          fgUINT32;  //!< 32 bits unsigned
  typedef signed char           fgINT8;    //!< 8 bits signed
  typedef signed short int      fgINT16;   //!< 16 bits signed
  typedef signed int            fgINT32;   //!< 32 bits signed
  typedef float                 fgFLOAT32; //!< Simple Float
  typedef double                fgFLOAT64; //!< Double Float


#if defined(x86_64) || defined(__powerpc64__)
  typedef fgUINT64              fgBIT;     //!< bit type. Pixels are packed by 32 or 64 depending on the processor type (32 bits or 64 bits)
#else 
  typedef fgUINT32              fgBIT;     //!< bit type. Pixels are packed by 32 or 64 depending on the processor type (32 bits or 64 bits)
#endif

  //type for boolean
  typedef fgUINT8               fgBOOL;

#define FLGR_PVOID_VAL(dtype,val)    (*((dtype *) (val)))
#define FLGR_PVOID_PTR(dtype,val)    ((dtype*) (val))

#define FLGR_TRUE  1
#define FLGR_FALSE 0

#define MAX_fgBIT (1)
#define MIN_fgBIT (0)


#define MAX_fgUINT8 UINT8_MAX
#define MIN_fgUINT8 0

#define MAX_fgUINT16 UINT16_MAX
#define MIN_fgUINT16 0

#define MAX_fgUINT32 UINT32_MAX
#define MIN_fgUINT32 0

#define MAX_fgUINT64 UINT64_MAX
#define MIN_fgUINT64 0

#define MAX_fgINT8 INT8_MAX
#define MIN_fgINT8 INT8_MIN

#define MAX_fgINT16 INT16_MAX
#define MIN_fgINT16 INT16_MIN

#define MAX_fgINT32 INT32_MAX
#define MIN_fgINT32 INT32_MIN

#define MAX_fgINT64 INT64_MAX
#define MIN_fgINT64 INT64_MIN

#define MAX_fgFLOAT32 3.4e37
#define MIN_fgFLOAT32 -3.4e-37

#define MAX_fgFLOAT64 1.7e307
#define MIN_fgFLOAT64 -1.7e-307

#define FLGR_PI ((double) 3.141592653589793238462643383279502884195)

#define FLGR_MIN(a,b) ( (a) < (b)  ? (a) : (b) )
#define FLGR_MAX(a,b) ( (a) < (b)  ? (b) : (a) )

  EXPORT_LIB FLGR_Ret flgr_is_vector_type_valid(FLGR_Type type);
  EXPORT_LIB int flgr_get_bps_from_type(FLGR_Type type);
  EXPORT_LIB int flgr_get_sizeof(FLGR_Type type);
  EXPORT_LIB FLGR_Ret flgr_is_data_type_valid(FLGR_Type type);
  EXPORT_LIB char *flgr_get_type_string(FLGR_Type type);
  EXPORT_LIB char *flgr_get_shape_string(FLGR_Shape shape);
  EXPORT_LIB FLGR_Type flgr_get_type_from_string(char *type);


  //! @}


#endif

#ifdef __cplusplus
}
#endif

