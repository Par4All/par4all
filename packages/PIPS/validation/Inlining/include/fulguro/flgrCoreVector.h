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


#ifndef __FLGR_CORE_VECTOR_H
#define __FLGR_CORE_VECTOR_H

#include <flgrCoreDll.h>
#include <flgrCoreTypes.h>
#include <flgrCoreErrors.h>

/*!
 * \defgroup group_fulguro_core_vector Vector manipulations
 * \ingroup group_fulguro_core
 * \brief Functions to create and manipulate Vector
 * @{
 */

  /*! 
   * Vector structure
   */
  typedef struct {
    int spp;           //!< Number of elements
    int bps;           //!< Number of bits per pixels
    FLGR_Type type;    //!< Data type of elements
    void *array;       //!< Array of elements
  } FLGR_Vector;


  typedef int (*FLGR_VectorCompare) (FLGR_Vector *vec1, FLGR_Vector *vec2); 

  //! @}

  

  EXPORT_LIB FLGR_Vector *flgr_vector_create(int spp, FLGR_Type type);

  EXPORT_LIB FLGR_Ret flgr_vector_destroy(FLGR_Vector *vct);

  EXPORT_LIB FLGR_Ret flgr_vector_get_element(FLGR_Vector *vct, int index, void *value);
  EXPORT_LIB FLGR_Ret flgr_vector_set_element(FLGR_Vector *vct, int index, void *value);

  EXPORT_LIB FLGR_Ret flgr_vector_import_raw(FLGR_Vector *vctdest, void *raw_src);
  EXPORT_LIB FLGR_Ret flgr_vector_import_string(FLGR_Vector *vctdest, char *string_src);
  EXPORT_LIB FLGR_Ret flgr_vector_export_raw(void *raw_dest, FLGR_Vector *vctsrc);

  EXPORT_LIB FLGR_Ret flgr_vector_populate_from_scalar(FLGR_Vector *vctdest, void *scalar);
  EXPORT_LIB FLGR_Ret flgr_vector_populate_from_string(FLGR_Vector *vctdest, char *string);

  EXPORT_LIB FLGR_Ret flgr_vector_is_type_valid(FLGR_Vector *vct);

  EXPORT_LIB FLGR_Ret flgr_vector_clear(FLGR_Vector *vct);

  EXPORT_LIB int flgr_vector_equal(FLGR_Vector *vct1, FLGR_Vector *vct2);
  EXPORT_LIB int flgr_vector_is_zero(FLGR_Vector *vct);

  EXPORT_LIB FLGR_Ret flgr_vector_is_same_type(FLGR_Vector *vct1, FLGR_Vector *vct2);
  EXPORT_LIB FLGR_Ret flgr_vector_is_same_spp(FLGR_Vector *vct1, FLGR_Vector *vct2);

  EXPORT_LIB FLGR_Ret flgr_vector_copy(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);

  EXPORT_LIB FLGR_Ret flgr_vector_revert_element(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  



  EXPORT_LIB void flgr_vector_revert_element_fgBIT(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_revert_element_fgUINT8(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_revert_element_fgUINT16(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_revert_element_fgUINT32(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_revert_element_fgUINT64(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_revert_element_fgINT8(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_revert_element_fgINT16(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_revert_element_fgINT32(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_revert_element_fgINT64(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_revert_element_fgFLOAT32(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_revert_element_fgFLOAT64(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);

  EXPORT_LIB void flgr_vector_import_raw_fgBIT(FLGR_Vector *vctdest, void *raw_src);
  EXPORT_LIB void flgr_vector_import_raw_fgUINT8(FLGR_Vector *vctdest, void *raw_src);
  EXPORT_LIB void flgr_vector_import_raw_fgUINT16(FLGR_Vector *vctdest, void *raw_src);
  EXPORT_LIB void flgr_vector_import_raw_fgUINT32(FLGR_Vector *vctdest, void *raw_src);
  EXPORT_LIB void flgr_vector_import_raw_fgUINT64(FLGR_Vector *vctdest, void *raw_src);
  EXPORT_LIB void flgr_vector_import_raw_fgINT8(FLGR_Vector *vctdest, void *raw_src);
  EXPORT_LIB void flgr_vector_import_raw_fgINT16(FLGR_Vector *vctdest, void *raw_src);
  EXPORT_LIB void flgr_vector_import_raw_fgINT32(FLGR_Vector *vctdest, void *raw_src);
  EXPORT_LIB void flgr_vector_import_raw_fgINT64(FLGR_Vector *vctdest, void *raw_src);
  EXPORT_LIB void flgr_vector_import_raw_fgFLOAT32(FLGR_Vector *vctdest, void *raw_src);
  EXPORT_LIB void flgr_vector_import_raw_fgFLOAT64(FLGR_Vector *vctdest, void *raw_src);

  EXPORT_LIB void flgr_vector_import_string_fgBIT(FLGR_Vector *vctdest, char *string_src);
  EXPORT_LIB void flgr_vector_import_string_fgUINT8(FLGR_Vector *vctdest, char *string_src);
  EXPORT_LIB void flgr_vector_import_string_fgUINT16(FLGR_Vector *vctdest, char *string_src);
  EXPORT_LIB void flgr_vector_import_string_fgUINT32(FLGR_Vector *vctdest, char *string_src);
  EXPORT_LIB void flgr_vector_import_string_fgUINT64(FLGR_Vector *vctdest, char *string_src);
  EXPORT_LIB void flgr_vector_import_string_fgINT8(FLGR_Vector *vctdest, char *string_src);
  EXPORT_LIB void flgr_vector_import_string_fgINT16(FLGR_Vector *vctdest, char *string_src);
  EXPORT_LIB void flgr_vector_import_string_fgINT32(FLGR_Vector *vctdest, char *string_src);
  EXPORT_LIB void flgr_vector_import_string_fgINT64(FLGR_Vector *vctdest, char *string_src);
  EXPORT_LIB void flgr_vector_import_string_fgFLOAT32(FLGR_Vector *vctdest, char *string_src);
  EXPORT_LIB void flgr_vector_import_string_fgFLOAT64(FLGR_Vector *vctdest, char *string_src);

  EXPORT_LIB void flgr_vector_export_raw_fgBIT(void *raw_dest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_export_raw_fgUINT8(void *raw_dest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_export_raw_fgUINT16(void *raw_dest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_export_raw_fgUINT32(void *raw_dest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_export_raw_fgUINT64(void *raw_dest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_export_raw_fgINT8(void *raw_dest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_export_raw_fgINT16(void *raw_dest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_export_raw_fgINT32(void *raw_dest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_export_raw_fgINT64(void *raw_dest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_export_raw_fgFLOAT32(void *raw_dest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_export_raw_fgFLOAT64(void *raw_dest, FLGR_Vector *vctsrc);

  EXPORT_LIB int flgr_vector_equal_fgBIT(FLGR_Vector *vct1, FLGR_Vector *vct2);
  EXPORT_LIB int flgr_vector_equal_fgUINT8(FLGR_Vector *vct1, FLGR_Vector *vct2);
  EXPORT_LIB int flgr_vector_equal_fgUINT16(FLGR_Vector *vct1, FLGR_Vector *vct2);
  EXPORT_LIB int flgr_vector_equal_fgUINT32(FLGR_Vector *vct1, FLGR_Vector *vct2);
  EXPORT_LIB int flgr_vector_equal_fgUINT64(FLGR_Vector *vct1, FLGR_Vector *vct2);
  EXPORT_LIB int flgr_vector_equal_fgINT8(FLGR_Vector *vct1, FLGR_Vector *vct2);
  EXPORT_LIB int flgr_vector_equal_fgINT16(FLGR_Vector *vct1, FLGR_Vector *vct2);
  EXPORT_LIB int flgr_vector_equal_fgINT32(FLGR_Vector *vct1, FLGR_Vector *vct2);
  EXPORT_LIB int flgr_vector_equal_fgINT64(FLGR_Vector *vct1, FLGR_Vector *vct2);
  EXPORT_LIB int flgr_vector_equal_fgFLOAT32(FLGR_Vector *vct1, FLGR_Vector *vct2);
  EXPORT_LIB int flgr_vector_equal_fgFLOAT64(FLGR_Vector *vct1, FLGR_Vector *vct2);

  EXPORT_LIB int flgr_vector_is_zero_fgBIT(FLGR_Vector *vct);
  EXPORT_LIB int flgr_vector_is_zero_fgUINT8(FLGR_Vector *vct);
  EXPORT_LIB int flgr_vector_is_zero_fgUINT16(FLGR_Vector *vct);
  EXPORT_LIB int flgr_vector_is_zero_fgUINT32(FLGR_Vector *vct);
  EXPORT_LIB int flgr_vector_is_zero_fgUINT64(FLGR_Vector *vct);
  EXPORT_LIB int flgr_vector_is_zero_fgINT8(FLGR_Vector *vct);
  EXPORT_LIB int flgr_vector_is_zero_fgINT16(FLGR_Vector *vct);
  EXPORT_LIB int flgr_vector_is_zero_fgINT32(FLGR_Vector *vct);
  EXPORT_LIB int flgr_vector_is_zero_fgINT64(FLGR_Vector *vct);
  EXPORT_LIB int flgr_vector_is_zero_fgFLOAT32(FLGR_Vector *vct);
  EXPORT_LIB int flgr_vector_is_zero_fgFLOAT64(FLGR_Vector *vct);

  EXPORT_LIB int flgr_vector_is_type_fgBIT(FLGR_Vector *vct);
  EXPORT_LIB int flgr_vector_is_type_fgUINT8(FLGR_Vector *vct);
  EXPORT_LIB int flgr_vector_is_type_fgUINT16(FLGR_Vector *vct);
  EXPORT_LIB int flgr_vector_is_type_fgUINT32(FLGR_Vector *vct);
  EXPORT_LIB int flgr_vector_is_type_fgUINT64(FLGR_Vector *vct);
  EXPORT_LIB int flgr_vector_is_type_fgINT8(FLGR_Vector *vct);
  EXPORT_LIB int flgr_vector_is_type_fgINT16(FLGR_Vector *vct);
  EXPORT_LIB int flgr_vector_is_type_fgINT32(FLGR_Vector *vct);
  EXPORT_LIB int flgr_vector_is_type_fgINT64(FLGR_Vector *vct);
  EXPORT_LIB int flgr_vector_is_type_fgFLOAT32(FLGR_Vector *vct);
  EXPORT_LIB int flgr_vector_is_type_fgFLOAT64(FLGR_Vector *vct);

  EXPORT_LIB void flgr_vector_populate_from_scalar_fgBIT(FLGR_Vector *vctdest, fgBIT scalar);
  EXPORT_LIB void flgr_vector_populate_from_scalar_fgUINT8(FLGR_Vector *vctdest, fgUINT8 scalar);
  EXPORT_LIB void flgr_vector_populate_from_scalar_fgUINT16(FLGR_Vector *vctdest, fgUINT16 scalar);
  EXPORT_LIB void flgr_vector_populate_from_scalar_fgUINT32(FLGR_Vector *vctdest, fgUINT32 scalar);
  EXPORT_LIB void flgr_vector_populate_from_scalar_fgUINT64(FLGR_Vector *vctdest, fgUINT64 scalar);
  EXPORT_LIB void flgr_vector_populate_from_scalar_fgINT8(FLGR_Vector *vctdest, fgINT8 scalar);
  EXPORT_LIB void flgr_vector_populate_from_scalar_fgINT16(FLGR_Vector *vctdest, fgINT16 scalar);
  EXPORT_LIB void flgr_vector_populate_from_scalar_fgINT32(FLGR_Vector *vctdest, fgINT32 scalar);
  EXPORT_LIB void flgr_vector_populate_from_scalar_fgINT64(FLGR_Vector *vctdest, fgINT64 scalar);
  EXPORT_LIB void flgr_vector_populate_from_scalar_fgFLOAT32(FLGR_Vector *vctdest, fgFLOAT32 scalar);
  EXPORT_LIB void flgr_vector_populate_from_scalar_fgFLOAT64(FLGR_Vector *vctdest, fgFLOAT64 scalar);

  EXPORT_LIB void flgr_vector_get_element_fgBIT(FLGR_Vector *vct, int index, fgBIT *value);
  EXPORT_LIB void flgr_vector_get_element_fgUINT8(FLGR_Vector *vct, int index, fgUINT8 *value);
  EXPORT_LIB void flgr_vector_get_element_fgUINT16(FLGR_Vector *vct, int index, fgUINT16 *value);
  EXPORT_LIB void flgr_vector_get_element_fgUINT32(FLGR_Vector *vct, int index, fgUINT32 *value);
  EXPORT_LIB void flgr_vector_get_element_fgUINT64(FLGR_Vector *vct, int index, fgUINT64 *value);
  EXPORT_LIB void flgr_vector_get_element_fgINT8(FLGR_Vector *vct, int index, fgINT8 *value);
  EXPORT_LIB void flgr_vector_get_element_fgINT16(FLGR_Vector *vct, int index, fgINT16 *value);
  EXPORT_LIB void flgr_vector_get_element_fgINT32(FLGR_Vector *vct, int index, fgINT32 *value);
  EXPORT_LIB void flgr_vector_get_element_fgINT64(FLGR_Vector *vct, int index, fgINT64 *value);
  EXPORT_LIB void flgr_vector_get_element_fgFLOAT32(FLGR_Vector *vct, int index, fgFLOAT32 *value);
  EXPORT_LIB void flgr_vector_get_element_fgFLOAT64(FLGR_Vector *vct, int index, fgFLOAT64 *value);

  EXPORT_LIB void flgr_vector_set_element_fgBIT(FLGR_Vector *vct, int index, fgBIT value);
  EXPORT_LIB void flgr_vector_set_element_fgUINT8(FLGR_Vector *vct, int index, fgUINT8 value);
  EXPORT_LIB void flgr_vector_set_element_fgUINT16(FLGR_Vector *vct, int index, fgUINT16 value);
  EXPORT_LIB void flgr_vector_set_element_fgUINT32(FLGR_Vector *vct, int index, fgUINT32 value);
  EXPORT_LIB void flgr_vector_set_element_fgUINT64(FLGR_Vector *vct, int index, fgINT64 value);
  EXPORT_LIB void flgr_vector_set_element_fgINT8(FLGR_Vector *vct, int index, fgINT8 value);
  EXPORT_LIB void flgr_vector_set_element_fgINT16(FLGR_Vector *vct, int index, fgINT16 value);
  EXPORT_LIB void flgr_vector_set_element_fgINT32(FLGR_Vector *vct, int index, fgINT32 value);
  EXPORT_LIB void flgr_vector_set_element_fgINT64(FLGR_Vector *vct, int index, fgINT64 value);
  EXPORT_LIB void flgr_vector_set_element_fgFLOAT32(FLGR_Vector *vct, int index, fgFLOAT32 value);
  EXPORT_LIB void flgr_vector_set_element_fgFLOAT64(FLGR_Vector *vct, int index, fgFLOAT64 value);

  EXPORT_LIB void flgr_vector_copy_fgBIT_fgBIT(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgBIT_fgUINT8(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgBIT_fgUINT16(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgBIT_fgUINT32(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgBIT_fgUINT64(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgBIT_fgINT8(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgBIT_fgINT16(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgBIT_fgINT32(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgBIT_fgINT64(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgBIT_fgFLOAT32(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgBIT_fgFLOAT64(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT8_fgBIT(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT8_fgUINT8(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT8_fgUINT16(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT8_fgUINT32(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT8_fgUINT64(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT8_fgINT8(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT8_fgINT16(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT8_fgINT32(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT8_fgINT64(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT8_fgFLOAT32(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT8_fgFLOAT64(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT16_fgBIT(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT16_fgUINT8(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT16_fgUINT16(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT16_fgUINT32(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT16_fgUINT64(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT16_fgINT8(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT16_fgINT16(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT16_fgINT32(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT16_fgINT64(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT16_fgFLOAT32(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT16_fgFLOAT64(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT32_fgBIT(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT32_fgUINT8(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT32_fgUINT16(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT32_fgUINT32(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT32_fgUINT64(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT32_fgINT8(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT32_fgINT16(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT32_fgINT32(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT32_fgINT64(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT32_fgFLOAT32(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT32_fgFLOAT64(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT64_fgBIT(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT64_fgUINT8(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT64_fgUINT16(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT64_fgUINT32(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT64_fgUINT64(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT64_fgINT8(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT64_fgINT16(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT64_fgINT32(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT64_fgINT64(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT64_fgFLOAT32(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgUINT64_fgFLOAT64(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT8_fgBIT(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT8_fgUINT8(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT8_fgUINT16(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT8_fgUINT32(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT8_fgUINT64(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT8_fgINT8(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT8_fgINT16(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT8_fgINT32(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT8_fgINT64(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT8_fgFLOAT32(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT8_fgFLOAT64(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT16_fgBIT(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT16_fgUINT8(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT16_fgUINT16(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT16_fgUINT32(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT16_fgUINT64(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT16_fgINT8(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT16_fgINT16(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT16_fgINT32(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT16_fgINT64(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT16_fgFLOAT32(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT16_fgFLOAT64(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT32_fgBIT(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT32_fgUINT8(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT32_fgUINT16(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT32_fgUINT32(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT32_fgUINT64(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT32_fgINT8(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT32_fgINT16(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT32_fgINT32(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT32_fgINT64(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT32_fgFLOAT32(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT32_fgFLOAT64(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT64_fgBIT(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT64_fgUINT8(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT64_fgUINT16(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT64_fgUINT32(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT64_fgUINT64(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT64_fgINT8(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT64_fgINT16(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT64_fgINT32(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT64_fgINT64(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT64_fgFLOAT32(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgINT64_fgFLOAT64(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgFLOAT32_fgBIT(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgFLOAT32_fgUINT8(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgFLOAT32_fgUINT16(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgFLOAT32_fgUINT32(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgFLOAT32_fgUINT64(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgFLOAT32_fgINT8(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgFLOAT32_fgINT16(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgFLOAT32_fgINT32(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgFLOAT32_fgINT64(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgFLOAT32_fgFLOAT32(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgFLOAT32_fgFLOAT64(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgFLOAT64_fgBIT(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgFLOAT64_fgUINT8(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgFLOAT64_fgUINT16(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgFLOAT64_fgUINT32(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgFLOAT64_fgUINT64(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgFLOAT64_fgINT8(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgFLOAT64_fgINT16(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgFLOAT64_fgINT32(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgFLOAT64_fgINT64(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgFLOAT64_fgFLOAT32(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);
  EXPORT_LIB void flgr_vector_copy_fgFLOAT64_fgFLOAT64(FLGR_Vector *vctdest, FLGR_Vector *vctsrc);


#endif

#ifdef __cplusplus
}
#endif

