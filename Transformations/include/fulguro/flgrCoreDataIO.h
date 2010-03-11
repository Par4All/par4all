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


#ifndef __FLGR_CORE_DATA_IO_H
#define __FLGR_CORE_DATA_IO_H

#include <flgrCoreDll.h>
#include <flgrCoreTypes.h>
#include <flgrCoreErrors.h>
#include <flgrCoreIO.h>
#include <flgrCoreData.h>
#include <flgrCoreVector.h>

  /*!
   * \addtogroup group_fulguro_core_array
   * @{
   */


  EXPORT_LIB FLGR_Ret flgr1d_set_data_vector(FLGR_Data1D *dat, int pos, FLGR_Vector *vct);
  EXPORT_LIB FLGR_Ret flgr1d_get_data_vector(FLGR_Data1D *dat, int pos, FLGR_Vector *vct);
  EXPORT_LIB FLGR_Ret flgr1d_get_data_vector_no_norm(FLGR_Data1D *dat, int pos, FLGR_Vector *vct);

  EXPORT_LIB FLGR_Ret flgr2d_set_data_vector(FLGR_Data2D *dat, int row, int col, FLGR_Vector *vct);
  EXPORT_LIB FLGR_Ret flgr2d_get_data_vector(FLGR_Data2D *dat, int row, int col, FLGR_Vector *vct);
  EXPORT_LIB FLGR_Ret flgr2d_get_data_vector_no_norm(FLGR_Data2D *dat, int row, int col, FLGR_Vector *vct);


  EXPORT_LIB FLGR_Ret flgr1d_set_data_ptr(FLGR_Data1D *dat, int pos, void *value);
  EXPORT_LIB FLGR_Ret flgr1d_set_data_str(FLGR_Data1D *dat, int pos, char *value);
  EXPORT_LIB FLGR_Ret flgr1d_get_data_ptr(FLGR_Data1D *dat, int pos, void *value);
  EXPORT_LIB FLGR_Ret flgr1d_get_data_no_norm_ptr(FLGR_Data1D *dat, int pos, void *value);



  EXPORT_LIB FLGR_Ret flgr2d_set_data_ptr(FLGR_Data2D *dat, int row, int col, void *value);
  EXPORT_LIB FLGR_Ret flgr2d_set_data_str(FLGR_Data2D *dat, int row, int col, char *value);
  EXPORT_LIB FLGR_Ret flgr2d_get_data_ptr(FLGR_Data2D *dat, int row, int col, void *value);
  EXPORT_LIB FLGR_Ret flgr2d_get_data_no_norm_ptr(FLGR_Data2D *dat, int row, int col, void *value);







  EXPORT_LIB void flgr1d_set_data_vector_fgBIT(FLGR_Data1D *dat, int pos, FLGR_Vector *vct);
  EXPORT_LIB void flgr1d_set_data_vector_fgUINT8(FLGR_Data1D *dat, int pos, FLGR_Vector *vct);
  EXPORT_LIB void flgr1d_set_data_vector_fgUINT16(FLGR_Data1D *dat, int pos, FLGR_Vector *vct);
  EXPORT_LIB void flgr1d_set_data_vector_fgUINT32(FLGR_Data1D *dat, int pos, FLGR_Vector *vct);
  EXPORT_LIB void flgr1d_set_data_vector_fgINT8(FLGR_Data1D *dat, int pos, FLGR_Vector *vct);
  EXPORT_LIB void flgr1d_set_data_vector_fgINT16(FLGR_Data1D *dat, int pos, FLGR_Vector *vct);
  EXPORT_LIB void flgr1d_set_data_vector_fgINT32(FLGR_Data1D *dat, int pos, FLGR_Vector *vct);
  EXPORT_LIB void flgr1d_set_data_vector_fgFLOAT32(FLGR_Data1D *dat, int pos, FLGR_Vector *vct);
  EXPORT_LIB void flgr1d_set_data_vector_fgFLOAT64(FLGR_Data1D *dat, int pos, FLGR_Vector *vct);

  EXPORT_LIB void flgr1d_get_data_vector_fgBIT(FLGR_Data1D *dat, int pos, FLGR_Vector *vct);
  EXPORT_LIB void flgr1d_get_data_vector_fgUINT8(FLGR_Data1D *dat, int pos, FLGR_Vector *vct);
  EXPORT_LIB void flgr1d_get_data_vector_fgUINT16(FLGR_Data1D *dat, int pos, FLGR_Vector *vct);
  EXPORT_LIB void flgr1d_get_data_vector_fgUINT32(FLGR_Data1D *dat, int pos, FLGR_Vector *vct);
  EXPORT_LIB void flgr1d_get_data_vector_fgINT8(FLGR_Data1D *dat, int pos, FLGR_Vector *vct);
  EXPORT_LIB void flgr1d_get_data_vector_fgINT16(FLGR_Data1D *dat, int pos, FLGR_Vector *vct);
  EXPORT_LIB void flgr1d_get_data_vector_fgINT32(FLGR_Data1D *dat, int pos, FLGR_Vector *vct);
  EXPORT_LIB void flgr1d_get_data_vector_fgFLOAT32(FLGR_Data1D *dat, int pos, FLGR_Vector *vct);
  EXPORT_LIB void flgr1d_get_data_vector_fgFLOAT64(FLGR_Data1D *dat, int pos, FLGR_Vector *vct);

  EXPORT_LIB void flgr1d_get_data_vector_no_norm_fgBIT(FLGR_Data1D *dat, int pos, FLGR_Vector *vct);
  EXPORT_LIB void flgr1d_get_data_vector_no_norm_fgUINT8(FLGR_Data1D *dat, int pos, FLGR_Vector *vct);
  EXPORT_LIB void flgr1d_get_data_vector_no_norm_fgUINT16(FLGR_Data1D *dat, int pos, FLGR_Vector *vct);
  EXPORT_LIB void flgr1d_get_data_vector_no_norm_fgUINT32(FLGR_Data1D *dat, int pos, FLGR_Vector *vct);
  EXPORT_LIB void flgr1d_get_data_vector_no_norm_fgINT8(FLGR_Data1D *dat, int pos, FLGR_Vector *vct);
  EXPORT_LIB void flgr1d_get_data_vector_no_norm_fgINT16(FLGR_Data1D *dat, int pos, FLGR_Vector *vct);
  EXPORT_LIB void flgr1d_get_data_vector_no_norm_fgINT32(FLGR_Data1D *dat, int pos, FLGR_Vector *vct);
  EXPORT_LIB void flgr1d_get_data_vector_no_norm_fgFLOAT32(FLGR_Data1D *dat, int pos, FLGR_Vector *vct);
  EXPORT_LIB void flgr1d_get_data_vector_no_norm_fgFLOAT64(FLGR_Data1D *dat, int pos, FLGR_Vector *vct);


  EXPORT_LIB void flgr2d_set_data_vector_fgBIT(FLGR_Data2D *dat, int row, int col, FLGR_Vector *vct);
  EXPORT_LIB void flgr2d_set_data_vector_fgUINT8(FLGR_Data2D *dat, int row, int col, FLGR_Vector *vct);
  EXPORT_LIB void flgr2d_set_data_vector_fgUINT16(FLGR_Data2D *dat, int row, int col, FLGR_Vector *vct);
  EXPORT_LIB void flgr2d_set_data_vector_fgUINT32(FLGR_Data2D *dat, int row, int col, FLGR_Vector *vct);
  EXPORT_LIB void flgr2d_set_data_vector_fgINT8(FLGR_Data2D *dat, int row, int col, FLGR_Vector *vct);
  EXPORT_LIB void flgr2d_set_data_vector_fgINT16(FLGR_Data2D *dat, int row, int col, FLGR_Vector *vct);
  EXPORT_LIB void flgr2d_set_data_vector_fgINT32(FLGR_Data2D *dat, int row, int col, FLGR_Vector *vct);
  EXPORT_LIB void flgr2d_set_data_vector_fgFLOAT32(FLGR_Data2D *dat, int row, int col, FLGR_Vector *vct);
  EXPORT_LIB void flgr2d_set_data_vector_fgFLOAT64(FLGR_Data2D *dat, int row, int col, FLGR_Vector *vct);

  EXPORT_LIB void flgr2d_get_data_vector_fgBIT(FLGR_Data2D *dat, int row, int col, FLGR_Vector *vct);
  EXPORT_LIB void flgr2d_get_data_vector_fgUINT8(FLGR_Data2D *dat, int row, int col, FLGR_Vector *vct);
  EXPORT_LIB void flgr2d_get_data_vector_fgUINT16(FLGR_Data2D *dat, int row, int col, FLGR_Vector *vct);
  EXPORT_LIB void flgr2d_get_data_vector_fgUINT32(FLGR_Data2D *dat, int row, int col, FLGR_Vector *vct);
  EXPORT_LIB void flgr2d_get_data_vector_fgINT8(FLGR_Data2D *dat, int row, int col, FLGR_Vector *vct);
  EXPORT_LIB void flgr2d_get_data_vector_fgINT16(FLGR_Data2D *dat, int row, int col, FLGR_Vector *vct);
  EXPORT_LIB void flgr2d_get_data_vector_fgINT32(FLGR_Data2D *dat, int row, int col, FLGR_Vector *vct);
  EXPORT_LIB void flgr2d_get_data_vector_fgFLOAT32(FLGR_Data2D *dat, int row, int col, FLGR_Vector *vct);
  EXPORT_LIB void flgr2d_get_data_vector_fgFLOAT64(FLGR_Data2D *dat, int row, int col, FLGR_Vector *vct);

  EXPORT_LIB void flgr2d_get_data_vector_no_norm_fgBIT(FLGR_Data2D *dat, int row, int col, FLGR_Vector *vct);
  EXPORT_LIB void flgr2d_get_data_vector_no_norm_fgUINT8(FLGR_Data2D *dat, int row, int col, FLGR_Vector *vct);
  EXPORT_LIB void flgr2d_get_data_vector_no_norm_fgUINT16(FLGR_Data2D *dat, int row, int col, FLGR_Vector *vct);
  EXPORT_LIB void flgr2d_get_data_vector_no_norm_fgUINT32(FLGR_Data2D *dat, int row, int col, FLGR_Vector *vct);
  EXPORT_LIB void flgr2d_get_data_vector_no_norm_fgINT8(FLGR_Data2D *dat, int row, int col, FLGR_Vector *vct);
  EXPORT_LIB void flgr2d_get_data_vector_no_norm_fgINT16(FLGR_Data2D *dat, int row, int col, FLGR_Vector *vct);
  EXPORT_LIB void flgr2d_get_data_vector_no_norm_fgINT32(FLGR_Data2D *dat, int row, int col, FLGR_Vector *vct);
  EXPORT_LIB void flgr2d_get_data_vector_no_norm_fgFLOAT32(FLGR_Data2D *dat, int row, int col, FLGR_Vector *vct);
  EXPORT_LIB void flgr2d_get_data_vector_no_norm_fgFLOAT64(FLGR_Data2D *dat, int row, int col, FLGR_Vector *vct);




  EXPORT_LIB void flgr1d_set_data_fgBIT(FLGR_Data1D *dat, int pos, fgBIT *value);
  EXPORT_LIB void flgr1d_set_data_fgUINT8(FLGR_Data1D *dat, int pos, fgUINT8 *value);
  EXPORT_LIB void flgr1d_set_data_fgUINT16(FLGR_Data1D *dat, int pos, fgUINT16 *value);
  EXPORT_LIB void flgr1d_set_data_fgUINT32(FLGR_Data1D *dat, int pos, fgUINT32 *value);
  EXPORT_LIB void flgr1d_set_data_fgINT8(FLGR_Data1D *dat, int pos, fgINT8 *value);
  EXPORT_LIB void flgr1d_set_data_fgINT16(FLGR_Data1D *dat, int pos, fgINT16 *value);
  EXPORT_LIB void flgr1d_set_data_fgINT32(FLGR_Data1D *dat, int pos, fgINT32 *value);
  EXPORT_LIB void flgr1d_set_data_fgFLOAT32(FLGR_Data1D *dat, int pos, fgFLOAT32 *value);
  EXPORT_LIB void flgr1d_set_data_fgFLOAT64(FLGR_Data1D *dat, int pos, fgFLOAT64 *value);

  EXPORT_LIB void flgr1d_get_data_fgBIT(FLGR_Data1D *dat, int pos, fgBIT *value);
  EXPORT_LIB void flgr1d_get_data_fgUINT8(FLGR_Data1D *dat, int pos, fgUINT8 *value);
  EXPORT_LIB void flgr1d_get_data_fgUINT16(FLGR_Data1D *dat, int pos, fgUINT16 *value);
  EXPORT_LIB void flgr1d_get_data_fgUINT32(FLGR_Data1D *dat, int pos, fgUINT32 *value);
  EXPORT_LIB void flgr1d_get_data_fgINT8(FLGR_Data1D *dat, int pos, fgINT8 *value);
  EXPORT_LIB void flgr1d_get_data_fgINT16(FLGR_Data1D *dat, int pos, fgINT16 *value);
  EXPORT_LIB void flgr1d_get_data_fgINT32(FLGR_Data1D *dat, int pos, fgINT32 *value);
  EXPORT_LIB void flgr1d_get_data_fgFLOAT32(FLGR_Data1D *dat, int pos, fgFLOAT32 *value);
  EXPORT_LIB void flgr1d_get_data_fgFLOAT64(FLGR_Data1D *dat, int pos, fgFLOAT64 *value);

  EXPORT_LIB void flgr1d_get_data_no_norm_fgBIT(FLGR_Data1D *dat, int pos, fgBIT *value);
  EXPORT_LIB void flgr1d_get_data_no_norm_fgUINT8(FLGR_Data1D *dat, int pos, fgUINT8 *value);
  EXPORT_LIB void flgr1d_get_data_no_norm_fgUINT16(FLGR_Data1D *dat, int pos, fgUINT16 *value);
  EXPORT_LIB void flgr1d_get_data_no_norm_fgUINT32(FLGR_Data1D *dat, int pos, fgUINT32 *value);
  EXPORT_LIB void flgr1d_get_data_no_norm_fgINT8(FLGR_Data1D *dat, int pos, fgINT8 *value);
  EXPORT_LIB void flgr1d_get_data_no_norm_fgINT16(FLGR_Data1D *dat, int pos, fgINT16 *value);
  EXPORT_LIB void flgr1d_get_data_no_norm_fgINT32(FLGR_Data1D *dat, int pos, fgINT32 *value);
  EXPORT_LIB void flgr1d_get_data_no_norm_fgFLOAT32(FLGR_Data1D *dat, int pos, fgFLOAT32 *value);
  EXPORT_LIB void flgr1d_get_data_no_norm_fgFLOAT64(FLGR_Data1D *dat, int pos, fgFLOAT64 *value);






  EXPORT_LIB void flgr2d_set_data_fgBIT(FLGR_Data2D *dat, int row, int col, fgBIT *value);
  EXPORT_LIB void flgr2d_set_data_fgUINT8(FLGR_Data2D *dat, int row, int col, fgUINT8 *value);
  EXPORT_LIB void flgr2d_set_data_fgUINT16(FLGR_Data2D *dat, int row, int col, fgUINT16 *value);
  EXPORT_LIB void flgr2d_set_data_fgUINT32(FLGR_Data2D *dat, int row, int col, fgUINT32 *value);
  EXPORT_LIB void flgr2d_set_data_fgINT8(FLGR_Data2D *dat, int row, int col, fgINT8 *value);
  EXPORT_LIB void flgr2d_set_data_fgINT16(FLGR_Data2D *dat, int row, int col, fgINT16 *value);
  EXPORT_LIB void flgr2d_set_data_fgINT32(FLGR_Data2D *dat, int row, int col, fgINT32 *value);
  EXPORT_LIB void flgr2d_set_data_fgFLOAT32(FLGR_Data2D *dat, int row, int col, fgFLOAT32 *value);
  EXPORT_LIB void flgr2d_set_data_fgFLOAT64(FLGR_Data2D *dat, int row, int col, fgFLOAT64 *value);

  EXPORT_LIB void flgr2d_get_data_fgBIT(FLGR_Data2D *dat, int row, int col, fgBIT *value);
  EXPORT_LIB void flgr2d_get_data_fgUINT8(FLGR_Data2D *dat, int row, int col, fgUINT8 *value);
  EXPORT_LIB void flgr2d_get_data_fgUINT16(FLGR_Data2D *dat, int row, int col, fgUINT16 *value);
  EXPORT_LIB void flgr2d_get_data_fgUINT32(FLGR_Data2D *dat, int row, int col, fgUINT32 *value);
  EXPORT_LIB void flgr2d_get_data_fgINT8(FLGR_Data2D *dat, int row, int col, fgINT8 *value);
  EXPORT_LIB void flgr2d_get_data_fgINT16(FLGR_Data2D *dat, int row, int col, fgINT16 *value);
  EXPORT_LIB void flgr2d_get_data_fgINT32(FLGR_Data2D *dat, int row, int col, fgINT32 *value);
  EXPORT_LIB void flgr2d_get_data_fgFLOAT32(FLGR_Data2D *dat, int row, int col, fgFLOAT32 *value);
  EXPORT_LIB void flgr2d_get_data_fgFLOAT64(FLGR_Data2D *dat, int row, int col, fgFLOAT64 *value);

  EXPORT_LIB void flgr2d_get_data_no_norm_fgBIT(FLGR_Data2D *dat, int row, int col, fgBIT *value);
  EXPORT_LIB void flgr2d_get_data_no_norm_fgUINT8(FLGR_Data2D *dat, int row, int col, fgUINT8 *value);
  EXPORT_LIB void flgr2d_get_data_no_norm_fgUINT16(FLGR_Data2D *dat, int row, int col, fgUINT16 *value);
  EXPORT_LIB void flgr2d_get_data_no_norm_fgUINT32(FLGR_Data2D *dat, int row, int col, fgUINT32 *value);
  EXPORT_LIB void flgr2d_get_data_no_norm_fgINT8(FLGR_Data2D *dat, int row, int col, fgINT8 *value);
  EXPORT_LIB void flgr2d_get_data_no_norm_fgINT16(FLGR_Data2D *dat, int row, int col, fgINT16 *value);
  EXPORT_LIB void flgr2d_get_data_no_norm_fgINT32(FLGR_Data2D *dat, int row, int col, fgINT32 *value);
  EXPORT_LIB void flgr2d_get_data_no_norm_fgFLOAT32(FLGR_Data2D *dat, int row, int col, fgFLOAT32 *value);
  EXPORT_LIB void flgr2d_get_data_no_norm_fgFLOAT64(FLGR_Data2D *dat, int row, int col, fgFLOAT64 *value);







  static __inline__ void flgr1d_set_data_array_fgBIT(fgBIT *array, int pos, fgBIT value) {
    flgr_set_array_fgBIT(array,pos,value);
  }

  static __inline__ void flgr1d_set_data_array_fgUINT8(fgUINT8* array, int pos, fgUINT8 value) {
    flgr_set_array_fgUINT8(array,pos,value);
  }

  static __inline__ void flgr1d_set_data_array_fgUINT16(fgUINT16* array, int pos, fgUINT16 value) {
    flgr_set_array_fgUINT16(array,pos,value);
  }

  static __inline__ void flgr1d_set_data_array_fgUINT32(fgUINT32* array, int pos, fgUINT32 value) {
    flgr_set_array_fgUINT32(array,pos,value);
  }

  static __inline__ void flgr1d_set_data_array_fgUINT64(fgUINT64* array, int pos, fgUINT64 value) {
    flgr_set_array_fgUINT64(array,pos,value);
  }

  static __inline__ void flgr1d_set_data_array_fgINT8(fgINT8* array, int pos, fgINT8 value) {
    flgr_set_array_fgINT8(array,pos,value);
  }

  static __inline__ void flgr1d_set_data_array_fgINT16(fgINT16* array, int pos, fgINT16 value) {
    flgr_set_array_fgINT16(array,pos,value);
  }

  static __inline__ void flgr1d_set_data_array_fgINT32(fgINT32* array, int pos, fgINT32 value) {
    flgr_set_array_fgINT32(array,pos,value);
  }

  static __inline__ void flgr1d_set_data_array_fgINT64(fgINT64* array, int pos, fgINT64 value) {
    flgr_set_array_fgINT64(array,pos,value);
  }

  static __inline__ void flgr1d_set_data_array_fgFLOAT32(fgFLOAT32* array, int pos, fgFLOAT32 value) {
    flgr_set_array_fgFLOAT32(array,pos,value);
  }

  static __inline__ void flgr1d_set_data_array_fgFLOAT64(fgFLOAT64* array, int pos, fgFLOAT64 value) {
    flgr_set_array_fgFLOAT64(array,pos,value);
  }







  static __inline__ fgBIT flgr1d_get_data_array_fgBIT(fgBIT* array, int pos) {
    return flgr_get_array_fgBIT(array,pos);
  }

  static __inline__ fgUINT8 flgr1d_get_data_array_fgUINT8(fgUINT8* array, int pos) {
    return flgr_get_array_fgUINT8(array,pos);
  }

  static __inline__ fgUINT16 flgr1d_get_data_array_fgUINT16(fgUINT16* array, int pos) {
    return flgr_get_array_fgUINT16(array,pos);
  }

  static __inline__ fgUINT32 flgr1d_get_data_array_fgUINT32(fgUINT32* array, int pos) {
    return flgr_get_array_fgUINT32(array,pos);
  }

  static __inline__ fgUINT64 flgr1d_get_data_array_fgUINT64(fgUINT64* array, int pos) {
    return flgr_get_array_fgUINT64(array,pos);
  }

  static __inline__ fgINT8 flgr1d_get_data_array_fgINT8(fgINT8* array, int pos) {
    return flgr_get_array_fgINT8(array,pos);
  }

  static __inline__ fgINT16 flgr1d_get_data_array_fgINT16(fgINT16* array, int pos) {
    return flgr_get_array_fgINT16(array,pos);
  }

  static __inline__ fgINT32 flgr1d_get_data_array_fgINT32(fgINT32* array, int pos) {
    return flgr_get_array_fgINT32(array,pos);
  }

  static __inline__ fgINT64 flgr1d_get_data_array_fgINT64(fgINT64* array, int pos) {
    return flgr_get_array_fgINT64(array,pos);
  }

  static __inline__ fgFLOAT32 flgr1d_get_data_array_fgFLOAT32(fgFLOAT32* array, int pos) {
    return flgr_get_array_fgFLOAT32(array,pos);
  }

  static __inline__ fgFLOAT64 flgr1d_get_data_array_fgFLOAT64(fgFLOAT64* array, int pos) {
    return flgr_get_array_fgFLOAT64(array,pos);
  }

 





  static __inline__ void flgr2d_set_data_array_fgBIT(fgBIT** array, int row, int col, fgBIT value) {
    flgr_set_array_fgBIT(array[row],col,value);
  }
  static __inline__ void flgr2d_set_data_array_fgUINT8(fgUINT8** array, int row, int col, fgUINT8 value) {
    flgr_set_array_fgUINT8(array[row],col,value);
  }
  static __inline__ void flgr2d_set_data_array_fgUINT16(fgUINT16** array, int row, int col, fgUINT16 value) {
    flgr_set_array_fgUINT16(array[row],col,value);
  }
  static __inline__ void flgr2d_set_data_array_fgUINT32(fgUINT32** array, int row, int col, fgUINT32 value) {
    flgr_set_array_fgUINT32(array[row],col,value);
  }
  static __inline__ void flgr2d_set_data_array_fgINT8(fgINT8** array, int row, int col, fgINT8 value) {
    flgr_set_array_fgINT8(array[row],col,value);
  }
  static __inline__ void flgr2d_set_data_array_fgINT16(fgINT16** array, int row, int col, fgINT16 value) {
    flgr_set_array_fgINT16(array[row],col,value);
  }
  static __inline__ void flgr2d_set_data_array_fgINT32(fgINT32** array, int row, int col, fgINT32 value) {
    flgr_set_array_fgINT32(array[row],col,value);
  }
  static __inline__ void flgr2d_set_data_array_fgFLOAT32(fgFLOAT32** array, int row, int col, fgFLOAT32 value) {
    flgr_set_array_fgFLOAT32(array[row],col,value);
  }
  static __inline__ void flgr2d_set_data_array_fgFLOAT64(fgFLOAT64** array, int row, int col, fgFLOAT64 value) {
    flgr_set_array_fgFLOAT64(array[row],col,value);
  }



  static __inline__ fgBIT flgr2d_get_data_array_fgBIT(fgBIT** array, int row, int col) {
    return flgr_get_array_fgBIT(array[row],col);
  }
  static __inline__ fgUINT8 flgr2d_get_data_array_fgUINT8(fgUINT8** array, int row, int col) {
    return flgr_get_array_fgUINT8(array[row],col);
  }
  static __inline__ fgUINT16 flgr2d_get_data_array_fgUINT16(fgUINT16** array, int row, int col) {
    return flgr_get_array_fgUINT16(array[row],col);
  }
  static __inline__ fgUINT32 flgr2d_get_data_array_fgUINT32(fgUINT32** array, int row, int col) {
    return flgr_get_array_fgUINT32(array[row],col);
  }
  static __inline__ fgINT8 flgr2d_get_data_array_fgINT8(fgINT8** array, int row, int col) {
    return flgr_get_array_fgINT8(array[row],col);
  }
  static __inline__ fgINT16 flgr2d_get_data_array_fgINT16(fgINT16** array, int row, int col) {
    return flgr_get_array_fgINT16(array[row],col);
  }
  static __inline__ fgINT32 flgr2d_get_data_array_fgINT32(fgINT32** array, int row, int col) {
    return flgr_get_array_fgINT32(array[row],col);
  }
  static __inline__ fgFLOAT32 flgr2d_get_data_array_fgFLOAT32(fgFLOAT32** array, int row, int col) {
    return flgr_get_array_fgFLOAT32(array[row],col);
  }
  static __inline__ fgFLOAT64 flgr2d_get_data_array_fgFLOAT64(fgFLOAT64** array, int row, int col) {
    return flgr_get_array_fgFLOAT64(array[row],col);
  }
  



  //! @}

#endif

#ifdef __cplusplus
}
#endif
