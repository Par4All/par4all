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


#ifndef __FLGR_CORE_COPY_H
#define __FLGR_CORE_COPY_H


#include <flgrCoreDll.h>
#include <flgrCoreErrors.h>
#include <flgrCoreTypes.h>
#include <flgrCoreData.h>

  EXPORT_LIB FLGR_Ret flgr1d_copy(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB FLGR_Ret flgr2d_copy(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);

  EXPORT_LIB FLGR_Ret flgr1d_import_raw_ptr(FLGR_Data1D *dat, void* raw);
  EXPORT_LIB FLGR_Ret flgr1d_export_raw_ptr(void* raw, FLGR_Data1D *dat);

  EXPORT_LIB FLGR_Ret flgr2d_import_raw_ptr(FLGR_Data2D *dat, void* raw);
  EXPORT_LIB FLGR_Ret flgr2d_export_raw_ptr(void* raw, FLGR_Data2D *dat);

  EXPORT_LIB FLGR_Ret flgr1d_mirror(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB FLGR_Ret flgr1d_mirror_hmorph(FLGR_Data1D *dat);

  EXPORT_LIB FLGR_Ret flgr2d_mirror_horizontal(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB FLGR_Ret flgr2d_mirror_horizontal_hmorph(FLGR_Data2D *dat);

  EXPORT_LIB FLGR_Ret flgr2d_mirror_vertical(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB FLGR_Ret flgr2d_mirror_vertical_hmorph(FLGR_Data2D *dat);

  EXPORT_LIB FLGR_Ret flgr2d_fill_nhb_even_rows(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB FLGR_Ret flgr2d_fill_nhb_odd_rows(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
 
  EXPORT_LIB FLGR_Ret flgr2d_fill_nhbs_for_6_connexity(FLGR_Data2D *nhbEven, FLGR_Data2D *nhbOdd, 
						       FLGR_Data2D *nhb, int SYM);
  
  
  EXPORT_LIB FLGR_Ret flgr2d_copy_to_1d(FLGR_Data1D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB FLGR_Ret flgr1d_copy_to_2d(FLGR_Data2D *datdest, FLGR_Data1D *datsrc);





  EXPORT_LIB void flgr2d_copy_to_1d_fgBIT(FLGR_Data1D *datdest, FLGR_Data2D *datsrc); 
  EXPORT_LIB void flgr2d_copy_to_1d_fgUINT8(FLGR_Data1D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_to_1d_fgUINT16(FLGR_Data1D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_to_1d_fgUINT32(FLGR_Data1D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_to_1d_fgINT8(FLGR_Data1D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_to_1d_fgINT16(FLGR_Data1D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_to_1d_fgINT32(FLGR_Data1D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_to_1d_fgFLOAT32(FLGR_Data1D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_to_1d_fgFLOAT64(FLGR_Data1D *datdest, FLGR_Data2D *datsrc);

  EXPORT_LIB void flgr1d_copy_to_2d_fgBIT(FLGR_Data2D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_to_2d_fgUINT8(FLGR_Data2D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_to_2d_fgUINT16(FLGR_Data2D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_to_2d_fgUINT32(FLGR_Data2D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_to_2d_fgINT8(FLGR_Data2D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_to_2d_fgINT16(FLGR_Data2D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_to_2d_fgINT32(FLGR_Data2D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_to_2d_fgFLOAT32(FLGR_Data2D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_to_2d_fgFLOAT64(FLGR_Data2D *datdest, FLGR_Data1D *datsrc);


  EXPORT_LIB void flgr2d_fill_nhb_odd_rows_fgBIT(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_fill_nhb_odd_rows_fgUINT8(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_fill_nhb_odd_rows_fgUINT16(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_fill_nhb_odd_rows_fgUINT32(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_fill_nhb_odd_rows_fgINT8(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_fill_nhb_odd_rows_fgINT16(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_fill_nhb_odd_rows_fgINT32(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_fill_nhb_odd_rows_fgFLOAT32(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_fill_nhb_odd_rows_fgFLOAT64(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);

  EXPORT_LIB void flgr2d_fill_nhb_even_rows_fgBIT(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_fill_nhb_even_rows_fgUINT8(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_fill_nhb_even_rows_fgUINT16(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_fill_nhb_even_rows_fgUINT32(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_fill_nhb_even_rows_fgINT8(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_fill_nhb_even_rows_fgINT16(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_fill_nhb_even_rows_fgINT32(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_fill_nhb_even_rows_fgFLOAT32(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_fill_nhb_even_rows_fgFLOAT64(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);




  EXPORT_LIB void flgr1d_mirror_fgBIT(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_mirror_fgUINT8(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_mirror_fgUINT16(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_mirror_fgUINT32(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_mirror_fgINT8(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_mirror_fgINT16(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_mirror_fgINT32(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_mirror_fgFLOAT32(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_mirror_fgFLOAT64(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);


  EXPORT_LIB void flgr1d_mirror_hmorph_fgBIT(FLGR_Data1D *dat);
  EXPORT_LIB void flgr1d_mirror_hmorph_fgUINT8(FLGR_Data1D *dat);
  EXPORT_LIB void flgr1d_mirror_hmorph_fgUINT16(FLGR_Data1D *dat);
  EXPORT_LIB void flgr1d_mirror_hmorph_fgUINT32(FLGR_Data1D *dat);
  EXPORT_LIB void flgr1d_mirror_hmorph_fgINT8(FLGR_Data1D *dat);
  EXPORT_LIB void flgr1d_mirror_hmorph_fgINT16(FLGR_Data1D *dat);
  EXPORT_LIB void flgr1d_mirror_hmorph_fgINT32(FLGR_Data1D *dat);
  EXPORT_LIB void flgr1d_mirror_hmorph_fgFLOAT32(FLGR_Data1D *dat);
  EXPORT_LIB void flgr1d_mirror_hmorph_fgFLOAT64(FLGR_Data1D *dat);
 

  EXPORT_LIB void flgr2d_mirror_horizontal_fgBIT(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_mirror_horizontal_fgUINT8(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_mirror_horizontal_fgUINT16(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_mirror_horizontal_fgUINT32(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_mirror_horizontal_fgINT8(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_mirror_horizontal_fgINT16(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_mirror_horizontal_fgINT32(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_mirror_horizontal_fgFLOAT32(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_mirror_horizontal_fgFLOAT64(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);


  EXPORT_LIB void flgr2d_mirror_horizontal_hmorph_fgBIT(FLGR_Data2D *dat);
  EXPORT_LIB void flgr2d_mirror_horizontal_hmorph_fgUINT8(FLGR_Data2D *dat);
  EXPORT_LIB void flgr2d_mirror_horizontal_hmorph_fgUINT16(FLGR_Data2D *dat);
  EXPORT_LIB void flgr2d_mirror_horizontal_hmorph_fgUINT32(FLGR_Data2D *dat);
  EXPORT_LIB void flgr2d_mirror_horizontal_hmorph_fgINT8(FLGR_Data2D *dat);
  EXPORT_LIB void flgr2d_mirror_horizontal_hmorph_fgINT16(FLGR_Data2D *dat);
  EXPORT_LIB void flgr2d_mirror_horizontal_hmorph_fgINT32(FLGR_Data2D *dat);
  EXPORT_LIB void flgr2d_mirror_horizontal_hmorph_fgFLOAT32(FLGR_Data2D *dat);
  EXPORT_LIB void flgr2d_mirror_horizontal_hmorph_fgFLOAT64(FLGR_Data2D *dat);
 

  EXPORT_LIB void flgr1d_import_raw_fgBIT(FLGR_Data1D *dat, void* raw);
  EXPORT_LIB void flgr1d_import_raw_fgUINT8(FLGR_Data1D *dat, void* raw);
  EXPORT_LIB void flgr1d_import_raw_fgUINT16(FLGR_Data1D *dat, void* raw);
  EXPORT_LIB void flgr1d_import_raw_fgUINT32(FLGR_Data1D *dat, void* raw);
  EXPORT_LIB void flgr1d_import_raw_fgINT8(FLGR_Data1D *dat, void* raw);
  EXPORT_LIB void flgr1d_import_raw_fgINT16(FLGR_Data1D *dat, void* raw);
  EXPORT_LIB void flgr1d_import_raw_fgINT32(FLGR_Data1D *dat, void* raw);
  EXPORT_LIB void flgr1d_import_raw_fgFLOAT32(FLGR_Data1D *dat, void* raw);
  EXPORT_LIB void flgr1d_import_raw_fgFLOAT64(FLGR_Data1D *dat, void* raw);

  EXPORT_LIB void flgr1d_export_raw_fgBIT(void *raw, FLGR_Data1D *dat);
  EXPORT_LIB void flgr1d_export_raw_fgUINT8(void *raw, FLGR_Data1D *dat);
  EXPORT_LIB void flgr1d_export_raw_fgUINT16(void *raw, FLGR_Data1D *dat);
  EXPORT_LIB void flgr1d_export_raw_fgUINT32(void *raw, FLGR_Data1D *dat);
  EXPORT_LIB void flgr1d_export_raw_fgINT8(void *raw, FLGR_Data1D *dat);
  EXPORT_LIB void flgr1d_export_raw_fgINT16(void *raw, FLGR_Data1D *dat);
  EXPORT_LIB void flgr1d_export_raw_fgINT32(void *raw, FLGR_Data1D *dat);
  EXPORT_LIB void flgr1d_export_raw_fgFLOAT32(void *raw, FLGR_Data1D *dat);
  EXPORT_LIB void flgr1d_export_raw_fgFLOAT64(void *raw, FLGR_Data1D *dat);

  EXPORT_LIB void flgr2d_import_raw_fgBIT(FLGR_Data2D *dat, void* raw);
  EXPORT_LIB void flgr2d_import_raw_fgUINT8(FLGR_Data2D *dat, void* raw);
  EXPORT_LIB void flgr2d_import_raw_fgUINT16(FLGR_Data2D *dat, void* raw);
  EXPORT_LIB void flgr2d_import_raw_fgUINT32(FLGR_Data2D *dat, void* raw);
  EXPORT_LIB void flgr2d_import_raw_fgINT8(FLGR_Data2D *dat, void* raw);
  EXPORT_LIB void flgr2d_import_raw_fgINT16(FLGR_Data2D *dat, void* raw);
  EXPORT_LIB void flgr2d_import_raw_fgINT32(FLGR_Data2D *dat, void* raw);
  EXPORT_LIB void flgr2d_import_raw_fgFLOAT32(FLGR_Data2D *dat, void* raw);
  EXPORT_LIB void flgr2d_import_raw_fgFLOAT64(FLGR_Data2D *dat, void* raw);

  EXPORT_LIB void flgr2d_export_raw_fgBIT(void *raw, FLGR_Data2D *dat);
  EXPORT_LIB void flgr2d_export_raw_fgUINT8(void *raw, FLGR_Data2D *dat);
  EXPORT_LIB void flgr2d_export_raw_fgUINT16(void *raw, FLGR_Data2D *dat);
  EXPORT_LIB void flgr2d_export_raw_fgUINT32(void *raw, FLGR_Data2D *dat);
  EXPORT_LIB void flgr2d_export_raw_fgINT8(void *raw, FLGR_Data2D *dat);
  EXPORT_LIB void flgr2d_export_raw_fgINT16(void *raw, FLGR_Data2D *dat);
  EXPORT_LIB void flgr2d_export_raw_fgINT32(void *raw, FLGR_Data2D *dat);
  EXPORT_LIB void flgr2d_export_raw_fgFLOAT32(void *raw, FLGR_Data2D *dat);
  EXPORT_LIB void flgr2d_export_raw_fgFLOAT64(void *raw, FLGR_Data2D *dat);



  EXPORT_LIB void flgr1d_copy_fgBIT_fgBIT(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgUINT8_fgBIT(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgUINT16_fgBIT(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgUINT32_fgBIT(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgINT8_fgBIT(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgINT16_fgBIT(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgINT32_fgBIT(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgFLOAT32_fgBIT(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgFLOAT64_fgUINT8(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);

  EXPORT_LIB void flgr1d_copy_fgBIT_fgUINT8(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgUINT8_fgUINT8(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgUINT16_fgUINT8(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgUINT32_fgUINT8(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgINT8_fgUINT8(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgINT16_fgUINT8(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgINT32_fgUINT8(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgFLOAT32_fgUINT8(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgFLOAT64_fgUINT8(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);

  EXPORT_LIB void flgr1d_copy_fgBIT_fgUINT16(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgUINT8_fgUINT16(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgUINT16_fgUINT16(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgUINT32_fgUINT16(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgINT8_fgUINT16(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgINT16_fgUINT16(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgINT32_fgUINT16(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgFLOAT32_fgUINT16(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgFLOAT64_fgUINT16(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);

  EXPORT_LIB void flgr1d_copy_fgBIT_fgUINT32(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgUINT8_fgUINT32(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgUINT16_fgUINT32(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgUINT32_fgUINT32(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgINT8_fgUINT32(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgINT16_fgUINT32(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgINT32_fgUINT32(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgFLOAT32_fgUINT32(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgFLOAT64_fgUINT32(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);

  EXPORT_LIB void flgr1d_copy_fgBIT_fgINT8(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgUINT8_fgINT8(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgUINT16_fgINT8(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgUINT32_fgINT8(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgINT8_fgINT8(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgINT16_fgINT8(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgINT32_fgINT8(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgFLOAT32_fgINT8(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgFLOAT64_fgINT8(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);

  EXPORT_LIB void flgr1d_copy_fgBIT_fgINT16(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgUINT8_fgINT16(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgUINT16_fgINT16(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgUINT32_fgINT16(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgINT8_fgINT16(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgINT16_fgINT16(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgINT32_fgINT16(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgFLOAT32_fgINT16(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgFLOAT64_fgINT16(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);

  EXPORT_LIB void flgr1d_copy_fgBIT_fgINT32(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgUINT8_fgINT32(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgUINT16_fgINT32(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgUINT32_fgINT32(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgINT8_fgINT32(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgINT16_fgINT32(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgINT32_fgINT32(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgFLOAT32_fgINT32(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgFLOAT64_fgINT32(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);

  EXPORT_LIB void flgr1d_copy_fgBIT_fgFLOAT32(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgUINT8_fgFLOAT32(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgUINT16_fgFLOAT32(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgUINT32_fgFLOAT32(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgINT8_fgFLOAT32(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgINT16_fgFLOAT32(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgINT32_fgFLOAT32(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgFLOAT32_fgFLOAT32(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgFLOAT64_fgFLOAT32(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);

  EXPORT_LIB void flgr1d_copy_fgBIT_fgFLOAT64(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgUINT8_fgFLOAT64(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgUINT16_fgFLOAT64(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgUINT32_fgFLOAT64(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgINT8_fgFLOAT64(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgINT16_fgFLOAT64(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgINT32_fgFLOAT64(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgFLOAT32_fgFLOAT64(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);
  EXPORT_LIB void flgr1d_copy_fgFLOAT64_fgFLOAT64(FLGR_Data1D *datdest, FLGR_Data1D *datsrc);









  EXPORT_LIB void flgr2d_copy_fgBIT_fgBIT(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgUINT8_fgBIT(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgUINT16_fgBIT(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgUINT32_fgBIT(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgINT8_fgBIT(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgINT16_fgBIT(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgINT32_fgBIT(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgFLOAT32_fgBIT(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgFLOAT64_fgBIT(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);

  EXPORT_LIB void flgr2d_copy_fgBIT_fgUINT8(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgUINT8_fgUINT8(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgUINT16_fgUINT8(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgUINT32_fgUINT8(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgINT8_fgUINT8(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgINT16_fgUINT8(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgINT32_fgUINT8(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgFLOAT32_fgUINT8(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgFLOAT64_fgUINT8(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);

  EXPORT_LIB void flgr2d_copy_fgBIT_fgUINT16(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgUINT8_fgUINT16(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgUINT16_fgUINT16(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgUINT32_fgUINT16(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgINT8_fgUINT16(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgINT16_fgUINT16(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgINT32_fgUINT16(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgFLOAT32_fgUINT16(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgFLOAT64_fgUINT16(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);

  EXPORT_LIB void flgr2d_copy_fgBIT_fgUINT32(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgUINT8_fgUINT32(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgUINT16_fgUINT32(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgUINT32_fgUINT32(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgINT8_fgUINT32(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgINT16_fgUINT32(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgINT32_fgUINT32(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgFLOAT32_fgUINT32(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgFLOAT64_fgUINT32(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);

  EXPORT_LIB void flgr2d_copy_fgBIT_fgINT8(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgUINT8_fgINT8(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgUINT16_fgINT8(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgUINT32_fgINT8(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgINT8_fgINT8(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgINT16_fgINT8(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgINT32_fgINT8(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgFLOAT32_fgINT8(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgFLOAT64_fgINT8(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);

  EXPORT_LIB void flgr2d_copy_fgBIT_fgINT16(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgUINT8_fgINT16(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgUINT16_fgINT16(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgUINT32_fgINT16(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgINT8_fgINT16(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgINT16_fgINT16(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgINT32_fgINT16(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgFLOAT32_fgINT16(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgFLOAT64_fgINT16(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);

  EXPORT_LIB void flgr2d_copy_fgBIT_fgINT32(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgUINT8_fgINT32(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgUINT16_fgINT32(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgUINT32_fgINT32(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgINT8_fgINT32(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgINT16_fgINT32(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgINT32_fgINT32(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgFLOAT32_fgINT32(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgFLOAT64_fgINT32(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);

  EXPORT_LIB void flgr2d_copy_fgBIT_fgFLOAT32(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgUINT8_fgFLOAT32(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgUINT16_fgFLOAT32(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgUINT32_fgFLOAT32(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgINT8_fgFLOAT32(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgINT16_fgFLOAT32(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgINT32_fgFLOAT32(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgFLOAT32_fgFLOAT32(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgFLOAT64_fgFLOAT32(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);

  EXPORT_LIB void flgr2d_copy_fgBIT_fgFLOAT64(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgUINT8_fgFLOAT64(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgUINT16_fgFLOAT64(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgUINT32_fgFLOAT64(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgINT8_fgFLOAT64(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgINT16_fgFLOAT64(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgINT32_fgFLOAT64(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgFLOAT32_fgFLOAT64(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);
  EXPORT_LIB void flgr2d_copy_fgFLOAT64_fgFLOAT64(FLGR_Data2D *datdest, FLGR_Data2D *datsrc);



#endif

#ifdef __cplusplus
}
#endif
