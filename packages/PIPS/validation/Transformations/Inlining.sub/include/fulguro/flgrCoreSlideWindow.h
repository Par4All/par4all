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

#ifndef __FLGR_CORE_SLIDE_WINDOW_H
#define __FLGR_CORE_SLIDE_WINDOW_H

#include <flgrCoreDll.h>
#include <flgrCoreErrors.h>
#include <flgrCoreTypes.h>
#include <flgrCoreData.h>
#include <flgrCoreNhbManage.h>

  /*!
   * \addtogroup group_fulguro_core
   * @{
   */

  //! pointer to a function doing computation over an extracted neighborhood
  typedef void (*FLGR_ComputeNhb2D)    (FLGR_Vector *result, FLGR_NhbBox2D *extr); 
  //! pointer to a function doing computation over an extracted neighborhood
  typedef void (*FLGR_ComputeNhb1D)    (FLGR_Vector *result, FLGR_NhbBox1D *extr); 
  //! pointer to a function doing arith computation over vectors
  typedef void (*FLGR_ComputeArith)    (FLGR_Vector *vector_dest, FLGR_Vector *vector_1, FLGR_Vector *vector_2); 


  //! @}

 
  EXPORT_LIB FLGR_Ret flgr1d_apply_raster_scan_method(FLGR_Data1D *nhb);
  EXPORT_LIB FLGR_Ret flgr1d_apply_anti_raster_scan_method(FLGR_Data1D *nhb);

  EXPORT_LIB FLGR_Ret flgr2d_apply_raster_scan_method(FLGR_Data2D *nhb);
  EXPORT_LIB FLGR_Ret flgr2d_apply_anti_raster_scan_method(FLGR_Data2D *nhb);



  EXPORT_LIB FLGR_Ret flgr1d_raster_slide_window(FLGR_Data1D *imgdest,FLGR_Data1D *imgsrc,FLGR_Data1D *nhb,int nhb_sym,
						 FLGR_ComputeNhb1D getNhbVal);

  EXPORT_LIB FLGR_Ret flgr1d_anti_raster_slide_window(FLGR_Data1D *imgdest,FLGR_Data1D *imgsrc,FLGR_Data1D *nhb,int nhb_sym,
						      FLGR_ComputeNhb1D getNhbVal);

  EXPORT_LIB FLGR_Ret flgr1d_raster_slide_window_unfolded(FLGR_Data1D *imgdest,FLGR_Data1D *imgsrc,FLGR_Data1D *nhb,int nhb_sym,
							  FLGR_ComputeNhb1D getNhbVal);

  EXPORT_LIB FLGR_Ret flgr1d_anti_raster_slide_window_unfolded(FLGR_Data1D *imgdest,FLGR_Data1D *imgsrc,FLGR_Data1D *nhb,int nhb_sym,
							       FLGR_ComputeNhb1D getNhbVal);
 
  EXPORT_LIB FLGR_Ret flgr1d_raster_slide_structuring_function(FLGR_Data1D *datdest, FLGR_Data1D *datsrc, 
							       FLGR_Data1D *semap, FLGR_Shape shape, 
							       FLGR_ComputeNhb1D computeNhb);






  EXPORT_LIB FLGR_Ret flgr2d_raster_slide_window(FLGR_Data2D *imgdest,FLGR_Data2D *imgsrc,FLGR_Data2D *nhb,int nhb_sym,
						 FLGR_ComputeNhb2D getNhbVal);

  EXPORT_LIB FLGR_Ret flgr2d_anti_raster_slide_window(FLGR_Data2D *imgdest,FLGR_Data2D *imgsrc,FLGR_Data2D *nhb,int nhb_sym,
						      FLGR_ComputeNhb2D getNhbVal);

  EXPORT_LIB FLGR_Ret flgr2d_raster_slide_window_unfolded(FLGR_Data2D *imgdest,FLGR_Data2D *imgsrc,FLGR_Data2D *nhb,int nhb_sym,
							  FLGR_ComputeNhb2D getNhbVal);

  EXPORT_LIB FLGR_Ret flgr2d_anti_raster_slide_window_unfolded(FLGR_Data2D *imgdest,FLGR_Data2D *imgsrc,FLGR_Data2D *nhb,int nhb_sym,
							       FLGR_ComputeNhb2D getNhbVal);

  EXPORT_LIB FLGR_Ret flgr2d_raster_slide_structuring_function(FLGR_Data2D *datdest, FLGR_Data2D *datsrc, 
							       FLGR_Data2D *semap, FLGR_Shape shape, 
							       FLGR_ComputeNhb2D computeNhb);




  EXPORT_LIB void flgr1d_raster_slide_structuring_function_fgBIT(FLGR_Data1D *datdest, FLGR_Data1D *datsrc, 
								 FLGR_Data1D *semap, FLGR_Shape shape, 
								 FLGR_ComputeNhb1D computeNhb);
  EXPORT_LIB void flgr1d_raster_slide_structuring_function_fgUINT8(FLGR_Data1D *datdest, FLGR_Data1D *datsrc, 
								   FLGR_Data1D *semap, FLGR_Shape shape, 
								   FLGR_ComputeNhb1D computeNhb);
  EXPORT_LIB void flgr1d_raster_slide_structuring_function_fgUINT16(FLGR_Data1D *datdest, FLGR_Data1D *datsrc, 
								    FLGR_Data1D *semap, FLGR_Shape shape, 
								    FLGR_ComputeNhb1D computeNhb);
  EXPORT_LIB void flgr1d_raster_slide_structuring_function_fgUINT32(FLGR_Data1D *datdest, FLGR_Data1D *datsrc, 
								    FLGR_Data1D *semap, FLGR_Shape shape, 
								    FLGR_ComputeNhb1D computeNhb);
  EXPORT_LIB void flgr1d_raster_slide_structuring_function_fgINT8(FLGR_Data1D *datdest, FLGR_Data1D *datsrc, 
								  FLGR_Data1D *semap, FLGR_Shape shape, 
								  FLGR_ComputeNhb1D computeNhb);
  EXPORT_LIB void flgr1d_raster_slide_structuring_function_fgINT16(FLGR_Data1D *datdest, FLGR_Data1D *datsrc, 
								   FLGR_Data1D *semap, FLGR_Shape shape, 
								   FLGR_ComputeNhb1D computeNhb);
  EXPORT_LIB void flgr1d_raster_slide_structuring_function_fgINT32(FLGR_Data1D *datdest, FLGR_Data1D *datsrc, 
								   FLGR_Data1D *semap, FLGR_Shape shape, 
								   FLGR_ComputeNhb1D computeNhb);
  EXPORT_LIB void flgr1d_raster_slide_structuring_function_fgFLOAT32(FLGR_Data1D *datdest, FLGR_Data1D *datsrc, 
								     FLGR_Data1D *semap, FLGR_Shape shape,
								     FLGR_ComputeNhb1D computeNhb);
  EXPORT_LIB void flgr1d_raster_slide_structuring_function_fgFLOAT64(FLGR_Data1D *datdest, FLGR_Data1D *datsrc, 
								     FLGR_Data1D *semap, FLGR_Shape shape, 
								     FLGR_ComputeNhb1D computeNhb);



  EXPORT_LIB void flgr2d_raster_slide_structuring_function_fgBIT(FLGR_Data2D *datdest, FLGR_Data2D *datsrc, 
								 FLGR_Data2D *semap, FLGR_Shape shape, 
								 FLGR_ComputeNhb2D computeNhb);
  EXPORT_LIB void flgr2d_raster_slide_structuring_function_fgUINT8(FLGR_Data2D *datdest, FLGR_Data2D *datsrc, 
								   FLGR_Data2D *semap, FLGR_Shape shape, 
								   FLGR_ComputeNhb2D computeNhb);
  EXPORT_LIB void flgr2d_raster_slide_structuring_function_fgUINT16(FLGR_Data2D *datdest, FLGR_Data2D *datsrc, 
								    FLGR_Data2D *semap, FLGR_Shape shape, 
								    FLGR_ComputeNhb2D computeNhb);
  EXPORT_LIB void flgr2d_raster_slide_structuring_function_fgUINT32(FLGR_Data2D *datdest, FLGR_Data2D *datsrc, 
								    FLGR_Data2D *semap, FLGR_Shape shape, 
								    FLGR_ComputeNhb2D computeNhb);
  EXPORT_LIB void flgr2d_raster_slide_structuring_function_fgINT8(FLGR_Data2D *datdest, FLGR_Data2D *datsrc, 
								  FLGR_Data2D *semap, FLGR_Shape shape, 
								  FLGR_ComputeNhb2D computeNhb);
  EXPORT_LIB void flgr2d_raster_slide_structuring_function_fgINT16(FLGR_Data2D *datdest, FLGR_Data2D *datsrc, 
								   FLGR_Data2D *semap, FLGR_Shape shape, 
								   FLGR_ComputeNhb2D computeNhb);
  EXPORT_LIB void flgr2d_raster_slide_structuring_function_fgINT32(FLGR_Data2D *datdest, FLGR_Data2D *datsrc, 
								   FLGR_Data2D *semap, FLGR_Shape shape, 
								   FLGR_ComputeNhb2D computeNhb);
  EXPORT_LIB void flgr2d_raster_slide_structuring_function_fgFLOAT32(FLGR_Data2D *datdest, FLGR_Data2D *datsrc, 
								     FLGR_Data2D *semap, FLGR_Shape shape,
								     FLGR_ComputeNhb2D computeNhb);
  EXPORT_LIB void flgr2d_raster_slide_structuring_function_fgFLOAT64(FLGR_Data2D *datdest, FLGR_Data2D *datsrc, 
								     FLGR_Data2D *semap, FLGR_Shape shape, 
								     FLGR_ComputeNhb2D computeNhb);








  EXPORT_LIB void flgr1d_apply_raster_scan_method_fgBIT(FLGR_Data1D *nhb);
  EXPORT_LIB void flgr1d_apply_raster_scan_method_fgUINT8(FLGR_Data1D *nhb);
  EXPORT_LIB void flgr1d_apply_raster_scan_method_fgUINT16(FLGR_Data1D *nhb);
  EXPORT_LIB void flgr1d_apply_raster_scan_method_fgUINT32(FLGR_Data1D *nhb);
  EXPORT_LIB void flgr1d_apply_raster_scan_method_fgINT8(FLGR_Data1D *nhb);
  EXPORT_LIB void flgr1d_apply_raster_scan_method_fgINT16(FLGR_Data1D *nhb);
  EXPORT_LIB void flgr1d_apply_raster_scan_method_fgINT32(FLGR_Data1D *nhb);
  EXPORT_LIB void flgr1d_apply_raster_scan_method_fgFLOAT32(FLGR_Data1D *nhb);
  EXPORT_LIB void flgr1d_apply_raster_scan_method_fgFLOAT64(FLGR_Data1D *nhb);

  EXPORT_LIB void flgr1d_apply_anti_raster_scan_method_fgBIT(FLGR_Data1D *nhb);
  EXPORT_LIB void flgr1d_apply_anti_raster_scan_method_fgUINT8(FLGR_Data1D *nhb);
  EXPORT_LIB void flgr1d_apply_anti_raster_scan_method_fgUINT16(FLGR_Data1D *nhb);
  EXPORT_LIB void flgr1d_apply_anti_raster_scan_method_fgUINT32(FLGR_Data1D *nhb);
  EXPORT_LIB void flgr1d_apply_anti_raster_scan_method_fgINT8(FLGR_Data1D *nhb);
  EXPORT_LIB void flgr1d_apply_anti_raster_scan_method_fgINT16(FLGR_Data1D *nhb);
  EXPORT_LIB void flgr1d_apply_anti_raster_scan_method_fgINT32(FLGR_Data1D *nhb);
  EXPORT_LIB void flgr1d_apply_anti_raster_scan_method_fgFLOAT32(FLGR_Data1D *nhb);
  EXPORT_LIB void flgr1d_apply_anti_raster_scan_method_fgFLOAT64(FLGR_Data1D *nhb);



  EXPORT_LIB void flgr1d_raster_slide_window_fgBIT(FLGR_Data1D *datdest,FLGR_Data1D *datsrc,FLGR_Data1D *nhb,int nhb_sym,
						 FLGR_ComputeNhb1D getNhbVal);
  
  EXPORT_LIB void flgr1d_raster_slide_window_fgUINT8(FLGR_Data1D *datdest,FLGR_Data1D *datsrc,FLGR_Data1D *nhb,int nhb_sym,
						   FLGR_ComputeNhb1D getNhbVal);
  
  EXPORT_LIB void flgr1d_raster_slide_window_fgUINT16(FLGR_Data1D *datdest,FLGR_Data1D *datsrc,FLGR_Data1D *nhb,int nhb_sym,
						    FLGR_ComputeNhb1D getNhbVal);
  
  EXPORT_LIB void flgr1d_raster_slide_window_fgUINT32(FLGR_Data1D *datdest,FLGR_Data1D *datsrc,FLGR_Data1D *nhb,int nhb_sym,
						    FLGR_ComputeNhb1D getNhbVal);
  
  EXPORT_LIB void flgr1d_raster_slide_window_fgINT8(FLGR_Data1D *datdest,FLGR_Data1D *datsrc,FLGR_Data1D *nhb,int nhb_sym,
						  FLGR_ComputeNhb1D getNhbVal);
  
  EXPORT_LIB void flgr1d_raster_slide_window_fgINT16(FLGR_Data1D *datdest,FLGR_Data1D *datsrc,FLGR_Data1D *nhb,int nhb_sym,
						   FLGR_ComputeNhb1D getNhbVal);

  EXPORT_LIB void flgr1d_raster_slide_window_fgINT32(FLGR_Data1D *datdest,FLGR_Data1D *datsrc,FLGR_Data1D *nhb,int nhb_sym,
						   FLGR_ComputeNhb1D getNhbVal);

  EXPORT_LIB void flgr1d_raster_slide_window_fgFLOAT32(FLGR_Data1D *datdest,FLGR_Data1D *datsrc,FLGR_Data1D *nhb,int nhb_sym,
						     FLGR_ComputeNhb1D getNhbVal);

  EXPORT_LIB void flgr1d_raster_slide_window_fgFLOAT64(FLGR_Data1D *datdest,FLGR_Data1D *datsrc,FLGR_Data1D *nhb,int nhb_sym,
						     FLGR_ComputeNhb1D getNhbVal);



  EXPORT_LIB void flgr1d_anti_raster_slide_window_fgBIT(FLGR_Data1D *datdest,FLGR_Data1D *datsrc,FLGR_Data1D *nhb,int nhb_sym,
						      FLGR_ComputeNhb1D getNhbVal);

  EXPORT_LIB void flgr1d_anti_raster_slide_window_fgUINT8(FLGR_Data1D *datdest,FLGR_Data1D *datsrc,FLGR_Data1D *nhb,int nhb_sym,
							FLGR_ComputeNhb1D getNhbVal);

  EXPORT_LIB void flgr1d_anti_raster_slide_window_fgUINT16(FLGR_Data1D *datdest,FLGR_Data1D *datsrc,FLGR_Data1D *nhb,int nhb_sym,
							 FLGR_ComputeNhb1D getNhbVal);

  EXPORT_LIB void flgr1d_anti_raster_slide_window_fgUINT32(FLGR_Data1D *datdest,FLGR_Data1D *datsrc,FLGR_Data1D *nhb,int nhb_sym,
							 FLGR_ComputeNhb1D getNhbVal);

  EXPORT_LIB void flgr1d_anti_raster_slide_window_fgINT8(FLGR_Data1D *datdest,FLGR_Data1D *datsrc,FLGR_Data1D *nhb,int nhb_sym,
						       FLGR_ComputeNhb1D getNhbVal);

  EXPORT_LIB void flgr1d_anti_raster_slide_window_fgINT16(FLGR_Data1D *datdest,FLGR_Data1D *datsrc,FLGR_Data1D *nhb,int nhb_sym,
							FLGR_ComputeNhb1D getNhbVal);

  EXPORT_LIB void flgr1d_anti_raster_slide_window_fgINT32(FLGR_Data1D *datdest,FLGR_Data1D *datsrc,FLGR_Data1D *nhb,int nhb_sym,
							FLGR_ComputeNhb1D getNhbVal);

  EXPORT_LIB void flgr1d_anti_raster_slide_window_fgFLOAT32(FLGR_Data1D *datdest,FLGR_Data1D *datsrc,FLGR_Data1D *nhb,int nhb_sym,
							  FLGR_ComputeNhb1D getNhbVal);

  EXPORT_LIB void flgr1d_anti_raster_slide_window_fgFLOAT64(FLGR_Data1D *datdest,FLGR_Data1D *datsrc,FLGR_Data1D *nhb,int nhb_sym,
							  FLGR_ComputeNhb1D getNhbVal);





  EXPORT_LIB void flgr1d_raster_slide_window_unfolded_fgBIT(FLGR_Data1D *datdest,FLGR_Data1D *datsrc,FLGR_Data1D *nhb,int nhb_sym,
						 FLGR_ComputeNhb1D getNhbVal);
  
  EXPORT_LIB void flgr1d_raster_slide_window_unfolded_fgUINT8(FLGR_Data1D *datdest,FLGR_Data1D *datsrc,FLGR_Data1D *nhb,int nhb_sym,
						   FLGR_ComputeNhb1D getNhbVal);
  
  EXPORT_LIB void flgr1d_raster_slide_window_unfolded_fgUINT16(FLGR_Data1D *datdest,FLGR_Data1D *datsrc,FLGR_Data1D *nhb,int nhb_sym,
						    FLGR_ComputeNhb1D getNhbVal);
  
  EXPORT_LIB void flgr1d_raster_slide_window_unfolded_fgUINT32(FLGR_Data1D *datdest,FLGR_Data1D *datsrc,FLGR_Data1D *nhb,int nhb_sym,
						    FLGR_ComputeNhb1D getNhbVal);
  
  EXPORT_LIB void flgr1d_raster_slide_window_unfolded_fgINT8(FLGR_Data1D *datdest,FLGR_Data1D *datsrc,FLGR_Data1D *nhb,int nhb_sym,
						  FLGR_ComputeNhb1D getNhbVal);
  
  EXPORT_LIB void flgr1d_raster_slide_window_unfolded_fgINT16(FLGR_Data1D *datdest,FLGR_Data1D *datsrc,FLGR_Data1D *nhb,int nhb_sym,
						   FLGR_ComputeNhb1D getNhbVal);

  EXPORT_LIB void flgr1d_raster_slide_window_unfolded_fgINT32(FLGR_Data1D *datdest,FLGR_Data1D *datsrc,FLGR_Data1D *nhb,int nhb_sym,
						   FLGR_ComputeNhb1D getNhbVal);

  EXPORT_LIB void flgr1d_raster_slide_window_unfolded_fgFLOAT32(FLGR_Data1D *datdest,FLGR_Data1D *datsrc,FLGR_Data1D *nhb,int nhb_sym,
						     FLGR_ComputeNhb1D getNhbVal);

  EXPORT_LIB void flgr1d_raster_slide_window_unfolded_fgFLOAT64(FLGR_Data1D *datdest,FLGR_Data1D *datsrc,FLGR_Data1D *nhb,int nhb_sym,
						     FLGR_ComputeNhb1D getNhbVal);



  EXPORT_LIB void flgr1d_anti_raster_slide_window_unfolded_fgBIT(FLGR_Data1D *datdest,FLGR_Data1D *datsrc,FLGR_Data1D *nhb,int nhb_sym,
						      FLGR_ComputeNhb1D getNhbVal);

  EXPORT_LIB void flgr1d_anti_raster_slide_window_unfolded_fgUINT8(FLGR_Data1D *datdest,FLGR_Data1D *datsrc,FLGR_Data1D *nhb,int nhb_sym,
							FLGR_ComputeNhb1D getNhbVal);

  EXPORT_LIB void flgr1d_anti_raster_slide_window_unfolded_fgUINT16(FLGR_Data1D *datdest,FLGR_Data1D *datsrc,FLGR_Data1D *nhb,int nhb_sym,
							 FLGR_ComputeNhb1D getNhbVal);

  EXPORT_LIB void flgr1d_anti_raster_slide_window_unfolded_fgUINT32(FLGR_Data1D *datdest,FLGR_Data1D *datsrc,FLGR_Data1D *nhb,int nhb_sym,
							 FLGR_ComputeNhb1D getNhbVal);

  EXPORT_LIB void flgr1d_anti_raster_slide_window_unfolded_fgINT8(FLGR_Data1D *datdest,FLGR_Data1D *datsrc,FLGR_Data1D *nhb,int nhb_sym,
						       FLGR_ComputeNhb1D getNhbVal);

  EXPORT_LIB void flgr1d_anti_raster_slide_window_unfolded_fgINT16(FLGR_Data1D *datdest,FLGR_Data1D *datsrc,FLGR_Data1D *nhb,int nhb_sym,
							FLGR_ComputeNhb1D getNhbVal);

  EXPORT_LIB void flgr1d_anti_raster_slide_window_unfolded_fgINT32(FLGR_Data1D *datdest,FLGR_Data1D *datsrc,FLGR_Data1D *nhb,int nhb_sym,
							FLGR_ComputeNhb1D getNhbVal);

  EXPORT_LIB void flgr1d_anti_raster_slide_window_unfolded_fgFLOAT32(FLGR_Data1D *datdest,FLGR_Data1D *datsrc,FLGR_Data1D *nhb,int nhb_sym,
							  FLGR_ComputeNhb1D getNhbVal);

  EXPORT_LIB void flgr1d_anti_raster_slide_window_unfolded_fgFLOAT64(FLGR_Data1D *datdest,FLGR_Data1D *datsrc,FLGR_Data1D *nhb,int nhb_sym,
							  FLGR_ComputeNhb1D getNhbVal);









  EXPORT_LIB void flgr1d_raster_slide_window_before_op_fgBIT(FLGR_Data1D *datdest, FLGR_Data1D *datsrc, FLGR_Data1D *datarith, 
							   FLGR_Data1D *nhb,int nhb_sym,
							   FLGR_ComputeNhb1D computeNhb,
							   FLGR_ComputeArith computeArith);

  EXPORT_LIB void flgr1d_raster_slide_window_before_op_fgUINT8(FLGR_Data1D *datdest, FLGR_Data1D *datsrc, FLGR_Data1D *datarith, 
							     FLGR_Data1D *nhb,int nhb_sym,
							     FLGR_ComputeNhb1D computeNhb,
							     FLGR_ComputeArith computeArith);

  EXPORT_LIB void flgr1d_raster_slide_window_before_op_fgUINT16(FLGR_Data1D *datdest, FLGR_Data1D *datsrc, FLGR_Data1D *datarith,
							      FLGR_Data1D *nhb,int nhb_sym,
							      FLGR_ComputeNhb1D computeNhb,
							      FLGR_ComputeArith computeArith);

  EXPORT_LIB void flgr1d_raster_slide_window_before_op_fgUINT32(FLGR_Data1D *datdest, FLGR_Data1D *datsrc, FLGR_Data1D *datarith,
							      FLGR_Data1D *nhb,int nhb_sym,
							      FLGR_ComputeNhb1D computeNhb,
							      FLGR_ComputeArith computeArith);

  EXPORT_LIB void flgr1d_raster_slide_window_before_op_fgINT8(FLGR_Data1D *datdest, FLGR_Data1D *datsrc, FLGR_Data1D *datarith,
							    FLGR_Data1D *nhb,int nhb_sym,
							    FLGR_ComputeNhb1D computeNhb,
							    FLGR_ComputeArith computeArith);

  EXPORT_LIB void flgr1d_raster_slide_window_before_op_fgINT16(FLGR_Data1D *datdest, FLGR_Data1D *datsrc, FLGR_Data1D *datarith,
							     FLGR_Data1D *nhb,int nhb_sym,
							     FLGR_ComputeNhb1D computeNhb,
							     FLGR_ComputeArith computeArith);

  EXPORT_LIB void flgr1d_raster_slide_window_before_op_fgINT32(FLGR_Data1D *datdest, FLGR_Data1D *datsrc, FLGR_Data1D *datarith,
							     FLGR_Data1D *nhb,int nhb_sym,
							     FLGR_ComputeNhb1D computeNhb,
							     FLGR_ComputeArith computeArith);

  EXPORT_LIB void flgr1d_raster_slide_window_before_op_fgFLOAT32(FLGR_Data1D *datdest, FLGR_Data1D *datsrc, FLGR_Data1D *datarith,
							       FLGR_Data1D *nhb,int nhb_sym,
							       FLGR_ComputeNhb1D computeNhb,
							       FLGR_ComputeArith computeArith);

  EXPORT_LIB void flgr1d_raster_slide_window_before_op_fgFLOAT64(FLGR_Data1D *datdest, FLGR_Data1D *datsrc, FLGR_Data1D *datarith, 
							       FLGR_Data1D *nhb,int nhb_sym,
							       FLGR_ComputeNhb1D computeNhb,
							       FLGR_ComputeArith computeArith);



  EXPORT_LIB void flgr1d_anti_raster_slide_window_before_op_fgBIT(FLGR_Data1D *datdest, FLGR_Data1D *datsrc, FLGR_Data1D *datarith,
								FLGR_Data1D *nhb,int nhb_sym,
								FLGR_ComputeNhb1D computeNhb,
								FLGR_ComputeArith computeArith);

  EXPORT_LIB void flgr1d_anti_raster_slide_window_before_op_fgUINT8(FLGR_Data1D *datdest, FLGR_Data1D *datsrc, FLGR_Data1D *datarith,
								  FLGR_Data1D *nhb,int nhb_sym,
								  FLGR_ComputeNhb1D computeNhb,
								  FLGR_ComputeArith computeArith);

  EXPORT_LIB void flgr1d_anti_raster_slide_window_before_op_fgUINT16(FLGR_Data1D *datdest, FLGR_Data1D *datsrc, FLGR_Data1D *datarith,
								   FLGR_Data1D *nhb,int nhb_sym,
								   FLGR_ComputeNhb1D computeNhb,
								   FLGR_ComputeArith computeArith);

  EXPORT_LIB void flgr1d_anti_raster_slide_window_before_op_fgUINT32(FLGR_Data1D *datdest, FLGR_Data1D *datsrc, FLGR_Data1D *datarith,
								   FLGR_Data1D *nhb,int nhb_sym,
								   FLGR_ComputeNhb1D computeNhb,
								   FLGR_ComputeArith computeArith);

  EXPORT_LIB void flgr1d_anti_raster_slide_window_before_op_fgINT8(FLGR_Data1D *datdest, FLGR_Data1D *datsrc, FLGR_Data1D *datarith,
								 FLGR_Data1D *nhb,int nhb_sym,
								 FLGR_ComputeNhb1D computeNhb,
								 FLGR_ComputeArith computeArith);

  EXPORT_LIB void flgr1d_anti_raster_slide_window_before_op_fgINT16(FLGR_Data1D *datdest, FLGR_Data1D *datsrc, FLGR_Data1D *datarith,
								  FLGR_Data1D *nhb,int nhb_sym,
								  FLGR_ComputeNhb1D computeNhb,
								  FLGR_ComputeArith computeArith);

  EXPORT_LIB void flgr1d_anti_raster_slide_window_before_op_fgINT32(FLGR_Data1D *datdest, FLGR_Data1D *datsrc, FLGR_Data1D *datarith,
								  FLGR_Data1D *nhb,int nhb_sym,
								  FLGR_ComputeNhb1D computeNhb,
								  FLGR_ComputeArith computeArith);

  EXPORT_LIB void flgr1d_anti_raster_slide_window_before_op_fgFLOAT32(FLGR_Data1D *datdest, FLGR_Data1D *datsrc, FLGR_Data1D *datarith,
								    FLGR_Data1D *nhb,int nhb_sym,
								    FLGR_ComputeNhb1D computeNhb,
								    FLGR_ComputeArith computeArith);

  EXPORT_LIB void flgr1d_anti_raster_slide_window_before_op_fgFLOAT64(FLGR_Data1D *datdest, FLGR_Data1D *datsrc, FLGR_Data1D *datarith,
								    FLGR_Data1D *nhb,int nhb_sym,
								    FLGR_ComputeNhb1D computeNhb,
								    FLGR_ComputeArith computeArith);





























  EXPORT_LIB void flgr2d_apply_raster_scan_method_fgBIT(FLGR_Data2D *nhb);
  EXPORT_LIB void flgr2d_apply_raster_scan_method_fgUINT8(FLGR_Data2D *nhb);
  EXPORT_LIB void flgr2d_apply_raster_scan_method_fgUINT16(FLGR_Data2D *nhb);
  EXPORT_LIB void flgr2d_apply_raster_scan_method_fgUINT32(FLGR_Data2D *nhb);
  EXPORT_LIB void flgr2d_apply_raster_scan_method_fgINT8(FLGR_Data2D *nhb);
  EXPORT_LIB void flgr2d_apply_raster_scan_method_fgINT16(FLGR_Data2D *nhb);
  EXPORT_LIB void flgr2d_apply_raster_scan_method_fgINT32(FLGR_Data2D *nhb);
  EXPORT_LIB void flgr2d_apply_raster_scan_method_fgFLOAT32(FLGR_Data2D *nhb);
  EXPORT_LIB void flgr2d_apply_raster_scan_method_fgFLOAT64(FLGR_Data2D *nhb);

  EXPORT_LIB void flgr2d_apply_anti_raster_scan_method_fgBIT(FLGR_Data2D *nhb);
  EXPORT_LIB void flgr2d_apply_anti_raster_scan_method_fgUINT8(FLGR_Data2D *nhb);
  EXPORT_LIB void flgr2d_apply_anti_raster_scan_method_fgUINT16(FLGR_Data2D *nhb);
  EXPORT_LIB void flgr2d_apply_anti_raster_scan_method_fgUINT32(FLGR_Data2D *nhb);
  EXPORT_LIB void flgr2d_apply_anti_raster_scan_method_fgINT8(FLGR_Data2D *nhb);
  EXPORT_LIB void flgr2d_apply_anti_raster_scan_method_fgINT16(FLGR_Data2D *nhb);
  EXPORT_LIB void flgr2d_apply_anti_raster_scan_method_fgINT32(FLGR_Data2D *nhb);
  EXPORT_LIB void flgr2d_apply_anti_raster_scan_method_fgFLOAT32(FLGR_Data2D *nhb);
  EXPORT_LIB void flgr2d_apply_anti_raster_scan_method_fgFLOAT64(FLGR_Data2D *nhb);




  EXPORT_LIB void flgr2d_raster_slide_window_fgBIT(FLGR_Data2D *imgdest,FLGR_Data2D *imgsrc,FLGR_Data2D *nhb,int nhb_sym,
						 FLGR_ComputeNhb2D getNhbVal);
  
  EXPORT_LIB void flgr2d_raster_slide_window_fgUINT8(FLGR_Data2D *imgdest,FLGR_Data2D *imgsrc,FLGR_Data2D *nhb,int nhb_sym,
						   FLGR_ComputeNhb2D getNhbVal);
  
  EXPORT_LIB void flgr2d_raster_slide_window_fgUINT16(FLGR_Data2D *imgdest,FLGR_Data2D *imgsrc,FLGR_Data2D *nhb,int nhb_sym,
						    FLGR_ComputeNhb2D getNhbVal);
  
  EXPORT_LIB void flgr2d_raster_slide_window_fgUINT32(FLGR_Data2D *imgdest,FLGR_Data2D *imgsrc,FLGR_Data2D *nhb,int nhb_sym,
						    FLGR_ComputeNhb2D getNhbVal);
  
  EXPORT_LIB void flgr2d_raster_slide_window_fgINT8(FLGR_Data2D *imgdest,FLGR_Data2D *imgsrc,FLGR_Data2D *nhb,int nhb_sym,
						  FLGR_ComputeNhb2D getNhbVal);
  
  EXPORT_LIB void flgr2d_raster_slide_window_fgINT16(FLGR_Data2D *imgdest,FLGR_Data2D *imgsrc,FLGR_Data2D *nhb,int nhb_sym,
						   FLGR_ComputeNhb2D getNhbVal);

  EXPORT_LIB void flgr2d_raster_slide_window_fgINT32(FLGR_Data2D *imgdest,FLGR_Data2D *imgsrc,FLGR_Data2D *nhb,int nhb_sym,
						   FLGR_ComputeNhb2D getNhbVal);

  EXPORT_LIB void flgr2d_raster_slide_window_fgFLOAT32(FLGR_Data2D *imgdest,FLGR_Data2D *imgsrc,FLGR_Data2D *nhb,int nhb_sym,
						     FLGR_ComputeNhb2D getNhbVal);

  EXPORT_LIB void flgr2d_raster_slide_window_fgFLOAT64(FLGR_Data2D *imgdest,FLGR_Data2D *imgsrc,FLGR_Data2D *nhb,int nhb_sym,
						     FLGR_ComputeNhb2D getNhbVal);



  EXPORT_LIB void flgr2d_anti_raster_slide_window_fgBIT(FLGR_Data2D *imgdest,FLGR_Data2D *imgsrc,FLGR_Data2D *nhb,int nhb_sym,
						      FLGR_ComputeNhb2D getNhbVal);

  EXPORT_LIB void flgr2d_anti_raster_slide_window_fgUINT8(FLGR_Data2D *imgdest,FLGR_Data2D *imgsrc,FLGR_Data2D *nhb,int nhb_sym,
							FLGR_ComputeNhb2D getNhbVal);

  EXPORT_LIB void flgr2d_anti_raster_slide_window_fgUINT16(FLGR_Data2D *imgdest,FLGR_Data2D *imgsrc,FLGR_Data2D *nhb,int nhb_sym,
							 FLGR_ComputeNhb2D getNhbVal);

  EXPORT_LIB void flgr2d_anti_raster_slide_window_fgUINT32(FLGR_Data2D *imgdest,FLGR_Data2D *imgsrc,FLGR_Data2D *nhb,int nhb_sym,
							 FLGR_ComputeNhb2D getNhbVal);

  EXPORT_LIB void flgr2d_anti_raster_slide_window_fgINT8(FLGR_Data2D *imgdest,FLGR_Data2D *imgsrc,FLGR_Data2D *nhb,int nhb_sym,
						       FLGR_ComputeNhb2D getNhbVal);

  EXPORT_LIB void flgr2d_anti_raster_slide_window_fgINT16(FLGR_Data2D *imgdest,FLGR_Data2D *imgsrc,FLGR_Data2D *nhb,int nhb_sym,
							FLGR_ComputeNhb2D getNhbVal);

  EXPORT_LIB void flgr2d_anti_raster_slide_window_fgINT32(FLGR_Data2D *imgdest,FLGR_Data2D *imgsrc,FLGR_Data2D *nhb,int nhb_sym,
							FLGR_ComputeNhb2D getNhbVal);

  EXPORT_LIB void flgr2d_anti_raster_slide_window_fgFLOAT32(FLGR_Data2D *imgdest,FLGR_Data2D *imgsrc,FLGR_Data2D *nhb,int nhb_sym,
							  FLGR_ComputeNhb2D getNhbVal);

  EXPORT_LIB void flgr2d_anti_raster_slide_window_fgFLOAT64(FLGR_Data2D *imgdest,FLGR_Data2D *imgsrc,FLGR_Data2D *nhb,int nhb_sym,
							  FLGR_ComputeNhb2D getNhbVal);





  EXPORT_LIB void flgr2d_raster_slide_window_unfolded_fgBIT(FLGR_Data2D *imgdest,FLGR_Data2D *imgsrc,FLGR_Data2D *nhb,int nhb_sym,
						 FLGR_ComputeNhb2D getNhbVal);
  
  EXPORT_LIB void flgr2d_raster_slide_window_unfolded_fgUINT8(FLGR_Data2D *imgdest,FLGR_Data2D *imgsrc,FLGR_Data2D *nhb,int nhb_sym,
						   FLGR_ComputeNhb2D getNhbVal);
  
  EXPORT_LIB void flgr2d_raster_slide_window_unfolded_fgUINT16(FLGR_Data2D *imgdest,FLGR_Data2D *imgsrc,FLGR_Data2D *nhb,int nhb_sym,
						    FLGR_ComputeNhb2D getNhbVal);
  
  EXPORT_LIB void flgr2d_raster_slide_window_unfolded_fgUINT32(FLGR_Data2D *imgdest,FLGR_Data2D *imgsrc,FLGR_Data2D *nhb,int nhb_sym,
						    FLGR_ComputeNhb2D getNhbVal);
  
  EXPORT_LIB void flgr2d_raster_slide_window_unfolded_fgINT8(FLGR_Data2D *imgdest,FLGR_Data2D *imgsrc,FLGR_Data2D *nhb,int nhb_sym,
						  FLGR_ComputeNhb2D getNhbVal);
  
  EXPORT_LIB void flgr2d_raster_slide_window_unfolded_fgINT16(FLGR_Data2D *imgdest,FLGR_Data2D *imgsrc,FLGR_Data2D *nhb,int nhb_sym,
						   FLGR_ComputeNhb2D getNhbVal);

  EXPORT_LIB void flgr2d_raster_slide_window_unfolded_fgINT32(FLGR_Data2D *imgdest,FLGR_Data2D *imgsrc,FLGR_Data2D *nhb,int nhb_sym,
						   FLGR_ComputeNhb2D getNhbVal);

  EXPORT_LIB void flgr2d_raster_slide_window_unfolded_fgFLOAT32(FLGR_Data2D *imgdest,FLGR_Data2D *imgsrc,FLGR_Data2D *nhb,int nhb_sym,
						     FLGR_ComputeNhb2D getNhbVal);

  EXPORT_LIB void flgr2d_raster_slide_window_unfolded_fgFLOAT64(FLGR_Data2D *imgdest,FLGR_Data2D *imgsrc,FLGR_Data2D *nhb,int nhb_sym,
						     FLGR_ComputeNhb2D getNhbVal);



  EXPORT_LIB void flgr2d_anti_raster_slide_window_unfolded_fgBIT(FLGR_Data2D *imgdest,FLGR_Data2D *imgsrc,FLGR_Data2D *nhb,int nhb_sym,
						      FLGR_ComputeNhb2D getNhbVal);

  EXPORT_LIB void flgr2d_anti_raster_slide_window_unfolded_fgUINT8(FLGR_Data2D *imgdest,FLGR_Data2D *imgsrc,FLGR_Data2D *nhb,int nhb_sym,
							FLGR_ComputeNhb2D getNhbVal);

  EXPORT_LIB void flgr2d_anti_raster_slide_window_unfolded_fgUINT16(FLGR_Data2D *imgdest,FLGR_Data2D *imgsrc,FLGR_Data2D *nhb,int nhb_sym,
							 FLGR_ComputeNhb2D getNhbVal);

  EXPORT_LIB void flgr2d_anti_raster_slide_window_unfolded_fgUINT32(FLGR_Data2D *imgdest,FLGR_Data2D *imgsrc,FLGR_Data2D *nhb,int nhb_sym,
							 FLGR_ComputeNhb2D getNhbVal);

  EXPORT_LIB void flgr2d_anti_raster_slide_window_unfolded_fgINT8(FLGR_Data2D *imgdest,FLGR_Data2D *imgsrc,FLGR_Data2D *nhb,int nhb_sym,
						       FLGR_ComputeNhb2D getNhbVal);

  EXPORT_LIB void flgr2d_anti_raster_slide_window_unfolded_fgINT16(FLGR_Data2D *imgdest,FLGR_Data2D *imgsrc,FLGR_Data2D *nhb,int nhb_sym,
							FLGR_ComputeNhb2D getNhbVal);

  EXPORT_LIB void flgr2d_anti_raster_slide_window_unfolded_fgINT32(FLGR_Data2D *imgdest,FLGR_Data2D *imgsrc,FLGR_Data2D *nhb,int nhb_sym,
							FLGR_ComputeNhb2D getNhbVal);

  EXPORT_LIB void flgr2d_anti_raster_slide_window_unfolded_fgFLOAT32(FLGR_Data2D *imgdest,FLGR_Data2D *imgsrc,FLGR_Data2D *nhb,int nhb_sym,
							  FLGR_ComputeNhb2D getNhbVal);

  EXPORT_LIB void flgr2d_anti_raster_slide_window_unfolded_fgFLOAT64(FLGR_Data2D *imgdest,FLGR_Data2D *imgsrc,FLGR_Data2D *nhb,int nhb_sym,
							  FLGR_ComputeNhb2D getNhbVal);









  EXPORT_LIB void flgr2d_raster_slide_window_before_op_fgBIT(FLGR_Data2D *imgdest, FLGR_Data2D *imgsrc, FLGR_Data2D *imgarith, 
							   FLGR_Data2D *nhb,int nhb_sym,
							   FLGR_ComputeNhb2D computeNhb,
							   FLGR_ComputeArith computeArith);

  EXPORT_LIB void flgr2d_raster_slide_window_before_op_fgUINT8(FLGR_Data2D *imgdest, FLGR_Data2D *imgsrc, FLGR_Data2D *imgarith, 
							     FLGR_Data2D *nhb,int nhb_sym,
							     FLGR_ComputeNhb2D computeNhb,
							     FLGR_ComputeArith computeArith);

  EXPORT_LIB void flgr2d_raster_slide_window_before_op_fgUINT16(FLGR_Data2D *imgdest, FLGR_Data2D *imgsrc, FLGR_Data2D *imgarith,
							      FLGR_Data2D *nhb,int nhb_sym,
							      FLGR_ComputeNhb2D computeNhb,
							      FLGR_ComputeArith computeArith);

  EXPORT_LIB void flgr2d_raster_slide_window_before_op_fgUINT32(FLGR_Data2D *imgdest, FLGR_Data2D *imgsrc, FLGR_Data2D *imgarith,
							      FLGR_Data2D *nhb,int nhb_sym,
							      FLGR_ComputeNhb2D computeNhb,
							      FLGR_ComputeArith computeArith);

  EXPORT_LIB void flgr2d_raster_slide_window_before_op_fgINT8(FLGR_Data2D *imgdest, FLGR_Data2D *imgsrc, FLGR_Data2D *imgarith,
							    FLGR_Data2D *nhb,int nhb_sym,
							    FLGR_ComputeNhb2D computeNhb,
							    FLGR_ComputeArith computeArith);

  EXPORT_LIB void flgr2d_raster_slide_window_before_op_fgINT16(FLGR_Data2D *imgdest, FLGR_Data2D *imgsrc, FLGR_Data2D *imgarith,
							     FLGR_Data2D *nhb,int nhb_sym,
							     FLGR_ComputeNhb2D computeNhb,
							     FLGR_ComputeArith computeArith);

  EXPORT_LIB void flgr2d_raster_slide_window_before_op_fgINT32(FLGR_Data2D *imgdest, FLGR_Data2D *imgsrc, FLGR_Data2D *imgarith,
							     FLGR_Data2D *nhb,int nhb_sym,
							     FLGR_ComputeNhb2D computeNhb,
							     FLGR_ComputeArith computeArith);

  EXPORT_LIB void flgr2d_raster_slide_window_before_op_fgFLOAT32(FLGR_Data2D *imgdest, FLGR_Data2D *imgsrc, FLGR_Data2D *imgarith,
							       FLGR_Data2D *nhb,int nhb_sym,
							       FLGR_ComputeNhb2D computeNhb,
							       FLGR_ComputeArith computeArith);

  EXPORT_LIB void flgr2d_raster_slide_window_before_op_fgFLOAT64(FLGR_Data2D *imgdest, FLGR_Data2D *imgsrc, FLGR_Data2D *imgarith, 
							       FLGR_Data2D *nhb,int nhb_sym,
							       FLGR_ComputeNhb2D computeNhb,
							       FLGR_ComputeArith computeArith);



  EXPORT_LIB void flgr2d_anti_raster_slide_window_before_op_fgBIT(FLGR_Data2D *imgdest, FLGR_Data2D *imgsrc, FLGR_Data2D *imgarith,
								FLGR_Data2D *nhb,int nhb_sym,
								FLGR_ComputeNhb2D computeNhb,
								FLGR_ComputeArith computeArith);

  EXPORT_LIB void flgr2d_anti_raster_slide_window_before_op_fgUINT8(FLGR_Data2D *imgdest, FLGR_Data2D *imgsrc, FLGR_Data2D *imgarith,
								  FLGR_Data2D *nhb,int nhb_sym,
								  FLGR_ComputeNhb2D computeNhb,
								  FLGR_ComputeArith computeArith);

  EXPORT_LIB void flgr2d_anti_raster_slide_window_before_op_fgUINT16(FLGR_Data2D *imgdest, FLGR_Data2D *imgsrc, FLGR_Data2D *imgarith,
								   FLGR_Data2D *nhb,int nhb_sym,
								   FLGR_ComputeNhb2D computeNhb,
								   FLGR_ComputeArith computeArith);

  EXPORT_LIB void flgr2d_anti_raster_slide_window_before_op_fgUINT32(FLGR_Data2D *imgdest, FLGR_Data2D *imgsrc, FLGR_Data2D *imgarith,
								   FLGR_Data2D *nhb,int nhb_sym,
								   FLGR_ComputeNhb2D computeNhb,
								   FLGR_ComputeArith computeArith);

  EXPORT_LIB void flgr2d_anti_raster_slide_window_before_op_fgINT8(FLGR_Data2D *imgdest, FLGR_Data2D *imgsrc, FLGR_Data2D *imgarith,
								 FLGR_Data2D *nhb,int nhb_sym,
								 FLGR_ComputeNhb2D computeNhb,
								 FLGR_ComputeArith computeArith);

  EXPORT_LIB void flgr2d_anti_raster_slide_window_before_op_fgINT16(FLGR_Data2D *imgdest, FLGR_Data2D *imgsrc, FLGR_Data2D *imgarith,
								  FLGR_Data2D *nhb,int nhb_sym,
								  FLGR_ComputeNhb2D computeNhb,
								  FLGR_ComputeArith computeArith);

  EXPORT_LIB void flgr2d_anti_raster_slide_window_before_op_fgINT32(FLGR_Data2D *imgdest, FLGR_Data2D *imgsrc, FLGR_Data2D *imgarith,
								  FLGR_Data2D *nhb,int nhb_sym,
								  FLGR_ComputeNhb2D computeNhb,
								  FLGR_ComputeArith computeArith);

  EXPORT_LIB void flgr2d_anti_raster_slide_window_before_op_fgFLOAT32(FLGR_Data2D *imgdest, FLGR_Data2D *imgsrc, FLGR_Data2D *imgarith,
								    FLGR_Data2D *nhb,int nhb_sym,
								    FLGR_ComputeNhb2D computeNhb,
								    FLGR_ComputeArith computeArith);

  EXPORT_LIB void flgr2d_anti_raster_slide_window_before_op_fgFLOAT64(FLGR_Data2D *imgdest, FLGR_Data2D *imgsrc, FLGR_Data2D *imgarith,
								    FLGR_Data2D *nhb,int nhb_sym,
								    FLGR_ComputeNhb2D computeNhb,
								    FLGR_ComputeArith computeArith);



#endif

#ifdef __cplusplus
}
#endif

