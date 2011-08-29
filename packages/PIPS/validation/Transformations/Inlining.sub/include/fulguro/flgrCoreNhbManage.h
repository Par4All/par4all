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


#ifndef __FLGR_CORE_NHB_MANAGE_H
#define __FLGR_CORE_NHB_MANAGE_H

#include <flgrCoreDll.h>
#include <flgrCoreErrors.h>
#include <flgrCoreTypes.h>
#include <flgrCoreData.h>
#include <flgrCoreVector.h>

  /*!
   * \addtogroup group_fulguro_core_nhb
   * @{
   */
  
  /*! 
   * Neighborhood Box 1D
   */
  typedef struct {
    /*! type of the data in the extracted neighborhood */
    FLGR_Type type;
    /*! Sample per Pixel */
    int spp;
    /*! column Coordinate in the image of the pixel considered as the center of the neighborhood */
    int center_coord_x;
    /*! value of the center of the extracted signal window */
    FLGR_Vector *center_data_val;
    /*! value of the center of nhb */
    FLGR_Vector *center_nhb_val;
    /*! X size of the nhb used to extract values */
    int nhb_size_x;
    /*! X coordinates list of pixel extracted */
    int **list_coord_x;
    /*! list of pixels values extracted in the signal */
    void **list_data_val;
    /*! list of Neighborhood values corresponding to extracted pixels in the signal */
    void **list_nhb_val;
    /*! size of the extracted values list */
    int *size;
  }FLGR_NhbBox1D;

  /*! 
   * Neighborhood Box 2D
   */
  typedef struct {
    /*! type of the data in the extracted neighborhood */
    FLGR_Type type;
    /*! Sample per Pixel */
    int spp;
    /*! Row Coordinate in the image of the pixel considered as the center of the neighborhood */
    int center_coord_y;
    /*! Column Coordinate in the image of the pixel considered as the center of the neighborhood */
    int center_coord_x;
    /*! value of the center of the extracted image box */
    FLGR_Vector *center_data_val;
    /*! value of the center of nhb */
    FLGR_Vector *center_nhb_val;
    /*! Y size of the nhb used to extract values */
    int nhb_size_y;
    /*! X size of the nhb used to extract values */
    int nhb_size_x;
    /*! Y coordinates list of pixel extracted */
    int **list_coord_y;
    /*! X coordinates list of pixel extracted */
    int **list_coord_x;
    /*! list of pixels values extracted in the image */
    void **list_data_val;
    /*! list of Neighborhood values corresponding to extracted pixels in the image */
    void **list_nhb_val;
    /*! size of the pixel list */
    int *size;
  }FLGR_NhbBox2D;

  //! @}

  EXPORT_LIB FLGR_MorphoOptim flgr1d_get_optimization_available(FLGR_Data1D *datdest, FLGR_Data1D *datsrc, FLGR_Data1D *nhb);

  EXPORT_LIB FLGR_NhbBox1D *flgr1d_create_neighbor_box(FLGR_Data1D *data);

  EXPORT_LIB void flgr1d_destroy_neighbor_box(FLGR_NhbBox1D *extr);

  EXPORT_LIB FLGR_Ret flgr1d_get_neighborhood(FLGR_NhbBox1D *extr,
					      FLGR_Data1D *dat, FLGR_Data1D *nhb, int x);


  EXPORT_LIB void flgr1d_get_neighborhood_fgBIT(FLGR_NhbBox1D *extr,
						FLGR_Data1D *img, FLGR_Data1D *nhb, int pos);

  EXPORT_LIB void flgr1d_get_neighborhood_fgUINT8(FLGR_NhbBox1D *extr,
						  FLGR_Data1D *img, FLGR_Data1D *nhb, int pos);

  EXPORT_LIB void flgr1d_get_neighborhood_fgUINT16(FLGR_NhbBox1D *extr,
						   FLGR_Data1D *img, FLGR_Data1D *nhb, int pos);

  EXPORT_LIB void flgr1d_get_neighborhood_fgUINT32(FLGR_NhbBox1D *extr,
						   FLGR_Data1D *img, FLGR_Data1D *nhb, int pos);

  EXPORT_LIB void flgr1d_get_neighborhood_fgINT8(FLGR_NhbBox1D *extr,
						 FLGR_Data1D *img, FLGR_Data1D *nhb, int pos);

  EXPORT_LIB void flgr1d_get_neighborhood_fgINT16(FLGR_NhbBox1D *extr,
						  FLGR_Data1D *img, FLGR_Data1D *nhb, int pos);

  EXPORT_LIB void flgr1d_get_neighborhood_fgINT32(FLGR_NhbBox1D *extr,
						  FLGR_Data1D *img, FLGR_Data1D *nhb, int pos);

  EXPORT_LIB void flgr1d_get_neighborhood_fgFLOAT32(FLGR_NhbBox1D *extr,
						    FLGR_Data1D *img, FLGR_Data1D *nhb, int pos);

  EXPORT_LIB void flgr1d_get_neighborhood_fgFLOAT64(FLGR_NhbBox1D *extr,
						    FLGR_Data1D *img, FLGR_Data1D *nhb, int pos);





  EXPORT_LIB FLGR_Ret flgr1d_get_neighborhood_unfolded(FLGR_NhbBox1D *extr,
						       FLGR_Data1D *dat, FLGR_Data1D *nhb, int x);


  EXPORT_LIB void flgr1d_get_neighborhood_unfolded_fgBIT(FLGR_NhbBox1D *extr,
							 FLGR_Data1D *img, FLGR_Data1D *nhb, int pos);

  EXPORT_LIB void flgr1d_get_neighborhood_unfolded_fgUINT8(FLGR_NhbBox1D *extr,
							   FLGR_Data1D *img, FLGR_Data1D *nhb, int pos);

  EXPORT_LIB void flgr1d_get_neighborhood_unfolded_fgUINT16(FLGR_NhbBox1D *extr,
							    FLGR_Data1D *img, FLGR_Data1D *nhb, int pos);

  EXPORT_LIB void flgr1d_get_neighborhood_unfolded_fgUINT32(FLGR_NhbBox1D *extr,
							    FLGR_Data1D *img, FLGR_Data1D *nhb, int pos);

  EXPORT_LIB void flgr1d_get_neighborhood_unfolded_fgINT8(FLGR_NhbBox1D *extr,
							  FLGR_Data1D *img, FLGR_Data1D *nhb, int pos);

  EXPORT_LIB void flgr1d_get_neighborhood_unfolded_fgINT16(FLGR_NhbBox1D *extr,
							   FLGR_Data1D *img, FLGR_Data1D *nhb, int pos);

  EXPORT_LIB void flgr1d_get_neighborhood_unfolded_fgINT32(FLGR_NhbBox1D *extr,
							   FLGR_Data1D *img, FLGR_Data1D *nhb, int pos);

  EXPORT_LIB void flgr1d_get_neighborhood_unfolded_fgFLOAT32(FLGR_NhbBox1D *extr,
							     FLGR_Data1D *img, FLGR_Data1D *nhb, int pos);

  EXPORT_LIB void flgr1d_get_neighborhood_unfolded_fgFLOAT64(FLGR_NhbBox1D *extr,
							     FLGR_Data1D *img, FLGR_Data1D *nhb, int pos);












  EXPORT_LIB FLGR_MorphoOptim flgr2d_get_optimization_available(FLGR_Data2D *datdest, FLGR_Data2D *datsrc, FLGR_Data2D *nhb);

  EXPORT_LIB FLGR_NhbBox2D *flgr2d_create_neighbor_box(FLGR_Data2D *data);

  EXPORT_LIB void flgr2d_destroy_neighbor_box(FLGR_NhbBox2D *extr);

  EXPORT_LIB FLGR_Ret flgr2d_get_neighborhood(FLGR_NhbBox2D *extr,
					  FLGR_Data2D *img, FLGR_Data2D *nhb, int x, int y);




  EXPORT_LIB void flgr2d_get_neighborhood_fgBIT(FLGR_NhbBox2D *extr,
					    FLGR_Data2D *img, FLGR_Data2D *nhb, int x, int y);

  EXPORT_LIB void flgr2d_get_neighborhood_fgUINT8(FLGR_NhbBox2D *extr,
					      FLGR_Data2D *img, FLGR_Data2D *nhb, int x, int y);

  EXPORT_LIB void flgr2d_get_neighborhood_fgUINT16(FLGR_NhbBox2D *extr,
					       FLGR_Data2D *img, FLGR_Data2D *nhb, int x, int y);

  EXPORT_LIB void flgr2d_get_neighborhood_fgUINT32(FLGR_NhbBox2D *extr,
					       FLGR_Data2D *img, FLGR_Data2D *nhb, int x, int y);

  EXPORT_LIB void flgr2d_get_neighborhood_fgINT8(FLGR_NhbBox2D *extr,
					     FLGR_Data2D *img, FLGR_Data2D *nhb, int x, int y);

  EXPORT_LIB void flgr2d_get_neighborhood_fgINT16(FLGR_NhbBox2D *extr,
					      FLGR_Data2D *img, FLGR_Data2D *nhb, int x, int y);

  EXPORT_LIB void flgr2d_get_neighborhood_fgINT32(FLGR_NhbBox2D *extr,
					      FLGR_Data2D *img, FLGR_Data2D *nhb, int x, int y);

  EXPORT_LIB void flgr2d_get_neighborhood_fgFLOAT32(FLGR_NhbBox2D *extr,
						FLGR_Data2D *img, FLGR_Data2D *nhb, int x, int y);

  EXPORT_LIB void flgr2d_get_neighborhood_fgFLOAT64(FLGR_NhbBox2D *extr,
						FLGR_Data2D *img, FLGR_Data2D *nhb, int x, int y);


  EXPORT_LIB FLGR_Ret flgr2d_get_neighborhood_unfolded(FLGR_NhbBox2D *extr,
					  FLGR_Data2D *img, FLGR_Data2D *nhb, int x, int y);




  EXPORT_LIB void flgr2d_get_neighborhood_unfolded_fgBIT(FLGR_NhbBox2D *extr,
					    FLGR_Data2D *img, FLGR_Data2D *nhb, int x, int y);

  EXPORT_LIB void flgr2d_get_neighborhood_unfolded_fgUINT8(FLGR_NhbBox2D *extr,
					      FLGR_Data2D *img, FLGR_Data2D *nhb, int x, int y);

  EXPORT_LIB void flgr2d_get_neighborhood_unfolded_fgUINT16(FLGR_NhbBox2D *extr,
					       FLGR_Data2D *img, FLGR_Data2D *nhb, int x, int y);

  EXPORT_LIB void flgr2d_get_neighborhood_unfolded_fgUINT32(FLGR_NhbBox2D *extr,
					       FLGR_Data2D *img, FLGR_Data2D *nhb, int x, int y);

  EXPORT_LIB void flgr2d_get_neighborhood_unfolded_fgINT8(FLGR_NhbBox2D *extr,
					     FLGR_Data2D *img, FLGR_Data2D *nhb, int x, int y);

  EXPORT_LIB void flgr2d_get_neighborhood_unfolded_fgINT16(FLGR_NhbBox2D *extr,
					      FLGR_Data2D *img, FLGR_Data2D *nhb, int x, int y);

  EXPORT_LIB void flgr2d_get_neighborhood_unfolded_fgINT32(FLGR_NhbBox2D *extr,
					      FLGR_Data2D *img, FLGR_Data2D *nhb, int x, int y);

  EXPORT_LIB void flgr2d_get_neighborhood_unfolded_fgFLOAT32(FLGR_NhbBox2D *extr,
						FLGR_Data2D *img, FLGR_Data2D *nhb, int x, int y);

  EXPORT_LIB void flgr2d_get_neighborhood_unfolded_fgFLOAT64(FLGR_NhbBox2D *extr,
						FLGR_Data2D *img, FLGR_Data2D *nhb, int x, int y);




#endif

#ifdef __cplusplus
}
#endif
