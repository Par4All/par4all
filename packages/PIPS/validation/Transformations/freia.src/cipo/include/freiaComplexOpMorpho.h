/***************************************************************
 * Author : Christophe Clienti
 * Date : 10/22/2008
 * Copyright (c) 2008 Thales Research & Technologies (FR)
 ***************************************************************/

#ifdef __cplusplus
extern "C" {
#endif


#ifndef __FREIA_COMPLEX_OP_MORPHO_H__
#define __FREIA_COMPLEX_OP_MORPHO_H__

#include <freiaCommonTypes.h>

extern const int32_t freia_morpho_kernel_8c[9];
extern const int32_t freia_morpho_kernel_6c[9];
extern const int32_t freia_morpho_kernel_4c[9];


  /*!
   * \defgroup freia_cipo_morpho  Mathematical morphology complex operations
   * \ingroup freia_cipo
   * @{
   */

  /*!  
    \brief Erode and image using a specific connexity and a specific size

     The Structuring element used depends of given connexity :
    - 8-connexity : 3x3 square
    - 6-connexity : hexagon of radius 1
    - 4-connexity : rhombus (inscribed in a square 3x3)

    Size represents the radius of structuring element. In practice,
    the size parameters is used to repeat 'size' time the erosion
    operation with a unitary structuring element.

    The operation could be done in-place 

    \param[out] imout destination image 
    \param[in] imin source image
    \param[in] connexity 4,6 or 8 connexity
    \param[in] size size the structuring element
    \return error code
  */
  extern freia_status freia_cipo_erode(freia_data2d *imout, freia_data2d *imin, int32_t connexity, uint32_t size);

  /*!  
    \brief Dilate and image using a specific connexity and a specific size

    The Structuring element used depends of given connexity :
    - 8-connexity : 3x3 square
    - 6-connexity : hexagon of radius 1
    - 4-connexity : rhombus (inscribed in a square 3x3)

    Size represents the radius of structuring element. In practice,
    the size parameters is used to repeat 'size' time the erosion
    operation with a unitary structuring element.

    The operation could be done in-place 

    \param[out] imout destination image 
    \param[in] imin source image
    \param[in] connexity 4,6 or 8 connexity
    \param[in] size size the structuring element
    \return error code
  */
  extern freia_status freia_cipo_dilate(freia_data2d *imout, freia_data2d *imin, int32_t connexity, uint32_t size);

  /*!  
    \brief Compute inner gradient using a specific connexity and a specific size

    The Structuring element used depends of given connexity :
    - 8-connexity : 3x3 square
    - 6-connexity : hexagon of radius 1
    - 4-connexity : rhombus (inscribed in a square 3x3)

    Size represents the radius of structuring element. In practice,
    the size parameters is used to repeat 'size' time the erosion
    operation with a unitary structuring element.

    The operation could NOT be done in-place 

    \param[out] imout destination image 
    \param[in] imin source image
    \param[in] connexity 4,6 or 8 connexity
    \param[in] size size the structuring element
    \return error code
  */
  extern freia_status freia_cipo_inner_gradient(freia_data2d *imout, freia_data2d *imin, int32_t connexity, uint32_t size);

  /*!  
    \brief Compute outer gradient using a specific connexity and a specific size

    The Structuring element used depends of given connexity :
    - 8-connexity : 3x3 square
    - 6-connexity : hexagon of radius 1
    - 4-connexity : rhombus (inscribed in a square 3x3)

    Size represents the radius of structuring element. In practice,
    the size parameters is used to repeat 'size' time the erosion
    operation with a unitary structuring element.

    The operation could NOT be done in-place 

    \param[out] imout destination image 
    \param[in] imin source image
    \param[in] connexity 4,6 or 8 connexity
    \param[in] size size the structuring element
    \return error code
  */
  extern freia_status freia_cipo_outer_gradient(freia_data2d *imout, freia_data2d *imin, int32_t connexity, uint32_t size);

  /*!  
    \brief Compute thick gradient using a specific connexity and a specific size

    The Structuring element used depends of given connexity :
    - 8-connexity : 3x3 square
    - 6-connexity : hexagon of radius 1
    - 4-connexity : rhombus (inscribed in a square 3x3)

    Size represents the radius of structuring element. In practice,
    the size parameters is used to repeat 'size' time the erosion
    operation with a unitary structuring element.

    The operation could be done in-place 

    \param[out] imout destination image 
    \param[in] imin source image
    \param[in] connexity 4,6 or 8 connexity
    \param[in] size size the structuring element
    \return error code
  */
  extern freia_status freia_cipo_gradient(freia_data2d *imout, freia_data2d *imin, int32_t connexity, uint32_t size);

  /*!  
    \brief Open an image using a specific connexity and a specific size

    The Structuring element used depends of given connexity :
    - 8-connexity : 3x3 square
    - 6-connexity : hexagon of radius 1
    - 4-connexity : rhombus (inscribed in a square 3x3)

    Size represents the radius of structuring element. In practice,
    the size parameters is used to repeat 'size' time the erosion
    operation with a unitary structuring element.

    The operation could be done in-place 

    \param[out] imout destination image 
    \param[in] imin source image
    \param[in] connexity 4,6 or 8 connexity
    \param[in] size size the structuring element
    \return error code
  */
  extern freia_status freia_cipo_open(freia_data2d *imout, freia_data2d *imin, int32_t connexity, uint32_t size);

  /*!  
    \brief Close an image using a specific connexity and a specific size

    The Structuring element used depends of given connexity :
    - 8-connexity : 3x3 square
    - 6-connexity : hexagon of radius 1
    - 4-connexity : rhombus (inscribed in a square 3x3)

    Size represents the radius of structuring element. In practice,
    the size parameters is used to repeat 'size' time the erosion
    operation with a unitary structuring element.

    The operation could be done in-place 

    \param[out] imout destination image 
    \param[in] imin source image
    \param[in] connexity 4,6 or 8 connexity
    \param[in] size size the structuring element
    \return error code
  */
  extern freia_status freia_cipo_close(freia_data2d *imout, freia_data2d *imin, int32_t connexity, uint32_t size);

  /*!  
    \brief Compute the Open Tophat transformation using a specific connexity and a specific size

    The Structuring element used depends of given connexity :
    - 8-connexity : 3x3 square
    - 6-connexity : hexagon of radius 1
    - 4-connexity : rhombus (inscribed in a square 3x3)

    Size represents the radius of structuring element. In practice,
    the size parameters is used to repeat 'size' time the erosion
    operation with a unitary structuring element.

    The operation could NOT be done in-place 

    \param[out] imout destination image 
    \param[in] imin source image
    \param[in] connexity 4,6 or 8 connexity
    \param[in] size size the structuring element
    \return error code
  */
  extern freia_status freia_cipo_open_tophat(freia_data2d *imout, freia_data2d *imin, int32_t connexity, uint32_t size);

  /*!  
    \brief Compute the Close Tophat transformation using a specific connexity and a specific size

    The Structuring element used depends of given connexity :
    - 8-connexity : 3x3 square
    - 6-connexity : hexagon of radius 1
    - 4-connexity : rhombus (inscribed in a square 3x3)

    Size represents the radius of structuring element. In practice,
    the size parameters is used to repeat 'size' time the erosion
    operation with a unitary structuring element.

    The operation could NOT be done in-place 

    \param[out] imout destination image 
    \param[in] imin source image
    \param[in] connexity 4,6 or 8 connexity
    \param[in] size size the structuring element
    \return error code
  */
  extern freia_status freia_cipo_close_tophat(freia_data2d *imout, freia_data2d *imin, int32_t connexity, uint32_t size);

  /*!  
    \brief Compute the a geodesic dilation using a specific connexity and a specific size

    The Structuring element used depends of given connexity :
    - 8-connexity : 3x3 square
    - 6-connexity : hexagon of radius 1
    - 4-connexity : rhombus (inscribed in a square 3x3)

    Size represents the radius of structuring element. In practice,
    the size parameters is used to repeat 'size' time the erosion
    operation with a unitary structuring element.

    The operation could be done in-place 

    \param[out] imout destination image 
    \param[in] immarker marker image
    \param[in] immask mask image
    \param[in] connexity 4,6 or 8 connexity
    \param[in] size size the structuring element
    \return error code
  */
  extern freia_status freia_cipo_geodesic_dilate(freia_data2d *imout, freia_data2d *immarker, freia_data2d *immask, int32_t connexity, uint32_t size);

  /*!  
    \brief Compute the a geodesic erosion using a specific connexity and a specific size

    The Structuring element used depends of given connexity :
    - 8-connexity : 3x3 square
    - 6-connexity : hexagon of radius 1
    - 4-connexity : rhombus (inscribed in a square 3x3)

    Size represents the radius of structuring element. In practice,
    the size parameters is used to repeat 'size' time the erosion
    operation with a unitary structuring element.

    The operation could be done in-place 

    \param[out] imout destination image 
    \param[in] immarker marker image
    \param[in] immask mask image
    \param[in] connexity 4,6 or 8 connexity
    \param[in] size size the structuring element
    \return error code
  */
  extern freia_status freia_cipo_geodesic_erode(freia_data2d *imout, freia_data2d *immarker, freia_data2d *immask, int32_t connexity, uint32_t size);

  /*!  
    \brief Compute a levelling using a specific connexity
    
    The Structuring element used depends of given connexity :
    - 8-connexity : 3x3 square
    - 6-connexity : hexagon of radius 1
    - 4-connexity : rhombus (inscribed in a square 3x3)

    \param[in,out] immarker: marker image
    \param[in] immask mask image
    \param[in] connexity 4,6 or 8 connexity
    \return error code
  */
  extern freia_status freia_cipo_geodesic_reconstruct_dual(freia_data2d *immarker, freia_data2d *immask, int32_t connexity);
  
  /*!  
    \brief Compute a geodesic reconstruction by erosion using a specific connexity

    The Structuring element used depends of given connexity :
    - 8-connexity : 3x3 square
    - 6-connexity : hexagon of radius 1
    - 4-connexity : rhombus (inscribed in a square 3x3)

    \param[in,out] immarker: marker image
    \param[in] immask mask image
    \param[in] connexity 4,6 or 8 connexity
    \return error code
  */
  extern freia_status freia_cipo_geodesic_reconstruct_erode(freia_data2d *immarker, freia_data2d *immask, int32_t connexity);

  /*!  
    \brief Compute a geodesic reconstruction by dilation using a specific connexity

     The Structuring element used depends of given connexity :
    - 8-connexity : 3x3 square
    - 6-connexity : hexagon of radius 1
    - 4-connexity : rhombus (inscribed in a square 3x3)

    \param[in,out] immarker: marker image
    \param[in] immask mask image
    \param[in] connexity 4,6 or 8 connexity
    \return error code
  */
  extern freia_status freia_cipo_geodesic_reconstruct_dilate(freia_data2d *immarker, freia_data2d *immask, int32_t connexity);

  /*!  
    \brief Compute a geodesic openning using a specific connexity and a specific size 

    The Structuring element used depends of given connexity :
    - 8-connexity : 3x3 square
    - 6-connexity : hexagon of radius 1
    - 4-connexity : rhombus (inscribed in a square 3x3)

    Size represents the radius of structuring element. In practice,
    the size parameters is used to repeat 'size' time the erosion
    operation with a unitary structuring element.

    The operation could NOT be done in-place 

    \param[out] imout destination image 
    \param[in] imin source image
    \param[in] connexity 4,6 or 8 connexity
    \param[in] size size the structuring element
    \return error code
  */
  extern freia_status freia_cipo_geodesic_reconstruct_open(freia_data2d *imout, freia_data2d *imin, int32_t connexity, uint32_t size);

  /*!  
    \brief Compute a geodesic closing using a specific connexity and a specific size 

    The Structuring element used depends of given connexity :
    - 8-connexity : 3x3 square
    - 6-connexity : hexagon of radius 1
    - 4-connexity : rhombus (inscribed in a square 3x3)

    Size represents the radius of structuring element. In practice,
    the size parameters is used to repeat 'size' time the erosion
    operation with a unitary structuring element.

    The operation could NOT be done in-place 

    \param[out] imout destination image 
    \param[in] imin source image
    \param[in] connexity 4,6 or 8 connexity
    \param[in] size size the structuring element
    \return error code
  */
  extern freia_status freia_cipo_geodesic_reconstruct_close(freia_data2d *imout, freia_data2d *imin, int32_t connexity, uint32_t size);
  
  /*!  
    \brief Compute a geodesic open tophat using a specific connexity and a specific size 

    The Structuring element used depends of given connexity :
    - 8-connexity : 3x3 square
    - 6-connexity : hexagon of radius 1
    - 4-connexity : rhombus (inscribed in a square 3x3)

    Size represents the radius of structuring element. In practice,
    the size parameters is used to repeat 'size' time the erosion
    operation with a unitary structuring element.

    The operation could NOT be done in-place 

    \param[out] imout destination image 
    \param[in] imin source image
    \param[in] connexity 4,6 or 8 connexity
    \param[in] size size the structuring element
    \return error code
  */
  extern freia_status freia_cipo_geodesic_reconstruct_open_tophat(freia_data2d *imout, freia_data2d *imin, int32_t connexity, uint32_t size);

  /*!  
    \brief Compute a geodesic close tophat using a specific connexity and a specific size 

    The Structuring element used depends of given connexity :
    - 8-connexity : 3x3 square
    - 6-connexity : hexagon of radius 1
    - 4-connexity : rhombus (inscribed in a square 3x3)

    Size represents the radius of structuring element. In practice,
    the size parameters is used to repeat 'size' time the erosion
    operation with a unitary structuring element.

    The operation could NOT be done in-place 

    \param[out] imout destination image 
    \param[in] imin source image
    \param[in] connexity 4,6 or 8 connexity
    \param[in] size size the structuring element
    \return error code
  */
  extern freia_status freia_cipo_geodesic_reconstruct_close_tophat(freia_data2d *imout, freia_data2d *imin, int32_t connexity, uint32_t size);
  
  /*!  
    \brief Compute a regional h-minima using a specific connexity

    The Structuring element used depends of given connexity :
    - 8-connexity : 3x3 square
    - 6-connexity : hexagon of radius 1
    - 4-connexity : rhombus (inscribed in a square 3x3)

    The operation could NOT be done in-place 

    \param[out] imout destination image 
    \param[in] imin source image
    \param[in] connexity 4,6 or 8 connexity
    \param[in] hlevel level to consider
    \return error code
  */
  extern freia_status freia_cipo_regional_hminima(freia_data2d *imout, freia_data2d *imin, int32_t hlevel, int32_t connexity);

  /*!  
    \brief Compute a regional h-maxima using a specific connexity

    The Structuring element used depends of given connexity :
    - 8-connexity : 3x3 square
    - 6-connexity : hexagon of radius 1
    - 4-connexity : rhombus (inscribed in a square 3x3)
    
    The operation could NOT be done in-place 

    \param[out] imout destination image 
    \param[in] imin source image
    \param[in] connexity 4,6 or 8 connexity
    \param[in] hlevel level to consider
    \return error code
  */
  extern freia_status freia_cipo_regional_hmaxima(freia_data2d *imout, freia_data2d *imin, int32_t hlevel, int32_t connexity);




  /*!@}*/


#endif

#ifdef __cplusplus
}
#endif
