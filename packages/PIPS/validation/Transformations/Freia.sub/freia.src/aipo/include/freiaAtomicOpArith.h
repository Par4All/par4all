/***************************************************************
 * Author : Christophe Clienti
 * Date : 10/22/2008
 * Copyright (c) 2008 Thales Research & Technologies (FR)
 ***************************************************************/

#ifdef __cplusplus
extern "C" {
#endif


#ifndef __FREIA_ATOMIC_OP_ARITH_H__
#define __FREIA_ATOMIC_OP_ARITH_H__

#include <freiaCommonTypes.h>

  /*!
   * \defgroup freia_aipo_arith Diadic and monoadic arithmetic operations
   * \ingroup freia_aipo
   * @{
   */

  /*!  
    \brief Minimum pixel by pixel of imin1 and imin2. Store the result in imout

    imout[p] = imin1[p] < imin2[p] ? imin1[p] : imin2[p];

    The operation could be done in-place 

    \param[out] imout destination image 
    \param[in] imin1 source image 1
    \param[in] imin2 source image 2
    \return error code
  */
  extern freia_status freia_aipo_inf(freia_data2d *imout, freia_data2d *imin1, freia_data2d *imin2);

  /*!  
    \brief Minimum pixel by pixel of imin and constant. Store the result in imout

    imout[p] = imin[p] < constant ? imin[p] : constant;

    The operation could be done in-place 

    \param[out] imout destination image 
    \param[in] imin source image
    \param[in] constant
    \return error code
  */
  extern freia_status freia_aipo_inf_const(freia_data2d *imout, freia_data2d *imin, int32_t constant);





  /*!  
    \brief Minimum pixel by pixel of imin1 and imin2. Store the result in imout

    imout[p] = imin1[p] > imin2[p] ? imin1[p] : imin2[p];

    The operation could be done in-place 

    \param[out] imout destination image 
    \param[in] imin1 source image 1
    \param[in] imin2 source image 2
    \return error code
  */
  extern freia_status freia_aipo_sup(freia_data2d *imout, freia_data2d *imin1, freia_data2d *imin2);

  /*!  
    \brief Minimum pixel by pixel of imin and constant. Store the result in imout

    imout[p] = imin[p] > constant ? imin[p] : constant;

    The operation could be done in-place 

    \param[out] imout destination image 
    \param[in] imin source image
    \param[in] constant
    \return error code
  */
  extern freia_status freia_aipo_sup_const(freia_data2d *imout, freia_data2d *imin, int32_t constant);






  /*!  
    \brief Subtract imin1 and imin2. Store the result in imout

    imout = imin1 - imin2

    The subtraction could be done in-place 

    \param[out] imout destination image 
    \param[in] imin1 source image 1
    \param[in] imin2 source image 2
    \return error code
  */
  extern freia_status freia_aipo_sub(freia_data2d *imout, freia_data2d *imin1, freia_data2d *imin2);

  /*!  
    \brief Subtract imin with a constant. Store the result in imout

    imout = imin - constant

    The subtraction could be done in-place 

    \param[out] imout destination image 
    \param[in] imin source image
    \param[in] constant
    \return error code
  */
  extern freia_status freia_aipo_sub_const(freia_data2d *imout, freia_data2d *imin, int32_t constant);






  /*!  
    \brief Subsattract imin1 and imin2. Store the result in imout

    imout = (imin1 - imin2) <= MIN ? MIN : (imin1 - imin2) 

    The subsattraction could be done in-place 

    \param[out] imout destination image 
    \param[in] imin1 source image 1
    \param[in] imin2 source image 2
    \return error code
  */
  extern freia_status freia_aipo_subsat(freia_data2d *imout, freia_data2d *imin1, freia_data2d *imin2);

  /*!  
    \brief Subsattract imin with a constant. Store the result in imout

    imout = (imin - constant) <= MIN ? MIN : (imin - constant)

    The subsattraction could be done in-place 

    \param[out] imout destination image 
    \param[in] imin source image
    \param[in] constant
    \return error code
  */
  extern freia_status freia_aipo_subsat_const(freia_data2d *imout, freia_data2d *imin, int32_t constant);





  /*!  
    \brief add imin1 and imin2. Store the result in imout

    imout = imin1 + imin2

    This operation could be done in-place 

    \param[out] imout destination image 
    \param[in] imin1 source image 1
    \param[in] imin2 source image 2
    \return error code
  */
  extern freia_status freia_aipo_add(freia_data2d *imout, freia_data2d *imin1, freia_data2d *imin2);

  /*!  
    \brief add imin and constant. Store the result in imout

    imout = imin + constant

    This operation could be done in-place 

    \param[out] imout destination image 
    \param[in] imin source image
    \param[in] constant
    \return error code
  */
  extern freia_status freia_aipo_add_const(freia_data2d *imout, freia_data2d *imin, int32_t constant);






  /*!  
    \brief addsat imin1 and imin2. Store the result in imout

    imout = (imin1 + imin2) >= MAX ? MAX : (imin1 + imin2)

    This operation could be done in-place 

    \param[out] imout destination image 
    \param[in] imin1 source image 1
    \param[in] imin2 source image 2
    \return error code
  */
  extern freia_status freia_aipo_addsat(freia_data2d *imout, freia_data2d *imin1, freia_data2d *imin2);

  /*!  
    \brief addsat imin and constant. Store the result in imout

    imout = (imin + constant) >= MAX ? MAX : (imin + constant)

    This operation could be done in-place 

    \param[out] imout destination image 
    \param[in] imin source image
    \param[in] constant
    \return error code
  */
  extern freia_status freia_aipo_addsat_const(freia_data2d *imout, freia_data2d *imin, int32_t constant);








  /*!  
    \brief Compute a absolute difference of imin1 and imin2. Store the result in imout

    imout = |imin1 - imin2|

    This operation could be done in-place 

    \param[out] imout destination image 
    \param[in] imin1 source image 1
    \param[in] imin2 source image 2
    \return error code
  */
  extern freia_status freia_aipo_absdiff(freia_data2d *imout, freia_data2d *imin1, freia_data2d *imin2);


  /*!  
    \brief Compute a absolute difference of imin and a constant. Store the result in imout

    imout = |imin - constant|

    This operation could be done in-place 

    \param[out] imout destination image 
    \param[in] imin source image
    \param[in] constant
    \return error code
  */
  extern freia_status freia_aipo_absdiff_const(freia_data2d *imout, freia_data2d *imin, int32_t constant);


 



  /*!  
    \brief Multiply imin1 and imin2. Store the result in imout

    imout = imin1 * imin2

    This operation could be done in-place 

    \param[out] imout destination image 
    \param[in] imin1 source image 1
    \param[in] imin2 source image 2
    \return error code
  */
  extern freia_status freia_aipo_mul(freia_data2d *imout, freia_data2d *imin1, freia_data2d *imin2);


  /*!  
    \brief Multiply imin and a constant. Store the result in imout

    imout = imin * constant

    This operation could be done in-place 

    \param[out] imout destination image 
    \param[in] imin source image
    \param[in] constant
    \return error code
  */
  extern freia_status freia_aipo_mul_const(freia_data2d *imout, freia_data2d *imin, int32_t constant);



  /*!  
    \brief Divide imin1 and imin2. Store the result in imout

    imout = imin1 / imin2

    This operation could be done in-place 

    \param[out] imout destination image 
    \param[in] imin1 source image 1
    \param[in] imin2 source image 2
    \return error code
  */
  extern freia_status freia_aipo_div(freia_data2d *imout, freia_data2d *imin1, freia_data2d *imin2);


  /*!  
    \brief Divide imin and a constant. Store the result in imout

    imout = imin / constant

    This operation could be done in-place 

    \param[out] imout destination image 
    \param[in] imin source image
    \param[in] constant
    \return error code
  */
  extern freia_status freia_aipo_div_const(freia_data2d *imout, freia_data2d *imin, int32_t constant);







  /*!  
    \brief Logical and between imin1 and imin2. Store the result in imout

    imout = imin1 & imin2

    This operation could be done in-place 

    \param[out] imout destination image 
    \param[in] imin1 source image 1
    \param[in] imin2 source image 2
    \return error code
  */
  extern freia_status freia_aipo_and(freia_data2d *imout, freia_data2d *imin1, freia_data2d *imin2);


  /*!  
    \brief Logical and between imin1 and imin2. Store the result in imout

    imout = imin & constant

    This operation could be done in-place 

    \param[out] imout destination image 
    \param[in] imin source image
    \param[in] constant
    \return error code
  */
  extern freia_status freia_aipo_and_const(freia_data2d *imout, freia_data2d *imin, int32_t constant);







  /*!  
    \brief Logical or between imin1 and imin2. Store the result in imout

    imout = imin1 | imin2

    This operation could be done in-place 

    \param[out] imout destination image 
    \param[in] imin1 source image 1
    \param[in] imin2 source image 2
    \return error code
  */
  extern freia_status freia_aipo_or(freia_data2d *imout, freia_data2d *imin1, freia_data2d *imin2);


  /*!  
    \brief Logical or between imin1 and imin2. Store the result in imout

    imout = imin | constant

    This operation could be done in-place 

    \param[out] imout destination image 
    \param[in] imin source image
    \param[in] constant
    \return error code
  */
  extern freia_status freia_aipo_or_const(freia_data2d *imout, freia_data2d *imin, int32_t constant);






  /*!  
    \brief Logical xor between imin1 and imin2. Store the result in imout

    imout = imin1 ^ imin2

    This operation could be done in-place 

    \param[out] imout destination image 
    \param[in] imin1 source image 1
    \param[in] imin2 source image 2
    \return error code
  */
  extern freia_status freia_aipo_xor(freia_data2d *imout, freia_data2d *imin1, freia_data2d *imin2);


  /*!  
    \brief Logical xor between imin1 and imin2. Store the result in imout

    imout = imin ^ constant

    This operation could be done in-place 

    \param[out] imout destination image 
    \param[in] imin source image
    \param[in] constant
    \return error code
  */
  extern freia_status freia_aipo_xor_const(freia_data2d *imout, freia_data2d *imin, int32_t constant);






  /*!  
    \brief Logical not between imin1 and imin2. Store the result in imout

    imout = ~imin

    This operation could be done in-place 

    \param[out] imout destination image 
    \param[in] imin source image
    \return error code
  */
  extern freia_status freia_aipo_not(freia_data2d *imout, freia_data2d *imin);



  /*!  
    \brief log2 of input image pixels

    imout(p) = log2(imin(p))

    This operation could be done in-place 

    \param[out] imout destination image 
    \param[in] imin source image
    \return error code
  */
  extern freia_status freia_aipo_log2(freia_data2d *imout, freia_data2d *imin);



  /*!@}*/


#endif

#ifdef __cplusplus
}
#endif
