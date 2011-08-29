/***************************************************************
 * Author : Christophe Clienti
 * Date : 10/22/2008
 * Copyright (c) 2008 Thales Research & Technologies (FR)
 ***************************************************************/

#ifdef __cplusplus
extern "C" {
#endif


#ifndef __FREIA_COMMON_ALLOC_H__
#define __FREIA_COMMON_ALLOC_H__

#include <freiaCommonTypes.h>

 /*!
   * \defgroup freia_common_alloc Memory allocation functions
   * \ingroup freia_common
   * @{
   */

  /*!  
    \brief Allocate size bytes
    
    \param[in] size size in byte
    \return pointer to allocated memory area
  */

  extern freia_ptr freia_common_alloc(size_t size);
  
  /*!  
    \brief Unallocate memory area
    
    \param[in] ptr pointer to allocated memory area
   */
  extern void freia_common_free(freia_ptr ptr);
  

  /*!@}*/

#endif

#ifdef __cplusplus
}
#endif
