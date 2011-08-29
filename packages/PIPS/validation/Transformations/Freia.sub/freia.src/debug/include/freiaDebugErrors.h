/***************************************************************
 * Author : Christophe Clienti
 * Date : 10/22/2008
 * Copyright (c) 2008 Thales Research & Technologies (FR)
 ***************************************************************/

#ifdef __cplusplus
extern "C" {
#endif

#ifndef __FREIA_DEBUG_ERRORS_H__
#define __FREIA_DEBUG_ERRORS_H__

#include <stdio.h>

  /*!
   * \defgroup freia_debug_data Debuging functions
   * \ingroup freia_debug
   * @{
   */

  /*!
   * Print stack
   *
   * \sa freia_debug_print_backtrace
   *
   */
#ifdef DEBUG_BACKTRACE
#define FREIA_PRINT_BACKTRACE freia_debug_print_backtrace()
#else
#define FREIA_PRINT_BACKTRACE while(0)
#endif

  /*!
   * Print an errror to console and display backtrace
   *
   * \sa freia_debug_print_backtrace
   *
   */
#define FREIA_ERROR(chaine,...)    fprintf(stderr,"ERROR: file %s, line %d, function %s: "chaine, \
					   __FILE__,__LINE__,__FUNCTION__,##__VA_ARGS__);FREIA_PRINT_BACKTRACE
  /*! 
   * Print a warning to console and display backtrace
   *
   * \sa freia_debug_print_backtrace
   *
   */
#define FREIA_WARNING(chaine,...)  fprintf(stderr,"WARNING: file %s, line %d, function %s: "chaine, \
					   __FILE__,__LINE__,__FUNCTION__,##__VA_ARGS__);FREIA_PRINT_BACKTRACE
  /*!
   * Print an information to console
   */
#define FREIA_INFO(chaine,...)     fprintf(stdout,"INFO: %s: "chaine,__FUNCTION__,##__VA_ARGS__)



  /*! \brief Display the current call stack. This functionality is
   *  guaranted only with fulguro and simulation platforms
   *
   * \return void
   */
   extern void freia_debug_print_backtrace(void);


  /*!@}*/


#endif

#ifdef __cplusplus
}
#endif
