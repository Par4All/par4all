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


#ifndef __FLGR_CORE_ERRORS_H
#define __FLGR_CORE_ERRORS_H

#include <flgrCoreDll.h>
#include <stdio.h>

  /*!
   * \addtogroup group_fulguro_core
   * @{
   */

  EXPORT_LIB void flgr_backtrace_print(void);
  
  //! Print an errror in console
#define POST_ERROR(chaine,...)    fprintf(stderr,"ERROR: File %s, Line %d, Function %s: "chaine, \
					  __FILE__,__LINE__,__FUNCTION__,##__VA_ARGS__); flgr_backtrace_print()
  //! Print a warning in console
#define POST_WARNING(chaine,...)  fprintf(stderr,"WARNING: File %s, Line %d, Function %s: "chaine, \
					  __FILE__,__LINE__,__FUNCTION__,##__VA_ARGS__); flgr_backtrace_print()
  //! Print an information in console
#define POST_INFO(chaine,...)     fprintf(stderr,"INFO: %s: "chaine,__FUNCTION__,##__VA_ARGS__)
  
#ifdef DEBUG
  //! Print a debug information in console
#define POST_DEBUG(chaine,...)    fprintf(stderr,"DEBUG INFO: "chaine,##__VA_ARGS__)
#else
#define POST_DEBUG(chaine,...)
#endif
  
#define EPRINTF(...)              fprintf(stderr,__VA_ARGS__);flgr_backtrace_print()
#define WPRINTF(...)              fprintf(stderr,__VA_ARGS__);flgr_backtrace_print()
#define IPRINTF(...)              fprintf(stderr,__VA_ARGS__);


#define PRINT_DATA(data,dtype)						\
  {									\
    int i,j;								\
    IPRINTF("\n");							\
    for(i=0;i<data->size_y;i++) {					\
      for(j=0;j<data->size_x;j++) {					\
	IPRINTF(stderr,"%3ld",						\
		(long) flgr2d_get_data_array_##dtype((dtype**) data->array,i,j)); \
      }									\
      IPRINTF("\n");							\
    }									\
    IPRINTF("\n");							\
  }


  //!@}

#endif

#ifdef __cplusplus
}
#endif
