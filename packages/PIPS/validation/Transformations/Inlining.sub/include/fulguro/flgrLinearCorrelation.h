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

#ifndef __FLGR_LINEAR_CORRELATION_H
#define __FLGR_LINEAR_CORRELATION_H

#include <flgrCoreDll.h>
#include <flgrCoreErrors.h>
#include <flgrCoreTypes.h>
#include <flgrCoreData.h>
#include <flgrCoreNhbManage.h>



  EXPORT_LIB FLGR_Ret flgr2d_fast_correlation(FLGR_Data2D *imout, FLGR_Data2D *imin, FLGR_Data2D *imref, int horizon);

 
  EXPORT_LIB void flgr2d_fast_correlation_fgBIT(FLGR_Data2D *imout, FLGR_Data2D *imin, FLGR_Data2D *imref, int horizon);
  EXPORT_LIB void flgr2d_fast_correlation_fgUINT8(FLGR_Data2D *imout, FLGR_Data2D *imin, FLGR_Data2D *imref, int horizon);
  EXPORT_LIB void flgr2d_fast_correlation_fgUINT16(FLGR_Data2D *imout, FLGR_Data2D *imin, FLGR_Data2D *imref, int horizon);
  EXPORT_LIB void flgr2d_fast_correlation_fgUINT32(FLGR_Data2D *imout, FLGR_Data2D *imin, FLGR_Data2D *imref, int horizon);
  EXPORT_LIB void flgr2d_fast_correlation_fgINT8(FLGR_Data2D *imout, FLGR_Data2D *imin, FLGR_Data2D *imref, int horizon);
  EXPORT_LIB void flgr2d_fast_correlation_fgINT16(FLGR_Data2D *imout, FLGR_Data2D *imin, FLGR_Data2D *imref, int horizon);
  EXPORT_LIB void flgr2d_fast_correlation_fgINT32(FLGR_Data2D *imout, FLGR_Data2D *imin, FLGR_Data2D *imref, int horizon);
  EXPORT_LIB void flgr2d_fast_correlation_fgFLOAT32(FLGR_Data2D *imout, FLGR_Data2D *imin, FLGR_Data2D *imref, int horizon);
  EXPORT_LIB void flgr2d_fast_correlation_fgFLOAT64(FLGR_Data2D *imout, FLGR_Data2D *imin, FLGR_Data2D *imref, int horizon);

 
#endif 

#ifdef __cplusplus
}
#endif
