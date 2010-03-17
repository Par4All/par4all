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


#ifndef __FLGR_CORE_IO_H
#define __FLGR_CORE_IO_H

#include <flgrCoreTypes.h>

#define FLGR_GET_ARRAY(dtype,array,pos) flgr_get_array_##dtype((dtype*) array, pos)

  static __inline__ void flgr_set_array_fgBIT(fgBIT *array, int pos, fgBIT value) {
    fgBIT val;
    fgBIT mask=0;
    int vectorbps = sizeof(fgBIT)<<3;
    int vectorSelect;
    int pixinVector;
    int nbdec;

    vectorSelect = pos / vectorbps;
    pixinVector = pos % vectorbps;

    nbdec = vectorbps-((fgBIT) 1)-pixinVector;
    mask = ((fgBIT) 1) << (nbdec);

    val=array[vectorSelect];
    val &= (~mask);
    value = ((value)&((fgBIT) 1)) << nbdec;
    val |= value;

    array[vectorSelect]=val;
    
  }

  static __inline__ void flgr_set_array_fgUINT8(fgUINT8* array, int pos, fgUINT8 value) {
    array[pos]=value;
  }

  static __inline__ void flgr_set_array_fgUINT16(fgUINT16* array, int pos, fgUINT16 value) {
    array[pos]=value;
  }

  static __inline__ void flgr_set_array_fgUINT32(fgUINT32* array, int pos, fgUINT32 value) {
    array[pos]=value;
  }

  static __inline__ void flgr_set_array_fgUINT64(fgUINT64* array, int pos, fgUINT64 value) {
    array[pos]=value;
  }

  static __inline__ void flgr_set_array_fgINT8(fgINT8* array, int pos, fgINT8 value) {
    array[pos]=value;
  }

  static __inline__ void flgr_set_array_fgINT16(fgINT16* array, int pos, fgINT16 value) {
    array[pos]=value;
  }

  static __inline__ void flgr_set_array_fgINT32(fgINT32* array, int pos, fgINT32 value) {
    array[pos]=value;
  }

  static __inline__ void flgr_set_array_fgINT64(fgINT64* array, int pos, fgINT64 value) {
    array[pos]=value;
  }

   static __inline__ void flgr_set_array_fgFLOAT32(fgFLOAT32* array, int pos, fgFLOAT32 value) {
    array[pos]=value;
  }

  static __inline__ void flgr_set_array_fgFLOAT64(fgFLOAT64* array, int pos, fgFLOAT64 value) {
    array[pos]=value;
  }




  static __inline__ fgBIT flgr_get_array_fgBIT(fgBIT* array, int pos) {
    fgBIT val;
    fgBIT mask=0;
    int vectorbps = sizeof(fgBIT)<<3;
    int vectorSelect;
    int pixinVector;
    int nbdec;

    vectorSelect = pos/vectorbps;
    pixinVector = pos % vectorbps;

    nbdec = vectorbps-((fgBIT) 1)-pixinVector;
    mask = ((fgBIT) 1) << (nbdec);

    val=array[vectorSelect];
    val &= mask;
    val = val >> nbdec;
    return val;
  }

  static __inline__ fgUINT8 flgr_get_array_fgUINT8(fgUINT8* array, int pos) {
    return array[pos];
  }

  static __inline__ fgUINT16 flgr_get_array_fgUINT16(fgUINT16* array, int pos) {
    return array[pos];
  }

  static __inline__ fgUINT32 flgr_get_array_fgUINT32(fgUINT32* array, int pos) {
    return array[pos];
  }

  static __inline__ fgUINT64 flgr_get_array_fgUINT64(fgUINT64* array, int pos) {
    return array[pos];
  }

   static __inline__ fgINT8 flgr_get_array_fgINT8(fgINT8* array, int pos) {
    return array[pos];
  }

  static __inline__ fgINT16 flgr_get_array_fgINT16(fgINT16* array, int pos) {
    return array[pos];
  }

  static __inline__ fgINT32 flgr_get_array_fgINT32(fgINT32* array, int pos) {
    return array[pos];
  }

  static __inline__ fgINT64 flgr_get_array_fgINT64(fgINT64* array, int pos) {
    return array[pos];
  }

  static __inline__ fgFLOAT32 flgr_get_array_fgFLOAT32(fgFLOAT32* array, int pos) {
    return array[pos];
  }

  static __inline__ fgFLOAT64 flgr_get_array_fgFLOAT64(fgFLOAT64* array, int pos) {
    return array[pos];
  }









  static __inline__ void flgr_get_data_array_vector_fgBIT(fgBIT *vector_array, fgBIT *data_array, int spp, int pos) {
    register fgBIT val;
    register int i,k;

    for(k=0,i=pos*spp ; k<spp ; k++,i++) {
      val = flgr_get_array_fgBIT(data_array,i);
      flgr_set_array_fgBIT(vector_array,k,val);
    }
  }

  static __inline__ void flgr_get_data_array_vector_fgUINT8(fgUINT8 *vector_array, fgUINT8 *data_array, int spp, int pos) {
    register fgUINT8 val;
    register int i,k;

    for(k=0,i=pos*spp ; k<spp ; k++,i++) {
      val = flgr_get_array_fgUINT8(data_array,i);
      flgr_set_array_fgUINT8(vector_array,k,val);
    }
  }

  static __inline__ void flgr_get_data_array_vector_fgUINT16(fgUINT16 *vector_array, fgUINT16 *data_array, int spp, int pos) {
    register fgUINT16 val;
    register int i,k;

    for(k=0,i=pos*spp ; k<spp ; k++,i++) {
      val = flgr_get_array_fgUINT16(data_array,i);
      flgr_set_array_fgUINT16(vector_array,k,val);
    }
  }

  static __inline__ void flgr_get_data_array_vector_fgUINT32(fgUINT32 *vector_array, fgUINT32 *data_array, int spp, int pos) {
    register fgUINT32 val;
    register int i,k;

    for(k=0,i=pos*spp ; k<spp ; k++,i++) {
      val = flgr_get_array_fgUINT32(data_array,i);
      flgr_set_array_fgUINT32(vector_array,k,val);
    }
  }

  static __inline__ void flgr_get_data_array_vector_fgINT8(fgINT8 *vector_array, fgINT8 *data_array, int spp, int pos) {
    register fgINT8 val;
    register int i,k;

    for(k=0,i=pos*spp ; k<spp ; k++,i++) {
      val = flgr_get_array_fgINT8(data_array,i);
      flgr_set_array_fgINT8(vector_array,k,val);
    }
  }

  static __inline__ void flgr_get_data_array_vector_fgINT16(fgINT16 *vector_array, fgINT16 *data_array, int spp, int pos) {
    register fgINT16 val;
    register int i,k;

    for(k=0,i=pos*spp ; k<spp ; k++,i++) {
      val = flgr_get_array_fgINT16(data_array,i);
      flgr_set_array_fgINT16(vector_array,k,val);
    }
  }

  static __inline__ void flgr_get_data_array_vector_fgINT32(fgINT32 *vector_array, fgINT32 *data_array, int spp, int pos) {
    register fgINT32 val;
    register int i,k;

    for(k=0,i=pos*spp ; k<spp ; k++,i++) {
      val = flgr_get_array_fgINT32(data_array,i);
      flgr_set_array_fgINT32(vector_array,k,val);
    }
  }

  static __inline__ void flgr_get_data_array_vector_fgFLOAT32(fgFLOAT32 *vector_array, fgFLOAT32 *data_array, int spp, int pos) {
    register fgFLOAT32 val;
    register int i,k;

    for(k=0,i=pos*spp ; k<spp ; k++,i++) {
      val = flgr_get_array_fgFLOAT32(data_array,i);
      flgr_set_array_fgFLOAT32(vector_array,k,val);
    }
  }

  static __inline__ void flgr_get_data_array_vector_fgFLOAT64(fgFLOAT64 *vector_array, fgFLOAT64 *data_array, int spp, int pos) {
    register fgFLOAT64 val;
    register int i,k;

    for(k=0,i=pos*spp ; k<spp ; k++,i++) {
      val = flgr_get_array_fgFLOAT64(data_array,i);
      flgr_set_array_fgFLOAT64(vector_array,k,val);
    }
  }



 






  static __inline__ void flgr_set_data_array_vector_fgBIT(fgBIT *data_array, fgBIT *vector_array, int spp, int pos) {
    register fgBIT val;
    register int i,k;

    for(k=0,i=pos*spp ; k<spp ; k++,i++) {
      val = flgr_get_array_fgBIT(vector_array,k);
      flgr_set_array_fgBIT(data_array,i,val);
    }
  }

  static __inline__ void flgr_set_data_array_vector_fgUINT8(fgUINT8 *data_array, fgUINT8 *vector_array, int spp, int pos) {
    register fgUINT8 val;
    register int i,k;

    for(k=0,i=pos*spp ; k<spp ; k++,i++) {
      val = flgr_get_array_fgUINT8(vector_array,k);
      flgr_set_array_fgUINT8(data_array,i,val);
    }
  }

  static __inline__ void flgr_set_data_array_vector_fgUINT16(fgUINT16 *data_array, fgUINT16 *vector_array, int spp, int pos) {
    register fgUINT16 val;
    register int i,k;

    for(k=0,i=pos*spp ; k<spp ; k++,i++) {
      val = flgr_get_array_fgUINT16(vector_array,k);
      flgr_set_array_fgUINT16(data_array,i,val);
    }
  }

  static __inline__ void flgr_set_data_array_vector_fgUINT32(fgUINT32 *data_array, fgUINT32 *vector_array, int spp, int pos) {
    register fgUINT32 val;
    register int i,k;

    for(k=0,i=pos*spp ; k<spp ; k++,i++) {
      val = flgr_get_array_fgUINT32(vector_array,k);
      flgr_set_array_fgUINT32(data_array,i,val);
    }
  }

  static __inline__ void flgr_set_data_array_vector_fgINT8(fgINT8 *data_array, fgINT8 *vector_array, int spp, int pos) {
    register fgINT8 val;
    register int i,k;

    for(k=0,i=pos*spp ; k<spp ; k++,i++) {
      val = flgr_get_array_fgINT8(vector_array,k);
      flgr_set_array_fgINT8(data_array,i,val);
    }
  }

  static __inline__ void flgr_set_data_array_vector_fgINT16(fgINT16 *data_array, fgINT16 *vector_array, int spp, int pos) {
    register fgINT16 val;
    register int i,k;

    for(k=0,i=pos*spp ; k<spp ; k++,i++) {
      val = flgr_get_array_fgINT16(vector_array,k);
      flgr_set_array_fgINT16(data_array,i,val);
    }
  }

  static __inline__ void flgr_set_data_array_vector_fgINT32(fgINT32 *data_array, fgINT32 *vector_array, int spp, int pos) {
    register fgINT32 val;
    register int i,k;

    for(k=0,i=pos*spp ; k<spp ; k++,i++) {
      val = flgr_get_array_fgINT32(vector_array,k);
      flgr_set_array_fgINT32(data_array,i,val);
    }
  }

  static __inline__ void flgr_set_data_array_vector_fgFLOAT32(fgFLOAT32 *data_array, fgFLOAT32 *vector_array, int spp, int pos) {
    register fgFLOAT32 val;
    register int i,k;

    for(k=0,i=pos*spp ; k<spp ; k++,i++) {
      val = flgr_get_array_fgFLOAT32(vector_array,k);
      flgr_set_array_fgFLOAT32(data_array,i,val);
    }
  }

  static __inline__ void flgr_set_data_array_vector_fgFLOAT64(fgFLOAT64 *data_array, fgFLOAT64 *vector_array, int spp, int pos) {
    register fgFLOAT64 val;
    register int i,k;

    for(k=0,i=pos*spp ; k<spp ; k++,i++) {
      val = flgr_get_array_fgFLOAT64(vector_array,k);
      flgr_set_array_fgFLOAT64(data_array,i,val);
    }
  }



 

#endif

#ifdef __cplusplus
}
#endif
