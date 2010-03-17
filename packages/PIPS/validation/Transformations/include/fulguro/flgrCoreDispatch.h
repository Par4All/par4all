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


#ifndef __FLGR_CORE_DISPATCH_H
#define __FLGR_CORE_DISPATCH_H

#include <flgrCoreDll.h>
#include <flgrCoreTypes.h>
#include <flgrCoreErrors.h>
#include <flgrCoreData.h>


#define flgr_no_define_type_function(type)  POST_ERROR("data in <%s> format is not implemented!\n",flgr_get_type_string(type));


#define FLGR_FULL_PROCEDURE_DISPATCH(typevar,function_base,...)		\
  if(typevar == FLGR_BIT) {						\
    POST_DEBUG("launching "#function_base"_fgBIT\n");			\
    function_base##_fgBIT(__VA_ARGS__);					\
    return FLGR_RET_OK;							\
  }									\
  if(typevar==FLGR_UINT8) {						\
    POST_DEBUG("launching "#function_base"_fgUINT8\n");			\
    function_base##_fgUINT8(__VA_ARGS__);				\
    return FLGR_RET_OK;							\
  }									\
  if(typevar==FLGR_UINT16) {						\
    POST_DEBUG("launching "#function_base"_fgUINT16\n");		\
    function_base##_fgUINT16(__VA_ARGS__);				\
    return FLGR_RET_OK;							\
  }									\
  if(typevar==FLGR_UINT32) {						\
    POST_DEBUG("launching "#function_base"_fgUINT32\n");		\
    function_base##_fgUINT32(__VA_ARGS__);				\
    return FLGR_RET_OK;							\
  }									\
  if(typevar==FLGR_UINT64) {						\
    POST_DEBUG("launching "#function_base"_fgUINT64\n");		\
    function_base##_fgUINT64(__VA_ARGS__);				\
    return FLGR_RET_OK;							\
  }									\
  if(typevar==FLGR_INT8) {						\
    POST_DEBUG("launching "#function_base"_fgINT8\n");			\
    function_base##_fgINT8(__VA_ARGS__);				\
    return FLGR_RET_OK;							\
  }									\
  if(typevar==FLGR_INT16) {						\
    POST_DEBUG("launching "#function_base"_fgINT16\n");			\
    function_base##_fgINT16(__VA_ARGS__);				\
    return FLGR_RET_OK;							\
  }									\
  if(typevar==FLGR_INT32) {						\
    POST_DEBUG("launching "#function_base"_fgINT32\n");			\
    function_base##_fgINT32(__VA_ARGS__);				\
    return FLGR_RET_OK;							\
  }									\
  if(typevar==FLGR_INT64) {						\
    POST_DEBUG("launching "#function_base"_fgINT64\n");			\
    function_base##_fgINT64(__VA_ARGS__);				\
    return FLGR_RET_OK;							\
  }									\
  if(typevar==FLGR_FLOAT32) {						\
    POST_DEBUG("launching "#function_base"_fgFLOAT32\n");		\
    function_base##_fgFLOAT32(__VA_ARGS__);				\
    return FLGR_RET_OK;							\
  }									\
  if(typevar==FLGR_FLOAT64) {						\
    POST_DEBUG("launching "#function_base"_fgFLOAT64\n");		\
    function_base##_fgFLOAT64(__VA_ARGS__);				\
    return FLGR_RET_OK;							\
  }									\
  POST_ERROR("type <%s> unknown!\n",flgr_get_type_string(typevar));	\
  return FLGR_RET_TYPE_UNKNOWN


#define FLGR_FULL_FUNCTION_DISPATCH(fail,typevar,function_base,...)	\
  if(typevar == FLGR_BIT) {						\
    POST_DEBUG("launching "#function_base"_fgBIT\n");			\
    return function_base##_fgBIT(__VA_ARGS__);				\
  }									\
  if(typevar==FLGR_UINT8) {						\
    POST_DEBUG("launching "#function_base"_fgUINT8\n");			\
    return function_base##_fgUINT8(__VA_ARGS__);			\
  }									\
  if(typevar==FLGR_UINT16) {						\
    POST_DEBUG("launching "#function_base"_fgUINT16\n");		\
    return function_base##_fgUINT16(__VA_ARGS__);			\
  }									\
  if(typevar==FLGR_UINT32) {						\
    POST_DEBUG("launching "#function_base"_fgUINT32\n");		\
    return function_base##_fgUINT32(__VA_ARGS__);			\
  }									\
  if(typevar==FLGR_UINT64) {						\
    POST_DEBUG("launching "#function_base"_fgUINT64\n");		\
    return function_base##_fgUINT64(__VA_ARGS__);			\
  }									\
  if(typevar==FLGR_INT8) {						\
    POST_DEBUG("launching "#function_base"_fgINT8\n");			\
    return function_base##_fgINT8(__VA_ARGS__);				\
  }									\
  if(typevar==FLGR_INT16) {						\
    POST_DEBUG("launching "#function_base"_fgINT16\n");			\
    return function_base##_fgINT16(__VA_ARGS__);			\
  }									\
  if(typevar==FLGR_INT32) {						\
    POST_DEBUG("launching "#function_base"_fgINT32\n");			\
    return function_base##_fgINT32(__VA_ARGS__);			\
  }									\
  if(typevar==FLGR_INT64) {						\
    POST_DEBUG("launching "#function_base"_fgINT64\n");			\
    return function_base##_fgINT64(__VA_ARGS__);			\
  }									\
  if(typevar==FLGR_FLOAT32) {						\
    POST_DEBUG("launching "#function_base"_fgFLOAT32\n");		\
    return function_base##_fgFLOAT32(__VA_ARGS__);			\
  }									\
  if(typevar==FLGR_FLOAT64) {						\
    POST_DEBUG("launching "#function_base"_fgFLOAT64\n");		\
    return function_base##_fgFLOAT64(__VA_ARGS__);			\
  }									\
  POST_ERROR("type <%s> unknown!\n",flgr_get_type_string(typevar));	\
  return fail


#define FLGR_FULL_DISPATCH_CONST_FUNCTION(fail, typevar, fctbase, cte, ...) \
  switch(typevar) {							\
  case FLGR_BIT:							\
    return fctbase##_fgBIT(__VA_ARGS__,FLGR_PVOID_VAL(fgBIT,cte));	\
  case FLGR_UINT8:							\
    return fctbase##_fgUINT8(__VA_ARGS__,FLGR_PVOID_VAL(fgUINT8,cte));	\
  case FLGR_UINT16:							\
    return fctbase##_fgUINT16(__VA_ARGS__,FLGR_PVOID_VAL(fgUINT16,cte)); \
  case FLGR_UINT32:							\
    return fctbase##_fgUINT32(__VA_ARGS__,FLGR_PVOID_VAL(fgUINT32,cte)); \
  case FLGR_UINT64:							\
    return fctbase##_fgUINT64(__VA_ARGS__,FLGR_PVOID_VAL(fgUINT64,cte)); \
  case FLGR_INT8:							\
    return fctbase##_fgINT8(__VA_ARGS__,FLGR_PVOID_VAL(fgINT8,cte));	\
  case FLGR_INT16:							\
    return fctbase##_fgINT16(__VA_ARGS__,FLGR_PVOID_VAL(fgINT16,cte));	\
  case FLGR_INT32:							\
    return fctbase##_fgINT32(__VA_ARGS__,FLGR_PVOID_VAL(fgINT32,cte));	\
  case FLGR_INT64:							\
    return fctbase##_fgINT64(__VA_ARGS__,FLGR_PVOID_VAL(fgINT64,cte));	\
  case FLGR_FLOAT32:							\
    return fctbase##_fgFLOAT32(__VA_ARGS__,FLGR_PVOID_VAL(fgFLOAT32,cte)); \
  case FLGR_FLOAT64:							\
    return fctbase##_fgFLOAT64(__VA_ARGS__,FLGR_PVOID_VAL(fgFLOAT64,cte)); \
  default:								\
    POST_ERROR("type unknown!\n");					\
    return fail;							\
  }									\
  return fail

#define FLGR_FULL_DISPATCH_CONST_PROCEDURE(typevar, fctbase, cte, ...)	\
  switch(typevar) {							\
  case FLGR_BIT:							\
    fctbase##_fgBIT(__VA_ARGS__,FLGR_PVOID_VAL(fgBIT,cte));break;	\
  case FLGR_UINT8:							\
    fctbase##_fgUINT8(__VA_ARGS__,FLGR_PVOID_VAL(fgUINT8,cte));break;	\
  case FLGR_UINT16:							\
    fctbase##_fgUINT16(__VA_ARGS__,FLGR_PVOID_VAL(fgUINT16,cte));break; \
  case FLGR_UINT32:							\
    fctbase##_fgUINT32(__VA_ARGS__,FLGR_PVOID_VAL(fgUINT32,cte));break; \
  case FLGR_UINT64:							\
    fctbase##_fgUINT64(__VA_ARGS__,FLGR_PVOID_VAL(fgUINT64,cte));break; \
  case FLGR_INT8:							\
    fctbase##_fgINT8(__VA_ARGS__,FLGR_PVOID_VAL(fgINT8,cte));break;	\
  case FLGR_INT16:							\
    fctbase##_fgINT16(__VA_ARGS__,FLGR_PVOID_VAL(fgINT16,cte));break;	\
  case FLGR_INT32:							\
    fctbase##_fgINT32(__VA_ARGS__,FLGR_PVOID_VAL(fgINT32,cte));break;	\
  case FLGR_INT64:							\
    fctbase##_fgINT64(__VA_ARGS__,FLGR_PVOID_VAL(fgINT64,cte));break;	\
  case FLGR_FLOAT32:							\
    fctbase##_fgFLOAT32(__VA_ARGS__,FLGR_PVOID_VAL(fgFLOAT32,cte));break; \
  case FLGR_FLOAT64:							\
    fctbase##_fgFLOAT64(__VA_ARGS__,FLGR_PVOID_VAL(fgFLOAT64,cte));break; \
  default:								\
    POST_ERROR("type unknown!\n");					\
    return FLGR_RET_TYPE_UNKNOWN;					\
  }									\
  return FLGR_RET_OK


#define FLGR_FULL_DISPATCH_CONST_PTR_FUNCTION(fail, typevar, fctbase, cte, ...) \
  switch(typevar) {							\
  case FLGR_BIT:							\
    return fctbase##_fgBIT(__VA_ARGS__,FLGR_PVOID_PTR(fgBIT,cte));	\
  case FLGR_UINT8:							\
    return fctbase##_fgUINT8(__VA_ARGS__,FLGR_PVOID_PTR(fgUINT8,cte));	\
  case FLGR_UINT16:							\
    return fctbase##_fgUINT16(__VA_ARGS__,FLGR_PVOID_PTR(fgUINT16,cte)); \
  case FLGR_UINT32:							\
    return fctbase##_fgUINT32(__VA_ARGS__,FLGR_PVOID_PTR(fgUINT32,cte)); \
  case FLGR_UINT64:							\
    return fctbase##_fgUINT64(__VA_ARGS__,FLGR_PVOID_PTR(fgUINT64,cte)); \
  case FLGR_INT8:							\
    return fctbase##_fgINT8(__VA_ARGS__,FLGR_PVOID_PTR(fgINT8,cte));	\
  case FLGR_INT16:							\
    return fctbase##_fgINT16(__VA_ARGS__,FLGR_PVOID_PTR(fgINT16,cte));	\
  case FLGR_INT32:							\
    return fctbase##_fgINT32(__VA_ARGS__,FLGR_PVOID_PTR(fgINT32,cte));	\
  case FLGR_INT64:							\
    return fctbase##_fgINT64(__VA_ARGS__,FLGR_PVOID_PTR(fgINT64,cte));	\
  case FLGR_FLOAT32:							\
    return fctbase##_fgFLOAT32(__VA_ARGS__,FLGR_PVOID_PTR(fgFLOAT32,cte)); \
  case FLGR_FLOAT64:							\
    return fctbase##_fgFLOAT64(__VA_ARGS__,FLGR_PVOID_PTR(fgFLOAT64,cte)); \
  default:								\
    POST_ERROR("type unknown!\n");					\
    return fail;							\
  }									\
  return fail



#define FLGR_FULL_DISPATCH_CONST_PTR_PROCEDURE(typevar, fctbase, cte, ...) \
  switch(typevar) {							\
  case FLGR_BIT:							\
    fctbase##_fgBIT(__VA_ARGS__,FLGR_PVOID_PTR(fgBIT,cte));break;	\
  case FLGR_UINT8:							\
    fctbase##_fgUINT8(__VA_ARGS__,FLGR_PVOID_PTR(fgUINT8,cte));break;	\
  case FLGR_UINT16:							\
    fctbase##_fgUINT16(__VA_ARGS__,FLGR_PVOID_PTR(fgUINT16,cte));break;	\
  case FLGR_UINT32:							\
    fctbase##_fgUINT32(__VA_ARGS__,FLGR_PVOID_PTR(fgUINT32,cte));break;	\
  case FLGR_UINT64:							\
    fctbase##_fgUINT64(__VA_ARGS__,FLGR_PVOID_PTR(fgUINT64,cte));break;	\
  case FLGR_INT8:							\
    fctbase##_fgINT8(__VA_ARGS__,FLGR_PVOID_PTR(fgINT8,cte));break;	\
  case FLGR_INT16:							\
    fctbase##_fgINT16(__VA_ARGS__,FLGR_PVOID_PTR(fgINT16,cte));break;	\
  case FLGR_INT32:							\
    fctbase##_fgINT32(__VA_ARGS__,FLGR_PVOID_PTR(fgINT32,cte));break;	\
  case FLGR_INT64:							\
    fctbase##_fgINT64(__VA_ARGS__,FLGR_PVOID_PTR(fgINT64,cte));break;	\
  case FLGR_FLOAT32:							\
    fctbase##_fgFLOAT32(__VA_ARGS__,FLGR_PVOID_PTR(fgFLOAT32,cte));break; \
  case FLGR_FLOAT64:							\
    fctbase##_fgFLOAT64(__VA_ARGS__,FLGR_PVOID_PTR(fgFLOAT64,cte));break; \
  default:								\
    POST_ERROR("type unknown!\n");					\
    return FLGR_RET_TYPE_UNKNOWN;					\
  }									\
  return FLGR_RET_TYPE_UNKNOWN

























#define FLGR_DISPATCH_PROCEDURE(typevar,function_base,...)		\
  if(typevar == FLGR_BIT) {						\
    POST_DEBUG("launching "#function_base"_fgBIT\n");			\
    function_base##_fgBIT(__VA_ARGS__);					\
    return FLGR_RET_OK;							\
  }									\
  if(typevar==FLGR_UINT8) {						\
    POST_DEBUG("launching "#function_base"_fgUINT8\n");			\
    function_base##_fgUINT8(__VA_ARGS__);				\
    return FLGR_RET_OK;							\
  }									\
  if(typevar==FLGR_UINT16) {						\
    POST_DEBUG("launching "#function_base"_fgUINT16\n");		\
    function_base##_fgUINT16(__VA_ARGS__);				\
    return FLGR_RET_OK;							\
  }									\
  if(typevar==FLGR_UINT32) {						\
    POST_DEBUG("launching "#function_base"_fgUINT32\n");		\
    function_base##_fgUINT32(__VA_ARGS__);				\
    return FLGR_RET_OK;							\
  }									\
  if(typevar==FLGR_INT8) {						\
    POST_DEBUG("launching "#function_base"_fgINT8\n");			\
    function_base##_fgINT8(__VA_ARGS__);				\
    return FLGR_RET_OK;							\
  }									\
  if(typevar==FLGR_INT16) {						\
    POST_DEBUG("launching "#function_base"_fgINT16\n");			\
    function_base##_fgINT16(__VA_ARGS__);				\
    return FLGR_RET_OK;							\
  }									\
  if(typevar==FLGR_INT32) {						\
    POST_DEBUG("launching "#function_base"_fgINT32\n");			\
    function_base##_fgINT32(__VA_ARGS__);				\
    return FLGR_RET_OK;							\
  }									\
  if(typevar==FLGR_FLOAT32) {						\
    POST_DEBUG("launching "#function_base"_fgFLOAT32\n");		\
    function_base##_fgFLOAT32(__VA_ARGS__);				\
    return FLGR_RET_OK;							\
  }									\
  if(typevar==FLGR_FLOAT64) {						\
    POST_DEBUG("launching "#function_base"_fgFLOAT64\n");		\
    function_base##_fgFLOAT64(__VA_ARGS__);				\
    return FLGR_RET_OK;							\
  }									\
  POST_ERROR("type <%s> unknown!\n",flgr_get_type_string(typevar));	\
  return FLGR_RET_TYPE_UNKNOWN

#define FLGR_DISPATCH_FUNCTION(fail,typevar,function_base,...)		\
  if(typevar == FLGR_BIT) {						\
    POST_DEBUG("launching "#function_base"_fgBIT\n");			\
    return function_base##_fgBIT(__VA_ARGS__);				\
  }									\
  if(typevar == FLGR_UINT8) {						\
    POST_DEBUG("launching "#function_base"_fgUINT8\n");			\
    return function_base##_fgUINT8(__VA_ARGS__);			\
  }									\
  if(typevar == FLGR_UINT16) {						\
    POST_DEBUG("launching "#function_base"_fgUINT16\n");		\
    return function_base##_fgUINT16(__VA_ARGS__);			\
  }									\
  if(typevar == FLGR_UINT32) {						\
    POST_DEBUG("launching "#function_base"_fgUINT32\n");		\
    return function_base##_fgUINT32(__VA_ARGS__);			\
  }									\
  if(typevar == FLGR_INT8) {						\
    POST_DEBUG("launching "#function_base"_fgINT8\n");			\
    return function_base##_fgINT8(__VA_ARGS__);				\
  }									\
  if(typevar == FLGR_INT16) {						\
    POST_DEBUG("launching "#function_base"_fgINT16\n");			\
    return function_base##_fgINT16(__VA_ARGS__);			\
  }									\
  if(typevar == FLGR_INT32) {						\
    POST_DEBUG("launching "#function_base"_fgINT32\n");			\
    return function_base##_fgINT32(__VA_ARGS__);			\
  }									\
  if(typevar == FLGR_FLOAT32) {						\
    POST_DEBUG("launching "#function_base"_fgFLOAT32\n");		\
    return function_base##_fgFLOAT32(__VA_ARGS__);			\
  }									\
  if(typevar == FLGR_FLOAT64) {						\
    POST_DEBUG("launching "#function_base"_fgFLOAT64\n");		\
    return function_base##_fgFLOAT64(__VA_ARGS__);			\
  }									\
  POST_ERROR("type <%s> unknown!\n",flgr_get_type_string(typevar));	\
  return fail



#define FLGR_DISPATCH_CONST_PROCEDURE(typevar, fctbase, cte, ...)	\
  switch(typevar) {							\
  case FLGR_BIT:							\
    fctbase##_fgBIT(__VA_ARGS__,FLGR_PVOID_VAL(fgBIT,cte));break;	\
  case FLGR_UINT8:							\
    fctbase##_fgUINT8(__VA_ARGS__,FLGR_PVOID_VAL(fgUINT8,cte));break;	\
  case FLGR_UINT16:							\
    fctbase##_fgUINT16(__VA_ARGS__,FLGR_PVOID_VAL(fgUINT16,cte));break; \
  case FLGR_UINT32:							\
    fctbase##_fgUINT32(__VA_ARGS__,FLGR_PVOID_VAL(fgUINT32,cte));break; \
  case FLGR_INT8:							\
    fctbase##_fgINT8(__VA_ARGS__,FLGR_PVOID_VAL(fgINT8,cte));break;	\
  case FLGR_INT16:							\
    fctbase##_fgINT16(__VA_ARGS__,FLGR_PVOID_VAL(fgINT16,cte));break;	\
  case FLGR_INT32:							\
    fctbase##_fgINT32(__VA_ARGS__,FLGR_PVOID_VAL(fgINT32,cte));break;	\
  case FLGR_FLOAT32:							\
    fctbase##_fgFLOAT32(__VA_ARGS__,FLGR_PVOID_VAL(fgFLOAT32,cte));break; \
  case FLGR_FLOAT64:							\
    fctbase##_fgFLOAT64(__VA_ARGS__,FLGR_PVOID_VAL(fgFLOAT64,cte));break; \
  default:								\
    POST_ERROR("type unknown!\n");					\
    return FLGR_RET_TYPE_UNKNOWN;					\
  }									\
  return FLGR_RET_OK

#define FLGR_DISPATCH_CONST_FUNCTION(fail,typevar, fctbase, cte, ...)	\
  switch(typevar) {							\
  case FLGR_BIT:							\
    return fctbase##_fgBIT(__VA_ARGS__,FLGR_PVOID_VAL(fgBIT,cte));	\
  case FLGR_UINT8:							\
    return fctbase##_fgUINT8(__VA_ARGS__,FLGR_PVOID_VAL(fgUINT8,cte));	\
  case FLGR_UINT16:							\
    return fctbase##_fgUINT16(__VA_ARGS__,FLGR_PVOID_VAL(fgUINT16,cte)); \
  case FLGR_UINT32:							\
    return fctbase##_fgUINT32(__VA_ARGS__,FLGR_PVOID_VAL(fgUINT32,cte)); \
  case FLGR_INT8:							\
    return fctbase##_fgINT8(__VA_ARGS__,FLGR_PVOID_VAL(fgINT8,cte));	\
  case FLGR_INT16:							\
    return fctbase##_fgINT16(__VA_ARGS__,FLGR_PVOID_VAL(fgINT16,cte));	\
  case FLGR_INT32:							\
    return fctbase##_fgINT32(__VA_ARGS__,FLGR_PVOID_VAL(fgINT32,cte));	\
  case FLGR_FLOAT32:							\
    return fctbase##_fgFLOAT32(__VA_ARGS__,FLGR_PVOID_VAL(fgFLOAT32,cte)); \
  case FLGR_FLOAT64:							\
    return fctbase##_fgFLOAT64(__VA_ARGS__,FLGR_PVOID_VAL(fgFLOAT64,cte)); \
  default:								\
    POST_ERROR("type unknown!\n");					\
    return fail;							\
  }									\
  return fail

#define FLGR_DISPATCH_CONST_PTR_PROCEDURE(typevar, fctbase, cte, ...)	\
  switch(typevar) {							\
  case FLGR_BIT:							\
    fctbase##_fgBIT(__VA_ARGS__,FLGR_PVOID_PTR(fgBIT,cte));break;	\
  case FLGR_UINT8:							\
    fctbase##_fgUINT8(__VA_ARGS__,FLGR_PVOID_PTR(fgUINT8,cte));break;	\
  case FLGR_UINT16:							\
    fctbase##_fgUINT16(__VA_ARGS__,FLGR_PVOID_PTR(fgUINT16,cte));break; \
  case FLGR_UINT32:							\
    fctbase##_fgUINT32(__VA_ARGS__,FLGR_PVOID_PTR(fgUINT32,cte));break; \
  case FLGR_INT8:							\
    fctbase##_fgINT8(__VA_ARGS__,FLGR_PVOID_PTR(fgINT8,cte));break;	\
  case FLGR_INT16:							\
    fctbase##_fgINT16(__VA_ARGS__,FLGR_PVOID_PTR(fgINT16,cte));break;	\
  case FLGR_INT32:							\
    fctbase##_fgINT32(__VA_ARGS__,FLGR_PVOID_PTR(fgINT32,cte));break;	\
  case FLGR_FLOAT32:							\
    fctbase##_fgFLOAT32(__VA_ARGS__,FLGR_PVOID_PTR(fgFLOAT32,cte));break; \
  case FLGR_FLOAT64:							\
    fctbase##_fgFLOAT64(__VA_ARGS__,FLGR_PVOID_PTR(fgFLOAT64,cte));break; \
  default:								\
    POST_ERROR("type unknown!\n");					\
    return FLGR_RET_TYPE_UNKNOWN;					\
  }									\
  return FLGR_RET_OK

#define FLGR_DISPATCH_CONST_PTR_FUNCTION(fail,typevar, fctbase, cte, ...) \
  switch(typevar) {							\
  case FLGR_BIT:							\
    return fctbase##_fgBIT(__VA_ARGS__,FLGR_PVOID_PTR(fgBIT,cte));	\
  case FLGR_UINT8:							\
    return fctbase##_fgUINT8(__VA_ARGS__,FLGR_PVOID_PTR(fgUINT8,cte));	\
  case FLGR_UINT16:							\
    return fctbase##_fgUINT16(__VA_ARGS__,FLGR_PVOID_PTR(fgUINT16,cte)); \
  case FLGR_UINT32:							\
    return fctbase##_fgUINT32(__VA_ARGS__,FLGR_PVOID_PTR(fgUINT32,cte)); \
  case FLGR_INT8:							\
    return fctbase##_fgINT8(__VA_ARGS__,FLGR_PVOID_PTR(fgINT8,cte));	\
  case FLGR_INT16:							\
    return fctbase##_fgINT16(__VA_ARGS__,FLGR_PVOID_PTR(fgINT16,cte));	\
  case FLGR_INT32:							\
    return fctbase##_fgINT32(__VA_ARGS__,FLGR_PVOID_PTR(fgINT32,cte));	\
  case FLGR_FLOAT32:							\
    return fctbase##_fgFLOAT32(__VA_ARGS__,FLGR_PVOID_PTR(fgFLOAT32,cte)); \
  case FLGR_FLOAT64:							\
    return fctbase##_fgFLOAT64(__VA_ARGS__,FLGR_PVOID_PTR(fgFLOAT64,cte)); \
  default:								\
    POST_ERROR("type unknown!\n");					\
    return fail;					\
  }									\
  return fail


  EXPORT_LIB FLGR_Ret flgr_parse_str_constant(FLGR_Type type, int spp, char *strin, void *constout);
  EXPORT_LIB void *flgr_allocate_vector_const(FLGR_Type type, int spp, void *valueForEachVectorElement);



#endif

#ifdef __cplusplus
}
#endif
