/**
* @file hyantes.h
* @brief hyantes.c header
* @author Sebastien Martinez and Serge Guelton
* @date 2011-06-01
*
* This file is part of hyantes.
*
* hyantes is free software; you can redistribute it and/or modify
* it under the terms of the CeCILL-C License
*
* You should have received a copy of the CeCILL-C License
* along with this program.  If not, see <http://www.cecill.info/licences>.
*/

#ifndef _HYANTES_H_
#define _HYANTES_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef USE_FLOAT
typedef float data_t;
#define INPUT_FORMAT "%f%*[ \t]%f%*[ \t]%f"
#define OUTPUT_FORMAT "%f %f %f\n"
#define COS cosf
#define SIN sinf
#define ACOS acosf
#else
typedef double data_t;
#define INPUT_FORMAT "%lf%*[ \t]%lf%*[ \t]%lf"
#define OUTPUT_FORMAT "%lf %lf %lf\n"
#define COS cos
#define SIN sin
#define ACOS acos
#endif

/*  _______________________________________
 * /                                       \
 * |             Enumerations              |
 * \_______________________________________/
 */

/**
 * @brief enumeration of all available smoothing functions
 */
typedef enum
	{
		F_DISK, /**< Disk smoothing method */
		F_AMORTIZED_DISK, /**< smoothing method amortizing potential */ 
		F_GAUSSIAN, /**< smoothing method using gaussian distribution */
		F_EXPONENTIAL, /**< smoothing method using exponential distribution */
		F_PARETO	/**< smoothing method using pareto distribution */
	} smoothing_fun_t;


/**
* @brief enumeration of various options for hyantes
*/
    typedef enum
    {
        HS_PARSE_ONLY,
                    /**< (deprecated) only require generation of precomputed quadtree, extra arg : "char *filename" */
        HS_THRESHOLD,
                    /**< (deprecated) set the threshold used for ignoring some area, extra arg: "double threshold" */
        HS_LOAD_RAW,/**< (deprecated) tells the library to consider input file as a raw data file, no extra arg */
        HS_LOAD_PRECOMPUTED,
                    /**< (deprecated) tells the library to consider input file as a precomputed file, no extra arg */
        HS_SMOOTH_FUNC,
                    /**< tells the library to use given function and param to perform smoothing, extra arg: "char *funcname, double extra param, ... */
        HS_MODULE_OPT
                    /**< (deprecated) pass option to module */
    } hs_option_t;

/*  _______________________________________
 * /                                       \
 * |              Structures               |
 * \_______________________________________/
 */

/**
 * @brief configuration structure to use custom settings in hyantes fuctions
 */

typedef struct {
	
	FILE * g_file_serialize; /**< deprecated */
	
	double threshold; /**< deprecated */

	int g_is_raw_data; /**< deprecated */

	smoothing_fun_t fid; /**< smoothing function to use */

	double fparam; /**< parameter used by smoothing function */

	int herrno; /**< code of last error encountered */

	unsigned long status; /**< status of the execution */

} hs_config_t;

/** 
* @brief structure containing the coordinate of a potential
* all coordiante are given in radians
* which means that
* -90 <= latitude <= +90
*  and
* -180 <= longitude <= +180
*/
    typedef struct hs_potential
    {
        data_t lat;
                /**< latitude of the potential*/
        data_t lon;
                /**< longitude of the potential*/
        data_t pot;
                /**< value of the potential*/
    } hs_potential_t;


/** 
* @brief structure containing the coordinates of an area
* all coordinate are given in degree
* which means that -90 <= latitude <= 90 and -180 <= longitude <= +180
*/
    typedef struct hs_coord
    {
        data_t mLat;
                 /**< minimum latitude*/
        data_t mLon;
                 /**< minimum longitude*/
        data_t MLat;
                 /**< maximum latitude*/
        data_t MLon;
                 /**< maximum longitude*/
    } hs_coord_t;

/*  _______________________________________
 * /                                       \
 * |               Functions               |
 * \_______________________________________/
 */

void hs_display(size_t rangex, size_t rangey, hs_potential_t pt[rangex][rangey]);

int hs_set_r(hs_config_t *, hs_option_t, ...);
int hs_set(hs_option_t, ...);

hs_potential_t * hs_smooth(int _reoLat, int _resoLon, hs_coord_t visu, FILE * pFileReference);

hs_potential_t * hs_smooth_r(int _reoLat, int _resoLon, hs_coord_t visu, FILE * pFileReference, hs_config_t * config);

hs_potential_t * hs_smoothing (int _resoLat, int _resoLon, const char *function_name, double function_param, hs_coord_t visu, FILE * pFileReference);

unsigned long hs_status();

const char ** hs_list_smoothing (size_t *sz);

#endif
