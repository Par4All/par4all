/**
* @file hyantes.c
* @brief core of hyantes
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

#include "options.h"
#include <errno.h>

/*  _________________________________
 * /                                 \
 * |              Globals            |
 * \_________________________________/
 */

/** 
* @brief static array containg function names.
* each function must also be listed in func_ptrs in the same order
*/
char const * const func_names[] = {
    "disk",
    "amortized_disk",
    "gaussian",
    "exponential",
    "pareto"
};

hs_config_t g_config = { NULL,0,0,0,500,0,0 }; // global configuration used by default


/*  _________________________________
 * /                                 \
 * |  potential computing functions  |
 * \_________________________________/
 */

#define POW2(a) ((a)*(a))

/*
 * We will use cpp to generate one do_run function per smoothing method
 */

#define AMORTIZED_DISK(pot,dst,res,range) do{res=((pot)/(1+(dst)));}while (0)
#define SMOOTHING_FUN AMORTIZED_DISK
#include "hyantes_run.c"

#define DISK(pot,dst,res,range) do{res=(pot);}while (0)
#define SMOOTHING_FUN DISK
#include "hyantes_run.c"

#define GAUSSIAN(pot,dst,res,range) do{res=(pot)*(exp(-(M_PI/(4.*POW2(range)))*POW2(dst)));} while (0)
#define SMOOTHING_FUN GAUSSIAN
#include "hyantes_run.c"

#define EXPONENTIAL(pot,dst,res,range) do{res=(pot)*(exp(-(2./(range))*(dst)));} while (0)
#define SMOOTHING_FUN EXPONENTIAL
#include "hyantes_run.c"

#define PARETO(pot,dst,res,range) do{data_t tmp = POW2(dst); res=(pot)*(1./(1+(2/(range)*POW2(tmp))));} while (0)
#define SMOOTHING_FUN PARETO
#include "hyantes_run.c"

/**
 * @brief dispatch call to the right smoothing function
 */

static void do_run(hs_coord_t visu, data_t lonStep, data_t latStep, data_t range, size_t lonRange, size_t latRange, size_t nb, hs_potential_t plots[latRange][lonRange], hs_potential_t the_towns[nb], hs_config_t * configuration){

	/*data_t (*contrib)[latRange][lonRange] = malloc(sizeof(data_t)*latRange*lonRange);
	if(!contrib) {
		configuration->herrno=ENOMEM;
		return;
	}*/
	
	switch(configuration->fid){	
		case F_DISK:
			do_run_DISK(visu.mLon*M_PI/180, visu.mLat*M_PI/180, lonStep, latStep , range, lonRange, latRange, nb, plots, the_towns, configuration);
			break;
		case F_AMORTIZED_DISK:
			do_run_AMORTIZED_DISK(visu.mLon*M_PI/180, visu.mLat*M_PI/180, lonStep, latStep , range, lonRange, latRange, nb, plots, the_towns, configuration);
			break;
		case F_GAUSSIAN:
			do_run_GAUSSIAN(visu.mLon*M_PI/180, visu.mLat*M_PI/180, lonStep, latStep , range, lonRange, latRange, nb, plots, the_towns, configuration);
			break;
		case F_EXPONENTIAL:
			do_run_EXPONENTIAL(visu.mLon*M_PI/180, visu.mLat*M_PI/180, lonStep, latStep , range, lonRange, latRange, nb, plots, the_towns, configuration);
			break;
		case F_PARETO:
			do_run_PARETO(visu.mLon*M_PI/180, visu.mLat*M_PI/180, lonStep, latStep , range, lonRange, latRange, nb, plots, the_towns, configuration);
			break;
		default:
			do_run_DISK(visu.mLon*M_PI/180, visu.mLat*M_PI/180, lonStep, latStep , range, lonRange, latRange, nb, plots, the_towns, configuration);
	}
	
	/*free(contrib);*/
}

/*  _________________________________
 * /                                 \
 * | parser for potential input file |
 * \_________________________________/
 */

/**
 * @brief reads a file containing towns and parse it
 * @param fd file containg the towns
 * @param len pointer to the number of towns
 * @return a vector of towns 
 */

static hs_potential_t * hs_read_towns(FILE *fd, size_t* len, hs_config_t * config)
{
    size_t curr=0;
	size_t nb = 1;
    hs_potential_t *the_towns = malloc(sizeof(hs_potential_t));
	if(!the_towns) {
		config->herrno=ENOMEM;
		return NULL;
	}
    fputs("begin parsing ...\n",stderr);

    while(!feof(fd))
    {
        if(nb==curr)
        {
            nb*=2;
            the_towns=realloc(the_towns,nb*sizeof(hs_potential_t));
			if(!the_towns) {
				config->herrno=ENOMEM;
				return NULL;
			}
        }
        if(fscanf(fd,INPUT_FORMAT,&the_towns[curr].lat,&the_towns[curr].lon,&the_towns[curr].pot) !=3 )
        {
            while(!feof(fd))
            {
                char c=(char)fgetc(fd);
                if(c=='\n' || c=='\r' || c=='#') break;
            }
        }
        else
        {
            the_towns[curr].lat*=M_PI/180;
            the_towns[curr].lon*=M_PI/180;
            ++curr;
        }
    }
    the_towns=realloc(the_towns,curr*sizeof(hs_potential_t));
	if(!the_towns) {
		config->herrno=ENOMEM;
		return NULL;
	}
    *len = curr;
    fprintf(stderr,"parsed %zd towns\n",curr);
    return the_towns;
}

/*  _________________________________
 * /                                 \
 * |          User functions         |
 * \_________________________________/
 */


/**
 * @brief displays the matrix of processed potentials
 * @param lonRange the longitudinal resolution of the matrix
 * @param latRange the resolution of the matrix
 * @param pt the matrix of potential which is of size latRange by lonRange
*/
void hs_display(size_t lonRange, size_t latRange, hs_potential_t pt[latRange][lonRange])
{
    for(size_t i=0;i<latRange;i++)
    {
        for(size_t j=0;j<lonRange;j++)
            printf(OUTPUT_FORMAT,pt[i][j].lon,pt[i][j].lat,pt[i][j].pot);
        putchar('\n');
    }
}

/** 
* @brief performs the smoothing of target area inside visu, using potentials from pFileReference
* the smoothing is performed using smoothing method given by hs_set(HS_SMOOTH_FUNC, ... ) 
* the resolution of the output matrix will be resoLat x resoLon
* 
* @param _resoLat number of latitude points computed
* @param _resoLon  number of longitude points computed
* @param visu visualization window 
* @param pFileReference file containg the data in the format 
*    latitude longitude potential
*    latitude longitude potential
*    ...
*    latitude longitude potential
*  where latitude and longitude are given in degrees
* 
* @return an allocated array of size resoLat x resoLon containing a struct (lat, lon, pot) 
*   or
*       NULL if an error occured
*/
hs_potential_t * hs_smooth (int _resoLat, int _resoLon, hs_coord_t visu, FILE * pFileReference)
{
	return hs_smooth_r(_resoLat,_resoLon,visu,pFileReference,&g_config);
}

/**
 * @brief list all available smoothing methods that can be configured using hs_config
 * @param pointer to the number of smoothing methods 
 * @return array of string constant of size *sz. Memory is still owned by hyantes 
 */
const char ** hs_list_smoothing (size_t * sz){
    static const size_t names_count = sizeof(func_names) / sizeof(*func_names);
    *sz = names_count;
    return (char const **)/*cast for backward compatibility only*/func_names;
}
/**
 * @brief observer of the execution of the computation
 * @return number of computed input potential points from the beginning of the computation
 */
unsigned long hs_status (){
	return g_config.status;	
}


/** 
* @brief performs the smoothing of target area inside visu, using potentials from pFileReference and using given hs_config
* the smoothing is performed using smoothing method acording to the configuration given in the arguments 
* the resolution of the output matrix will be resoLat x resoLon
* 
* @param _resoLat number of latitude points computed
* @param _resoLon  number of longitude points computed
* @param visu visualization window 
* @param pFileReference file containg the data in the format 
*    latitude longitude potential
*    latitude longitude potential
*    ...
*    latitude longitude potential
*  where latitude and longitude are given in degrees
* @param configuration configuration to use 
* @return an allocated array of size resoLat x resoLon containing structs (lat, lon, pot) 
*   or
*       NULL if an error occured
*/
hs_potential_t * hs_smooth_r (int _resoLat, int _resoLon, hs_coord_t visu, FILE * pFileReference, hs_config_t *configuration)
{

	if ((_resoLat <= 0) || (_resoLon <= 0)){
		return NULL;
	}

	size_t lonRange = (size_t) _resoLon;
	size_t latRange = (size_t) _resoLat;
	
	size_t nb;
	hs_potential_t *the_towns = hs_read_towns(pFileReference,&nb,configuration);
	hs_potential_t (*plots)[latRange][lonRange] = malloc(sizeof(hs_potential_t)*latRange*lonRange);
	if(!plots) {
		configuration->herrno=ENOMEM;
		return NULL;
	}
	data_t lonStep = (visu.MLon - visu.mLon)/_resoLon;
	data_t latStep = (visu.MLat - visu.mLat)/_resoLat;
	
	data_t range = (*configuration).fparam;

    /* init step: prepare output array */
    for(size_t i=0;i<latRange;i++) {
        for(size_t j=0;j<lonRange;j++) {
            (*plots)[i][j].lon=(visu.mLon+lonStep*j);
            (*plots)[i][j].lat=(visu.mLat+latStep*i);
            (*plots)[i][j].pot=0.;
        }
    }

	lonStep *=M_PI/180;
	latStep *=M_PI/180;

	do_run(visu,lonStep,latStep,range,lonRange,latRange,nb,*plots,*(hs_potential_t (*)[nb]) the_towns,configuration);
	free(the_towns);
	return (hs_potential_t*) plots; 	
}

/**
 * @brief perform the smoothing of target area inside visu, using potentials from pFileReference the smoothing is performed using function_name smoothing method, with a radius of function_param the resolution of the output matrix will be resoLat x resoLon  (obsolete function, use hs_smmoth_r instead)
 * @param _resoLat number of latitude points computed
 * @param _resoLon  number of longitude points computed
 * @param function_name name of a smoothing method listed by hs_list_smoothing 
 * @param parameter (in kilometers) of the smoothing method 
 * @param visu visualization window
 * @param file containg the data in the format latitude longitude potential latitude longitude potential ... latitude longitude potential where latitude and longitude are given in degrees
 * @return an allocated array of size resoLat x resoLon containing structs (lat, lon, pot) 
 */

hs_potential_t * hs_smoothing (int _resoLat, int _resoLon, const char *function_name, double function_param, hs_coord_t visu, FILE * pFileReference){
	hs_config_t config = { NULL,0,0,0,500,0,0 };
	hs_set_r(&config,HS_SMOOTH_FUNC,function_name,function_param);
	return hs_smooth_r (_resoLat, _resoLon, visu, pFileReference, &config);
}

