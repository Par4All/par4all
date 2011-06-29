/**
* @file options.c
* @brief functions to set and define parameters
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
#include <stdarg.h>
#include <string.h>

/**
 * @brief sets a given option of the given hs_config_t configuration 
 * @param config pointer to the configuration structure to modify
 * @param opt the configuration option
 * @param ap va_list of parameters to configure the given option
 * @return 1 if setting went well, 0 otherwise
 */
static int vhs_set_r(hs_config_t * config, hs_option_t opt, va_list *ap) {
	int res = 1;
	switch (opt){
		case HS_PARSE_ONLY:
			res = 0;
			break;
		case HS_THRESHOLD:
			break;
		case HS_LOAD_RAW:
			res=0;
			break;
		case HS_LOAD_PRECOMPUTED:
			res=0;
			break;
		case HS_SMOOTH_FUNC:
			{
			char *fname = va_arg(*ap, char *);
			double fparam = va_arg(*ap, double);
			config->fparam = fparam;
			//set_func_inter(fname, fparam);
			size_t sz;
			size_t i;
			hs_list_smoothing(&sz);
			for (i=0;i<sz;i++){
				if(strcmp(func_names[i],fname)==0){
				config->fid = (smoothing_fun_t) i;
				break;
				}
			}
			if (i==sz){
				res = 0;
				fprintf(stderr,"error : unreconized smoothing function \n");
			}
			}
			break;
		case HS_MODULE_OPT:
			//res = init_module(ap);
			res = 0;
			break;

		default:
			fprintf(stderr, "[hs_set] unknow option \n");
			res = 0;
	};
	if(!res){config->herrno = EINVAL;}
	return res;
}

/**
 * @brief sets the given option to the given parameters in the given configuration 
 * @param config pointer to the configuration to use
 * @param opt option to set
 * @param ... list of arguments 
 * @return 1 if setting went well, 0 otherwise
 */
int hs_set_r(hs_config_t *config, hs_option_t opt, ...){
	va_list args;
	va_start(args, opt);
	int res= vhs_set_r(config, opt,&args);
	va_end(args);
	return res;
}

/**
 * @brief sets the given option to the given parameters in the default configuration (deprecated, you should use your own configuration structure)
 * @param opt option to set
 * @param ... list of arguments 
 * @return 1 if setting went well, 0 otherwise
 */
int hs_set(hs_option_t opt, ...){
	va_list args;
	va_start(args, opt);
	int res= vhs_set_r(&g_config, opt,&args);
	va_end(args);
	return res;
}
