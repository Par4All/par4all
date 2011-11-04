/**
* @file hyantes_run.c
* @brief definition of processing functions
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
#ifndef SMOOTHING_FUN
	#error no smoothing function defined
#endif

#define __DO_RUN(s) do_run_##s
#define _DO_RUN(s) __DO_RUN(s)
#define DO_RUN _DO_RUN(SMOOTHING_FUN)

/**
 * @brief calculates the contribution of every town in the given window using SMOOTHING_FUN method
 * @param lonMin window minuimum longitude
 * @param latMin window minimum latitude
 * @param lonStep step used to cover the area along longitude coordinates
 * @param latStep step used to cover the area along the latitude coordinates 
 * @param range range of releveance for potential contribution
 * @param lonRange window resolution for longitude 
 * @param latRange window resolution for latitude
 * @param nb number of town to compute
 * @param pt after running : potential matrix of size latRange by lonRange 
 * @param t vector of towns to compute
 * @param contrib after running : non normalized potential matrix of size latRange by lonRange 
 */
static void DO_RUN (data_t lonMin, data_t latMin, data_t lonStep, data_t latStep,data_t range, size_t lonRange, size_t latRange, size_t nb, hs_potential_t pt[latRange][lonRange], hs_potential_t t[nb], hs_config_t * config)
{
	data_t town_sum = 0.;
	data_t total_sum = 0.;
	config->status = 0;
	/*for each town, we shall calculate its contribution on the window */
	
#pragma omp parallel 
#pragma omp for reduction(+:town_sum,total_sum) 
	for(size_t k=0;k<nb;k++) {
        data_t pot = t[k].pot;
		town_sum+= pot;
        /* only process if it is relevant */
        if(pot > 0) {
            /* contribution step: compute contribution of t[k] to the whole map */
            data_t sum = 0.;

			data_t latmax = acos(cos(t[k].lat)*cos(range/6368.)-fabs(sin(t[k].lat)*sin(range/6368.)));
			data_t latmin = acos(cos(t[k].lat)*cos(range/6368.)+fabs(sin(t[k].lat)*sin(range/6368.)));

			if (latmin > t[k].lat) latmin = 2*t[k].lat - latmin;

			long int imin = floor((latmin-latMin)/latStep);
			size_t imax = 1+ceil((latmax-latMin)/latStep);
			
			if (imin < 0) imin = 0;
			if (imax > latRange) imax = latRange; 

			data_t deltalon = acos((cos(range/6368)-pow(sin(t[k].lat),2))/pow(cos(t[k].lat),2));
			data_t lonmax = t[k].lon + deltalon;	
			data_t lonmin = t[k].lon - deltalon;

			long int jmin = floor((lonmin - lonMin)/lonStep);
			size_t jmax = 1+ceil((lonmax-lonMin)/lonStep);

			if (jmin < 0) jmin = 0;
			if (jmax > lonRange) jmax = lonRange;

			data_t contrib[imax-imin+1][jmax-jmin+1];

            for(size_t i=imin;i<imax;i++) {
                for(size_t j=jmin;j<jmax;j++) {
                    data_t tmp = 
						6368.* ACOS(COS(latMin+latStep*i)*COS( t[k].lat ) * ( COS(lonMin+lonStep*j)*COS(t[k].lon)+SIN(lonMin+lonStep*j)*SIN(t[k].lon)) + SIN(latMin+latStep*i)*SIN(t[k].lat));
                    /* if distance from town is within range, set contribution */
					if( tmp < range ) {
                        /* The next ligne wil be replace by cpp to match the wanted smoothing function*/
						SMOOTHING_FUN(pot,tmp,contrib[i-imin][j-jmin],range);
                        sum+=contrib[i-imin][j-jmin];
                    }
                    else
                        contrib[i-imin][j-jmin] = 0;
                }
            }

            /* normalization step: make sure pot is fully represented by its contributions */
            if(sum > 0) {
                for(size_t i=imin;i<imax;i++) {
                    for(size_t j=jmin;j<jmax;j++) {
                        data_t c = contrib[i-imin][j-jmin];
                        if( c > 0 ) {
                            pt[i][j].pot+= c * pot  / sum;
							total_sum += c * pot / sum;	
						}
                    }
                }
            }
        }
		config->status = (unsigned long) k ;
	}
	
	if (fabs(town_sum - total_sum) > 0.0001){
		fprintf(stderr,"Warning : information lost during processing, you may consider increasing the window resolution\n");
	}
}
#undef SMOOTHING_FUN
