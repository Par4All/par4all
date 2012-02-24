#include <stdlib.h>
#include <stdio.h>
#include <netcdf.h>
#include <string.h>
#include "lyapunov.h"

/* Handle errors by printing an error message and exiting with a
 * non-zero status. */
#define ERRCODE 2
#define ERR(e) {printf("Error: %s\n", nc_strerror(e)); exit(ERRCODE);}
#define NDIMS 2


double defaultVal = -9999.;

void read_file (char* filename, double** pu, double** pv, size_t* xDim, size_t* yDim, size_t* tDim) {

  /* This will be the netCDF ID for the file and data variable. */
  int ncid, varid, lat_dimid, lon_dimid, record_dimid;
       
  /* Loop indexes, and error handling. */
  int x, y, retval;
  
  /* buffer */
  double *buff_U;
  double *buff_V;
  double *u,*v;
  
  /* Open the file. NC_NOWRITE tells netCDF we want read-only access
   * to the file.*/
  if ((retval = nc_open(filename, NC_NOWRITE, &ncid)))
    ERR(retval);
  
  
  if ((retval = nc_inq_dimid(ncid, "latitude", &lat_dimid)))  /* get ID for latitude dimension */
    ERR(retval);
  if ((retval = nc_inq_dimlen(ncid, lat_dimid, yDim))) /* get lat length */
    ERR(retval);
    
  if ((retval = nc_inq_dimid(ncid, "longitude", &lon_dimid)))  /* get ID for longitude dimension */
    ERR(retval);
  if ((retval = nc_inq_dimlen(ncid, lon_dimid, xDim))) /* get lat length */
    ERR(retval);

  if ((retval = nc_inq_dimid(ncid, "record", &record_dimid)))  /* get ID for record dimension */
    ERR(retval);
  if ((retval = nc_inq_dimlen(ncid, record_dimid, tDim))) /* get record length */
    ERR(retval);
  
  
  printf("read u[%d][%d][%d] ..\n",(int)*xDim, (int)*yDim, (int)*tDim);
  long size=(long)( (*xDim) * (*yDim) * (*tDim) * sizeof(double));
  
  buff_U=(double *)malloc(size);
    /* Get the varid of the data variable, based on its name. */
  if ((retval = nc_inq_varid(ncid, "u", &varid)))
    ERR(retval);
  /* Read the data. */
  if ((retval = nc_get_var_double(ncid, varid, buff_U)))
    ERR(retval);

  
  printf("read v[%d][%d][%d] ..\n",(int)*xDim, (int)*yDim, (int)*tDim);
  buff_V=(double *)malloc(size);
    /* Get the varid of the data variable, based on its name. */
  if ((retval = nc_inq_varid(ncid, "v", &varid)))
    ERR(retval);
  /* Read the data. */
  if ((retval = nc_get_var_double(ncid, varid, buff_V)))
    ERR(retval);
  
  /* Check the data. */
  /* for (x = 0; x < NX; x++)
     for (y = 0; y < NY; y++)
     if (data_in[x][y] != x * NY + y)
     return ERRCODE; */
     
  /* Close the file, freeing all resources. */
  if ((retval = nc_close(ncid)))
    ERR(retval);
  
  
  // transpose data
  int t,i,j;
  int nt,ni,nj;
  nt=(int)*tDim;
  ni=(int)*xDim;
  nj=(int)*yDim;
  u=(double *)malloc(size);
  v=(double *)malloc(size);
  for(t=0;t<nt;t++)
  {
	for(i=0;i<ni;i++)
	{ 
		for(j=0;j<nj;j++)
		{
			if (buff_U[i+j*ni] > 100. || buff_V[i+j*ni] > 100.) {
				u[ni*nj*t + i*nj + j] = 0.;
				v[ni*nj*t + i*nj + j] = 0.;
			} else {
				u[ni*nj*t + i*nj + j] = buff_U[ni*nj*t + j*ni + i];
				v[ni*nj*t + i*nj + j] = buff_V[ni*nj*t + j*ni + i];
			}
		}
	  //printf("%e %e \n",u[j+i*yDim],v[j+i*yDim]);
	}
    }
    free(buff_U);
    free(buff_V);
    *pu=u;
    *pv=v;
}
  
  



void write_file (char* filename, double* lya, int xDim, int yDim, double* lat, double* lon, long *time) {

  /* This will be the netCDF ID for the file and data variable. */
  int ncid, varid, x_dimid, y_dimid, t_dimid;
  int dimids[NDIMS];
       
  /* Loop indexes, and error handling. */
  int retval;
  int x,y;
  
  double *tlya;  // transpose of lya

   tlya=(double *)malloc(xDim*yDim*sizeof(double));
   for(x=0;x<xDim;x++)
   { 
 	for(y=0;y<yDim;y++)
 	{
 		tlya[y*xDim + x] = lya[ x*yDim + y];
 	}
 	 //printf("%e %e \n",u[j+i*yDim],v[j+i*yDim]);
   }
 
 
  printf("Ouverture\n");

  /* Create the file. The NC_CLOBBER parameter tells netCDF to
   * overwrite this file, if it already exists.*/
  if ((retval = nc_create(filename, NC_CLOBBER, &ncid)))
    ERR(retval);
  printf("Dimensions\n");
     
  /* Define the dimensions. NetCDF will hand back an ID for each. */
  if ((retval = nc_def_dim(ncid, "lon", xDim, &x_dimid)))
    ERR(retval);
  if ((retval = nc_def_dim(ncid, "lat", yDim, &y_dimid)))
    ERR(retval);
  //if ((retval = nc_def_dim(ncid, "time", 1, &t_dimid)))
  //  ERR(retval);
     
  /* The dimids array is used to pass the IDs of the dimensions of
   * the variable. */
  dimids[1] = x_dimid;
  dimids[0] = y_dimid;
//  dimids[0] = t_dimid;
  
    printf("Variable\n");

  /* Define the variables */
  printf("Lat\n");
  int lat_varid;
  if ((retval = nc_def_var(ncid, "lat", NC_DOUBLE, 1,
			   &y_dimid, &lat_varid)))
    ERR(retval);
  if ((retval = nc_put_att_text(ncid, lat_varid, "units",
				strlen("degrees_north"), "degrees_north")))
    ERR(retval);
  printf("Lon\n");
  int lon_varid;
  if ((retval = nc_def_var(ncid, "lon", NC_DOUBLE, 1,
			   &x_dimid, &lon_varid)))
    ERR(retval);
  
  if ((retval = nc_put_att_text(ncid, lon_varid, "units", strlen("degrees_east"), "degrees_east")))
	 ERR(retval);
  //printf("Time\n");
  //int time_varid;
  //if ((retval = nc_def_var(ncid, "time", NC_LONG, 1,
  //			   &t_dimid, &time_varid)))
  //  ERR(retval);
  
  printf("lyapunov exponent\n");
  
  if ((retval = nc_def_var(ncid, "lyapunov_exponent", NC_DOUBLE, NDIMS, dimids, &varid)))
    ERR(retval);
  
  if ((retval = nc_put_att_double(ncid, varid, "_FillValue",NC_DOUBLE, 1, &defaultVal)))
    ERR(retval);



     
  /* End define mode. This tells netCDF we are done defining
   * metadata. */
  if ((retval = nc_enddef(ncid)))
    ERR(retval);
  printf("Data\n");
     
  /* Write the pretend data to the file. Although netCDF supports
   * reading and writing subsets of data, in this case we write all
   * the data in one operation. */
  if ((retval = nc_put_var_double(ncid, varid, &tlya[0])))
    ERR(retval);

  if ((retval = nc_put_var_double(ncid, lon_varid, &lon[0])))
    ERR(retval);
  if ((retval = nc_put_var_double(ncid, lat_varid, &lat[0])))
    ERR(retval);
  //if ((retval = nc_put_var_long(ncid, time_varid, &time[0])))
  //  ERR(retval);

  printf("Fermeture\n");
     
  /* Close the file. This frees up any internal netCDF resources
   * associated with the file, and flushes any buffers. */
  if ((retval = nc_close(ncid)))
    ERR(retval);
  
  free(tlya);

}



int main(int argc, char **argv){

	double scale	=4.0;   // revoir les lon/lat si != 4
	double dt	=86400.0;
	double bbox[]= { -80.0, -180.0, 90.0, 180.0 };
	
	
	char *mercator_in;
	double *u,*v, *u_slice, *v_slice;
	size_t	xDim,yDim,tDim;
	
	if(argc != 2){
		printf("usage: %s mercator_in.nc\n",argv[0]);
		printf("mercator_in.nc: ls glob/ext_u_v_0m-mercatorPsy3v1R1v_glo_mean_200711* | ncecat -O mercator_in.nc\n");
		exit(1);
	}
	
	mercator_in=argv[1];
	
	// netcdf read
	read_file (mercator_in, &u, &v, &xDim, &yDim, &tDim);  // u & v will be allocated
	
	
	
	printf("init with the first frame (no tDim)");
	u_slice=u;
	v_slice=v;
	lyapunov_init(xDim, yDim, u_slice, v_slice, dt, scale, bbox);  
	
	// loop over slices
	int t;
        // tDim=1; // DEBUG
	for(t=0;t<tDim;t++){
		printf("t: %d\n",t); 
		u_slice=&u[xDim*yDim*t];
		v_slice=&v[xDim*yDim*t];
		lyapunov_iterate(xDim,yDim,u_slice,v_slice);  // iterate on slice
	}
	
	printf("fin iterate");
	
	// get result
	int subXDim = (xDim - 1)*(int)scale +1;
	int subYDim = (yDim - 1)*(int)scale +1;
	double *ly2;
	ly2=malloc(subXDim*subYDim*sizeof(double));	
	lyapunov_finish(subXDim,subYDim,ly2);
	
	// write result to file
	char *output = "lyapunov_out.nc";
	double *lon = malloc(subXDim*sizeof(double));
	double *lat = malloc(subYDim*sizeof(double));
	long time[1];
	time[0] = 235456;
	int i;
	for (i=0;i<subXDim;i++) {
		lon[i]=-180 + i*(360./((xDim-1)*scale));
	}
	for (i=0;i<subYDim;i++) {
		lat[i]=-80 + i*(170./((yDim-1)*scale));  // mercator data is [-80 90]
	}

	write_file (output, ly2, subXDim,subYDim, lat, lon, time);

	free(u);
	free(v);
	free(lon);
	free(lat);
	free(ly2);
	
	exit(0);
}
