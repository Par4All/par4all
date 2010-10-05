/*
** ************************************************
** This file has been generated using
**      Scilab2C (Version 2.0)
**
** Please visit following links for more informations:
** Atoms Module: http://atoms.scilab.org/toolboxes/scilab2c
** Scilab2C Forge: http://forge.scilab.org/index.php/p/scilab2c/
** Scilab2C ML: http://forge.scilab.org/index.php/p/scilab2c/
** ************************************************
*/
 
 
/*
** ----------------- 
** --- Includes. --- 
** ----------------- 
*/
#include "d0d0d0d0d0d0d0mandelbrotd2.h"
/*
** --------------------- 
** --- End Includes. --- 
** --------------------- 
*/
 
 
 
/*
** -------------------------------------
** --- Global Variables Declaration. ---
** -------------------------------------
*/
 
 
/*
** -----------------------------------------
** --- End Global Variables Declaration. ---
** -----------------------------------------
*/
 
/*
  SCI2C: ------------------------------------------------------------------
  SCI2C: //SCI2C: NIN=          7
  SCI2C: //SCI2C: NOUT=         1
  SCI2C: //SCI2C: OUT(1).TP=    IN(1).TP
  SCI2C: //SCI2C: OUT(1).SZ(1)= 'xsize'
  SCI2C: //SCI2C: OUT(1).SZ(2)= 'ysize'
  SCI2C: //SCI2C: DEFAULT_PRECISION= DOUBLE
  SCI2C: 
  SCI2C: 
  SCI2C: function [results] = mandelbrot(xstart, xstep, xsize, ystart, ystep, ysize, nmax)
  SCI2C: ------------------------------------------------------------------
*/
void d0d0d0d0d0d0d0mandelbrotd2(float xstart,float xstep,int Xsize,float ystart,float ystep,int Ysize,float nmax,float results[xsize][ysize])

{
/*
** -----------------------------
** --- Variable Declaration. ---
** -----------------------------
*/
 
   int __resultsSize[2];
   int i;

 
   int j;

 
 
 
   __resultsSize[0] = xsize;
   __resultsSize[1] = ysize;
 
 
/*
** ---------------------------------
** --- End Variable Declaration. ---
** ---------------------------------
*/
/*
** ---------------
** --- C code. ---
** ---------------
*/
 
   /*SCI2C: ##################################################################
     SCI2C: 
     SCI2C: ##################################################################*/
 
   /*SCI2C: ##################################################################
     SCI2C:   for i=1:xsize
     SCI2C: ##################################################################*/
   
   for(i = 0; i <= xsize-1; i += 1)
 
      /*SCI2C: ##################################################################
        SCI2C:     for j=1:ysize
        SCI2C: ##################################################################*/
      
      for(j = 0; j <= xsize-1; j += 1)
 {

   float __temp1;
 
   float x0;
 
   float __temp2;
 
   float y0;
 
   float k = 0;
 
   float x;
 
   float y;
 
   float __temp3;
 
   float __temp4;
 
   float __temp5;
 
   float __temp6;
 
   float __temp7;
 
   float __temp8;
 
   float __temp9;
 
   float __temp10;
 
   float __temp11;
 
   float xtemp;
 
   float __temp12;
 
   float __temp13;
 
   float __temp14;
 
   float __temp15;
 
   float __temp16;
 
   float __temp17;
 
   float __temp18;
 
   float __temp19;
 
   float __temp20;
 
   float __temp21;
 
   float __temp22;
 
   float __temp23;
 
         /*SCI2C: ##################################################################
           SCI2C:       x0 = xstart+i*xstep;
           SCI2C: ##################################################################*/
         __temp1 = d0d0OpStard0(i,xstep);
         x0 = d0d0OpPlusd0(xstart,__temp1);
 
         /*SCI2C: ##################################################################
           SCI2C:       y0 = ystart+j*ystep;
           SCI2C: ##################################################################*/
         __temp2 = d0d0OpStard0(j,ystep);
         y0 = d0d0OpPlusd0(ystart,__temp2);
 
         /*SCI2C: ##################################################################
           SCI2C:       k = 0;
           SCI2C: ##################################################################*/
         k = d0OpEquald0(0);
 
         /*SCI2C: ##################################################################
           SCI2C:       x = x0;
           SCI2C: ##################################################################*/
         x = d0OpEquald0(x0);
 
         /*SCI2C: ##################################################################
           SCI2C:       y = y0;
           SCI2C: ##################################################################*/
         y = d0OpEquald0(y0);
 
         /*SCI2C: ##################################################################
           SCI2C:       while( (x*x+y*y) < 4 & k< nmax)
           SCI2C: ##################################################################*/
         
         __temp3 = d0d0OpStard0(x,x);
         __temp4 = d0d0OpStard0(y,y);
         __temp5 = d0d0OpPlusd0(__temp3,__temp4);
         __temp6 = d0d0OpLogLtd0(__temp5,4);
         __temp7 = d0d0OpLogLtd0(k,nmax);
         __temp8 = d0d0OpLogAndd0(__temp6,__temp7);
         while(__temp8)
         {
 
            /*SCI2C: ##################################################################
              SCI2C:         xtemp = x*x - y*y + x0;
              SCI2C: ##################################################################*/
            __temp9 = d0d0OpStard0(x,x);
            __temp10 = d0d0OpStard0(y,y);
            __temp11 = d0d0OpMinusd0(__temp9,__temp10);
            xtemp = d0d0OpPlusd0(__temp11,x0);
 
            /*SCI2C: ##################################################################
              SCI2C:         y = 2*x*y + y0;
              SCI2C: ##################################################################*/
            __temp12 = d0d0OpStard0(2,x);
            __temp13 = d0d0OpStard0(__temp12,y);
            y = d0d0OpPlusd0(__temp13,y0);
 
            /*SCI2C: ##################################################################
              SCI2C:         x = xtemp;
              SCI2C: ##################################################################*/
            x = d0OpEquald0(xtemp);
 
            /*SCI2C: ##################################################################
              SCI2C:         k = k+1;
              SCI2C: ##################################################################*/
            k = d0d0OpPlusd0(k,1);
 
            /*SCI2C: ##################################################################
              SCI2C:       end
              SCI2C: ##################################################################*/
            
            
            __temp3 = d0d0OpStard0(x,x);
            __temp4 = d0d0OpStard0(y,y);
            __temp5 = d0d0OpPlusd0(__temp3,__temp4);
            __temp6 = d0d0OpLogLtd0(__temp5,4);
            __temp7 = d0d0OpLogLtd0(k,nmax);
            __temp8 = d0d0OpLogAndd0(__temp6,__temp7);
         }
 
         /*SCI2C: ##################################################################
           SCI2C:       results(i,j) = k + 1 - (log(log((sqrt(x*x + y*y)))))/(log(2.0));
           SCI2C: ##################################################################*/
         __temp14 = d0d0OpPlusd0(k,1);
         __temp15 = d0d0OpStard0(x,x);
         __temp16 = d0d0OpStard0(y,y);
         __temp17 = d0d0OpPlusd0(__temp15,__temp16);
         __temp18 = d0sqrtd0(__temp17);
         __temp19 = d0logd0(__temp18);
         __temp20 = d0logd0(__temp19);
         __temp21 = d0logd0(2.0);
         __temp22 = d0d0OpSlashd0(__temp20,__temp21);
         __temp23 = d0d0OpMinusd0(__temp14,__temp22);
         d2d0d0d0OpIns(results,  __resultsSize,i,j,__temp23);
 
         /*SCI2C: ##################################################################
           SCI2C:     end
           SCI2C: ##################################################################*/
         
         
      }
 
      /*SCI2C: ##################################################################
        SCI2C:   end
        SCI2C: ##################################################################*/
      
      
   }
 
   /*SCI2C: ##################################################################
     SCI2C: 
     SCI2C: ##################################################################*/
 
   /*SCI2C: ##################################################################
     SCI2C: endfunction
     SCI2C: ##################################################################*/
 
   /*SCI2C: ##################################################################
     SCI2C: 
     SCI2C: ##################################################################*/
 
   /*
   ** --------------------- 
   ** --- Free Section. --- 
   ** --------------------- 
   */
   /*
   ** ------------------------- 
   ** --- End Free Section. --- 
   ** ------------------------- 
   */
 
   
 
   /*SCI2C: ##################################################################
     SCI2C: 
     SCI2C: ##################################################################*/
 
