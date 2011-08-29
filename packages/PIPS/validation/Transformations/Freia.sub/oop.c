// $Id$ 

#include "freia.h"
#include "geometrie.h"

/*
 oop.c 
<B>Out of position detection example</B>
taken from project PICS

Author: Christophe Clienti
port to freia  : Michel Bilodeau

(C) Armines 2007

*/

//====================================================

// nicer graph display
#define img0 i0
#define img1 i1
#define img2 i2
#define img3 i3
#define imgMask im
#define imgtt0 t0
#define imgtt1 t1
#define imgtt2 t2
#define imgtt3 t3
#define imgtt4 t4

int main (int argc, char * argv[])
{
  freia_dataio fdin;
  freia_dataio fdout;
  freia_dataio fdout1;
 
  freia_data2d *img0;
  freia_data2d *img1 ;
  freia_data2d *img2 ;
  freia_data2d *img3 ;
  freia_data2d *imgMask;
   
  freia_data2d *imgtt0;  // five last images
  freia_data2d *imgtt1 ;
  freia_data2d *imgtt2 ;
  freia_data2d *imgtt3 ;
  freia_data2d *imgtt4;

  freia_data2d *imgtmp;
  
  freia_data2d *imgg1 ;
  freia_data2d *imgg2 ;
  freia_data2d *imgsav;

  unsigned nbpix;
  int32_t x,y;
  int32_t valx, valy, valmax, valx_pre=0;

  // int32_t startx=10,starty=231, endx=183,endy=142; // hard-coded calibration
  //int32_t startx=10,starty=250, endx=152,endy=132; // hard-coded calibration
  int32_t startx=33,starty=272, endx=152,endy=132; // hard-coded calibration
  int32_t countfalsedetect=0;
  //  int32_t epaisseur=17;
  int32_t epaisseur=17;
  int32_t mx1,my1, mx2,my2, idx=10;
   
  int32_t dx;
  uint32_t *tabeqt;
  freia_status end = FREIA_OK;

  freia_initialize(argc, argv);

  // Input/output stream and image creations
  freia_common_open_input(&fdin, 0); 

  img0 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
#ifdef DEBUG
  printf("framewidth = %d frameheight %d framebpp %d\n", fdin.framewidth, fdin.frameheight, fdin.framebpp);
#endif // DEBUG

  freia_common_open_output(&fdout, 0, fdin.framewidth, fdin.frameheight, 8); 
  freia_common_open_output(&fdout1, 1, fdin.framewidth, fdin.frameheight, 8); 
  img1 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
  img2 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
  img3 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
  imgMask = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);

  imgtt0 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
  // printf("width = %d height %d bpp %d\n", img0->width, img0->height,img0->bpp);
  imgtt1 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
  imgtt2 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
  imgtt3 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
  imgtt4 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
 
  imgg1 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
  imgg2 = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
 
  imgtmp = freia_common_create_data(fdin.framebpp, fdin.framewidth, fdin.frameheight);
 
  tabeqt = freia_common_alloc(fdin.framewidth * sizeof(*tabeqt));
  x = fdin.framewidth;
  y = fdin.frameheight;
  dx=idx;
  
  nbpix=y*x;


  ligneInitTabEqt(tabeqt,fdin.framewidth,startx,starty,endx,endy,255);



  mx1=startx;
  my1=starty;
  mx2=endx;
  my2=endy;
  valx = mx2;
  end = freia_common_rx_image(img0, &fdin);

  while(end == FREIA_OK) {  

    // for debug freia_aipo_set_constant(imgMask, 0);
    // for debug ligneEpaisse(imgMask, mx1,my1, mx2,my2,epaisseur,255);

    // ??? POINTER SHUFFLING is not supported by pips!
    // It leads to wrong results...
    // ??? should use freia_copy instead...
    // hmmm... only every two images is used by the computation?
      imgsav=imgtt4;
      imgtt4=imgtt3;
      imgtt3=imgtt2;
      imgtt2=imgtt1;
      imgtt1=imgtt0;
      imgtt0=imgsav;
/*       imgsav=imgtt2; */
/*       imgtt2=imgtt1; */
/*       imgtt1=imgtt0; */
/*       imgtt0=imgsav; */

      // freia_auto_median(img1, img0, 8);
      freia_aipo_copy(img1, img0);
      freia_aipo_copy(imgtt0, img1);

      freia_cipo_gradient(imgtmp, imgtt0, 8, 1);
      freia_cipo_gradient(img1, imgtt2, 8, 1);
      freia_aipo_absdiff(img1, imgtmp, img1);

      freia_cipo_open(img1, img1, 8, 1);

      freia_cipo_gradient(imgtmp, imgtt2, 8, 1);
      freia_cipo_gradient(img2, imgtt4, 8, 1);
      freia_aipo_absdiff(img2, imgtmp, img2);
      
      freia_cipo_open(img2, img2, 8, 1);

      freia_aipo_inf(img3, img2, img1);
       
      freia_aipo_threshold(img3, img3, 15, 255, true);
       // freia_aipo_inf(img3, img3, imgMask);
	//	freia_aipo_sup(imgMask, img0, imgMask);
       //	 freia_common_tx_image(img3, &fdout);

      valx_pre=valx;
      valmax=getMaxInLigneEpaisse(img3, tabeqt, mx1,my1,mx2,my2,epaisseur, &valx, &valy);
#ifdef DEBUG
      printf("Valx %d Valy %d mx1 %d  mx2 %d valmax %d valx_pre %d false %d\n", valx, valy, mx1, mx2, valmax, valx_pre, countfalsedetect);
//#define img0 img3
#endif // DEBUG

// ??? freia compiler work around "illegal" pointer shuffle:
// show output on imgtt0
#undef img0
#define img0 imgtt0
     
      // barre(img0, valx, 0, 2,x, 255);  
      //	countfalsedetect++;

      dx=10;
      if(valmax==0) {
	countfalsedetect++;
	valx=valx_pre;
	if(countfalsedetect>45) {
	  countfalsedetect=1000;
	  //valx=(valx_pre+5)%mx2;
	  valx += 5;
	  if(valx>mx2) valx= mx2;
	  if(valx<mx1) valx=mx1;
	  dx=10;
	}

      }else {
	countfalsedetect=0;
	dx=20;
      }
      barre(img0, valx, 0, 2,x, 255);  
   
      mx1=valx-dx;
      if(mx1<startx) mx1=startx;
      my1=tabeqt[mx1];
	 
      mx2=valx+dx;
      if(mx2>endx) mx2=endx;
      my2=tabeqt[mx2];
	 
      valx_pre=endx-valx;
      
      freia_common_tx_image(img0, &fdout);
  
#undef img0
#define img0 i0
//#ifdef DEBUG
//#undef img0
//#endif // DEBUG

      end = freia_common_rx_image(img0, &fdin);
  }
 
      /* images destruction */
  freia_common_destruct_data(img0);
  freia_common_destruct_data(img1);
  freia_common_destruct_data(img2);
  freia_common_destruct_data(img3);
  freia_common_destruct_data(imgMask);
  freia_common_destruct_data(imgtt0);
  freia_common_destruct_data(imgtt1);
  freia_common_destruct_data(imgtt2);
  freia_common_destruct_data(imgtt3);
  freia_common_destruct_data(imgtt4);
  freia_common_destruct_data(imgg1);
  freia_common_destruct_data(imgg2);
  
  freia_common_destruct_data(imgtmp);

  /* close videos flow */
  freia_common_close_input(&fdin); 
  freia_common_close_output(&fdout); 

  freia_shutdown();
  return 0;
}

