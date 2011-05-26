#include <freiaDebug.h>
#include <freiaCommon.h>
#include <freiaAtomicOp.h>
#include <freiaComplexOp.h>

// for debug, use -DDEBUG
// for instance with "make CPPOPT=-DDEBUG ..."

freia_status freia_open_8c(freia_data2d *imout, freia_data2d *imin, const int32_t *kernel, int repetition) {
  freia_data2d *imtmp;
  freia_status ret;
  int i;

  imtmp = freia_common_create_data(imout->bpp, imout->widthWa, imout->heightWa);

  
  ret = freia_aipo_copy(imtmp, imin);

  for(i=0 ; i<repetition ; i++) {
    ret |= freia_aipo_erode_8c(imout, imtmp, kernel);
    ret |= freia_aipo_copy(imtmp, imout);
  }

  for(i=0 ; i<repetition ; i++) {
    ret |= freia_aipo_dilate_8c(imout, imtmp, kernel);
    ret |= freia_aipo_copy(imtmp, imout);
  }

  freia_common_destruct_data(imtmp);

  return ret;
}


freia_status freia_close_8c(freia_data2d *imout, freia_data2d *imin, const int32_t *kernel, int repetition) {
  freia_data2d *imtmp;
  freia_status ret;
  int i;

  imtmp = freia_common_create_data(imout->bpp, imout->widthWa, imout->heightWa);

  ret = freia_aipo_copy(imtmp, imin);

  for(i=0 ; i<repetition ; i++) {
    ret |= freia_aipo_dilate_8c(imout, imtmp, kernel);
    ret |= freia_aipo_copy(imtmp, imout);
  }

  for(i=0 ; i<repetition ; i++) {
    ret |= freia_aipo_erode_8c(imout, imtmp, kernel);
    ret |= freia_aipo_copy(imtmp, imout);
  }

  freia_common_destruct_data(imtmp);

  return ret;
}


int main(void)
{
  freia_dataio fdin;
  freia_dataio fdout;

  freia_data2d *imin;
  freia_data2d *immir;
  freia_data2d *imopen;
  freia_data2d *imclose;
  freia_data2d *imopenth;
  freia_data2d *imcloseth;
  freia_data2d *imand;
  freia_data2d *imfilt;
  freia_data2d *imout;
  freia_data2d *imoutrepl;

  const int32_t kernel1x3[9] =	{ 0,0,0,
				  1,1,1,
			  	  0,0,0 };
  
  const int32_t kernel3x1[9] = { 0,1,0,
			  	 0,1,0,
			  	 0,1,0 };
 
 
  /* open videos flow */
  freia_common_open_input(&fdin, 0); 
  freia_common_open_output(&fdout, 0, fdin.framewidth, fdin.frameheight, 8);

  /* images creation */
  imin = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
  immir = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
  imopen = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
  imclose = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
  imopenth = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
  imcloseth = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
  imand = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
  imfilt = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
  imout = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);
  imoutrepl = freia_common_create_data(16, fdin.framewidth, fdin.frameheight);

  /* processing */
  freia_common_rx_image(imin, &fdin);

  freia_open_8c(imopen, imin, kernel1x3, 15);
  freia_close_8c(imclose, imin, kernel1x3, 8);

  freia_aipo_threshold(imopenth,imopen, 1, 50, true);
  freia_aipo_threshold(imcloseth,imclose, 150, 255, true);

  freia_aipo_and(imand, imopenth, imcloseth);

  freia_open_8c(imfilt, imand, kernel3x1, 4);
  freia_open_8c(imout, imfilt, kernel1x3, 4);

  freia_cipo_dilate(imoutrepl, imout, 8, 3);

  freia_aipo_and(imoutrepl, imoutrepl, imin);

#ifdef DEBUG
  freia_common_tx_image(imopen, &fdout);//0
  freia_common_tx_image(imclose, &fdout);//1
  freia_common_tx_image(imopenth, &fdout);//2
  freia_common_tx_image(imcloseth, &fdout);//3
  freia_common_tx_image(imand, &fdout);//5
  freia_common_tx_image(imout, &fdout);//6
#endif /* DEBUG */
  freia_common_tx_image(imoutrepl, &fdout);//7

  /* images destruction */
  freia_common_destruct_data(imin);
  freia_common_destruct_data(immir);
  freia_common_destruct_data(imopen);
  freia_common_destruct_data(imclose);
  freia_common_destruct_data(imopenth);
  freia_common_destruct_data(imcloseth);
  freia_common_destruct_data(imand);
  freia_common_destruct_data(imfilt);
  freia_common_destruct_data(imout);
  freia_common_destruct_data(imoutrepl);

  /* close videos flow */
  freia_common_close_input(&fdin); 
  freia_common_close_output(&fdout); 

  return 0;
}
