/* Controlizer bug after unfolding in an application of the FREIA
   project. */

typedef int int32_t;
typedef int uint32_t;
typedef int freia_status;
typedef struct {int framewidth;int frameheight;int frameindex;} freia_dataio;
typedef struct {int width; int height; int * bpp; int widthWa; int heightWa;}  freia_data2d;

#define FREIA_OK (1)
#define FREIA_SIZE_ERROR (2)

int main(void)
{
  freia_dataio fdstab;

  freia_data2d *imcurrent;
  freia_data2d *imstab;
  freia_data2d *imehn;

  freia_data2d *immotion;

  int32_t maxmotion;
  int32_t minmotion;
  int32_t binvalue = 128;

  int32_t xshiftacc, yshiftacc;

  freia_status end = FREIA_OK;

  /* Param set by user */
  _Bool stabilize = 1;
  _Bool contrastenhance = 1;
  _Bool motiondetect = 1;

  uint32_t horizon = 10;
  int32_t maximal_shape = 21;
  int32_t motion_th = 30;
  int32_t motion_trig = 75;
  //PIPS generated variable
  int _return0, _return1, _return2, _return3;

 l99995:;

  if (!(end==FREIA_OK)) goto break_1;

  if (end!=FREIA_OK) goto break_1;

  if (stabilize==1) {
    freia_common_tx_image(imstab, &fdstab);
  }
  else {
    freia_common_set_wa(imcurrent, xshiftacc+horizon, yshiftacc+horizon, 288-2*horizon, 224-2*horizon);
  }

  if (contrastenhance==1) {
    {
      int I_0;
      I_0 = maximal_shape/2+1;
      {
	freia_status ret;

	_return2 = ret;
	goto l99997;
      l99997:       ;
      }
    }
    _return2;
    {
      freia_status ret = FREIA_OK;
      int32_t volcurrent;
      int32_t volprevious;
      if (1!=1) {
	;
	while (0)
	  ;
	_return3 = FREIA_SIZE_ERROR;
	goto l99996;
      }
      else
	if (1!=1) {
	  ;
	  while (0)
	    ;
	  _return3 = FREIA_SIZE_ERROR;
	  goto l99996;
	}
	else
	  ret = freia_aipo_global_vol(imehn, &volcurrent);
      do {
	ret |= freia_aipo_global_vol(imehn, &volcurrent);
      }
      while (volcurrent!=volprevious);


      _return3 = ret;
      goto l99996;
    l99996: ;
    }
    _return3;
  }
  else
  if (motiondetect==1) {
    freia_aipo_global_min(immotion, &minmotion);
    if (maxmotion-minmotion>motion_trig)
      binvalue = maxmotion*motion_th/100;
    {
      freia_status ret;

      _return1 = ret;
      goto l99998;
    l99998:       ;
    }
    _return1;
    {
      freia_data2d *imtmp;
      if (1!=1) {
	;
	while (0)
	  ;
	_return0 = FREIA_SIZE_ERROR;
	goto l99999;
      }
      else
	if (1!=1) {
	  ;
	  while (0)
	    ;
	  _return0 = FREIA_SIZE_ERROR;
	  goto l99999;
	}
	else
	  imtmp = freia_common_create_data(immotion->bpp, immotion->widthWa, immotion->heightWa);
      _return0 = FREIA_OK;
      goto l99999;
    l99999:         ;
    }
    _return0;
  }
  else
    ;
  goto l99995;
 break_1:   ;

  return 0;
}
