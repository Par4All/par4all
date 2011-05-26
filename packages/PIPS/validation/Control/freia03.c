/* Controlizer bug after unfolding in an application of the FREIA
   project.

   The code is meaningless because controlizer deals with the CFG and
   not with the executable parts of the source. Functions calls and
   useless declarations have been removed.

   Simplified version of freia02.c, still exhibiting code losses
 */

typedef int int32_t;
typedef int uint32_t;
typedef int freia_status;
typedef struct {int framewidth;int frameheight;int frameindex;} freia_dataio;
typedef struct {int width; int height; int * bpp; int widthWa; int heightWa;}  freia_data2d;

#define FREIA_OK (1)
#define FREIA_SIZE_ERROR (2)

int main(void)
{
  freia_status end = FREIA_OK;
  _Bool contrastenhance = 1;
  int32_t maximal_shape = 21;
  int _return2, _return3;

 l99995:;
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
  }
 break_1:   ;

  return 0;

}
