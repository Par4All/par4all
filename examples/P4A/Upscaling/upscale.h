#ifndef UPSALE_H
#define UPSCALE_H

#define clip(a) (((a)<0) ? 0 : (((a)>255) ? 255 : (a)))
#define clip0(a) ((a)<0 ? 0 : (a))
#define clipMax(a,b) ((a)>=(b) ? ((b)-1) :(a))

void upscale(uint8 y_in[SIZE_Y_IN], uint8 u_in[SIZE_UV_IN], uint8 v_in[SIZE_UV_IN],uint8 y_out[SIZE_Y_OUT], uint8 u_out[SIZE_UV_OUT], uint8 v_out[SIZE_UV_OUT]);

#endif 
