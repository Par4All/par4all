#ifndef UPSALE_H
#define UPSCALE_H


#define clip(a) (((a)<0) ? 0 : (((a)>255) ? 255 : (a)))
#define clip0(a) ((a)<0 ? 0 : (a))
#define clipMax(a,b) ((a)>=(b) ? ((b)-1) :(a))

void upscale(type_yuv_frame_in *frame_in,type_yuv_frame_out *frame_out);

#endif 
