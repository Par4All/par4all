#ifndef YUV_H
#define YUV_H

#define WIDTH 400
#define HEIGHT 226
#define OFFSET 2

#define SIZE WIDTH*HEIGHT

#define W_Y_IN (WIDTH+OFFSET*2)
#define H_Y_IN (HEIGHT+OFFSET*2)
#define W_UV_IN (WIDTH/2)
#define H_UV_IN (HEIGHT/2)
#define SIZE_UV_IN ((W_UV_IN)*(H_UV_IN))
#define SIZE_Y_IN ((W_Y_IN)*(H_Y_IN))

#define W_Y_OUT (WIDTH*2)
#define H_Y_OUT (HEIGHT*2)
#define W_UV_OUT ((W_Y_OUT)/2)
#define H_UV_OUT ((H_Y_OUT)/2)
#define SIZE_UV_OUT ((W_UV_OUT)*(H_UV_OUT))
#define SIZE_Y_OUT ((W_Y_OUT)*(H_Y_OUT))



typedef unsigned char uint8;
typedef struct type_yuv_frame_in type_yuv_frame_in;
typedef struct type_yuv_frame_out type_yuv_frame_out;


struct type_yuv_frame_in {
	uint8 y[SIZE_Y_IN];
	uint8 u[SIZE_UV_IN];
	uint8 v[SIZE_UV_IN];
};

struct type_yuv_frame_out {
	uint8 y[SIZE_Y_OUT];
	uint8 u[SIZE_UV_OUT];
	uint8 v[SIZE_UV_OUT];
};

int read_yuv_frame(FILE* fp,type_yuv_frame_in *frame);
int write_yuv_frame(FILE* fp,type_yuv_frame_out *frame);



#endif //YUV_H
