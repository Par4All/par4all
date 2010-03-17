
typedef unsigned char uint8;
typedef unsigned short uint16;
typedef struct type_yuv_frame type_yuv_frame;


struct type_yuv_frame {
    int sizew;
    int sizeh;
    int sizeChrW;
    int sizeChrH;
    uint8 *u;
    uint8 *v;
    short *y;
};

#define clip(a) (((a)<0) ? 0 : (((a)>255) ? 255 : (a)))
#define clip0(a) ((a)<0 ? 0 : (a))
#define clipMax(a,b) ((a)>=(b) ? ((b)-1) :(a))


/* fonction d'upcaling au demi pixel */
/* frame_in: frame initiale */
/* frame_out: frame upscalée */
double t1=0,t2=0,t3=0;
const short normalisation=5;
const short bias=16;
#define OFFSET 4


void upscale_chrominance(type_yuv_frame *frame_in,type_yuv_frame *frame_out)
{
    int i,j;
    for(i=0;i<frame_in->sizeChrW;i++){
        for(j=0;j<frame_in->sizeChrH;j++){
            frame_out->u[frame_out->sizeChrW*(2*j)+(2*i)]=	frame_in->u[frame_in->sizeChrW*j+i];
            frame_out->u[frame_out->sizeChrW*(2*j+1)+(2*i)]=	frame_in->u[frame_in->sizeChrW*j+i];
            frame_out->u[frame_out->sizeChrW*(2*j)+(2*i+1)]=	frame_in->u[frame_in->sizeChrW*j+i];
            frame_out->u[frame_out->sizeChrW*(2*j+1)+(2*i+1)]=frame_in->u[frame_in->sizeChrW*j+i];
            frame_out->v[frame_out->sizeChrW*(2*j)+(2*i)]=	frame_in->v[frame_in->sizeChrW*j+i];
            frame_out->v[frame_out->sizeChrW*(2*j+1)+(2*i)]=	frame_in->v[frame_in->sizeChrW*j+i];
            frame_out->v[frame_out->sizeChrW*(2*j)+(2*i+1)]=	frame_in->v[frame_in->sizeChrW*j+i];
            frame_out->v[frame_out->sizeChrW*(2*j+1)+(2*i+1)]=frame_in->v[frame_in->sizeChrW*j+i];
        }
    }
}

void upscale_luminance(type_yuv_frame *frame_in,type_yuv_frame *frame_out)
{
    int h=frame_in->sizeh;
    int w=frame_in->sizew;
    int W=frame_out->sizew;
    int y,x;
    //const short normalisation=5;
    //const short bias=16;

    for(y=0;y<2*h;y=y+2){
        for(x=0;x<2*w;x=x+2){
            int xdiv2=x/2;
            int ydiv2=y/2;
            int wydiv2=w*ydiv2;
            // recopie des points (x,y)
            frame_out->y[W*y+x]=frame_in->y[w*y/2+xdiv2];
            // calcul des point (x+1,y)
            /*if (x/2>w-3)
              printf("Signaler que x/2>2*w-3\n");
              if (xdiv2<2)
              printf("\tSignaler que xdiv2<2\n");
              */
            if (xdiv2<2 || x/2>w-4)  // pour minimiser le nombre de clipping
                frame_out->y[W*y+x+1]=
                    frame_in->y[clip0((xdiv2)-2)+wydiv2] -5*frame_in->y[clip0((xdiv2)-1)+wydiv2] + 20*frame_in->y[xdiv2+wydiv2]+
                    20*frame_in->y[clipMax((xdiv2)+1,w)+wydiv2] -5*frame_in->y[clipMax((xdiv2)+2,w)+wydiv2] + frame_in->y[clipMax((xdiv2)+3,w)+wydiv2];
            else
                frame_out->y[W*y+x+1]=
                    frame_in->y[((xdiv2)-2)+wydiv2] -5*frame_in->y[((xdiv2)-1)+wydiv2] + 20*frame_in->y[xdiv2+wydiv2]+
                    20*frame_in->y[(xdiv2)+1+wydiv2] -5*frame_in->y[(xdiv2)+2+wydiv2] + frame_in->y[(xdiv2)+3+wydiv2];
            frame_out->y[W*y+x+1]=clip((frame_out->y[W*y+x+1]+bias)>>normalisation);
            // calcul des points (x,y+1)
            /*if (ydiv2<2)
              printf("\tSignaler que ydiv2<2\n");
              if (ydiv2>h-3)
              printf("Signaler que ydiv2>2*h-3\n");
              */
            if (y/2<2 || ydiv2>h-4)
                frame_out->y[x+(y+1)*W]=
                    frame_in->y[xdiv2+w*clip0((ydiv2)-2)] -5*frame_in->y[xdiv2+w*clip0((ydiv2)-1)] +20*frame_in->y[xdiv2+wydiv2] 
                    +20*frame_in->y[xdiv2+w*clipMax((ydiv2)+1,h)] -5*frame_in->y[xdiv2+w*clipMax((ydiv2)+2,h)]+ frame_in->y[xdiv2+w*clipMax((ydiv2)+3,h)];
            else
                frame_out->y[x+(y+1)*W]=
                    frame_in->y[xdiv2+wydiv2-w*2] -5*frame_in->y[xdiv2+wydiv2-w] +20*frame_in->y[xdiv2+wydiv2] 
                    +20*frame_in->y[xdiv2+wydiv2+w] -5*frame_in->y[xdiv2+wydiv2+2*w]+ frame_in->y[xdiv2+wydiv2+3*w];
            frame_out->y[x+(y+1)*W]=clip((frame_out->y[x+(y+1)*W]+bias)>>normalisation);

        }
    }


    // calcul des points (x+1, y+1)
    for(y=0;y<2*h;y=y+2){
        for(x=0;x<2*w;x=x+2){
            if(x<4 ||x>2*w-8)
                frame_out->y[W*(y+1)+x+1]=clip((
                            frame_out->y[clip0(x-4)+W*(y+1)]-5*frame_out->y[clip0(x-2)+W*(y+1)]+20*frame_out->y[x+W*(y+1)]
                            +20*frame_out->y[clipMax(x+2,W)+W*(y+1)]-5*frame_out->y[clipMax(x+4,W)+W*(y+1)]+frame_out->y[clipMax(x+6,W)+W*(y+1)]+bias)>>normalisation); 
            else
                frame_out->y[W*(y+1)+x+1]=clip((
                            frame_out->y[(x-4)+W*(y+1)]-5*frame_out->y[(x-2)+W*(y+1)]+20*frame_out->y[x+W*(y+1)]
                            +20*frame_out->y[x+2+W*(y+1)]-5*frame_out->y[x+4+W*(y+1)]+frame_out->y[x+6+W*(y+1)]+bias)>>normalisation); 
        }
    }
    // le clipping est réalisé dans un second temps pour éviter des erreurs d'arrondis dans le calcul précédent
    // le gain en qualité ne me semble pas évident
    // si décommenté il faut prévoir une normalisation + 512 >> 10 pour le calcul du point x+1,y+1
    /*	for(y=0;y<2*h;y=y+2){
        for(x=0;x<2*w;x=x+2){
        frame_out->y[x+(y+1)*W]=clip((frame_out->y[x+(y+1)*W]+bias)>>normalisation);
        }
        }
        */

}
