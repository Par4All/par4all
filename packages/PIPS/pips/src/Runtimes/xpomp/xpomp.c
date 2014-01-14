/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

*/
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
/* xPOMP --- Run time use to display arrays graphically in PIPS/HPFC.

   It is mainly the Nicolas Paris pompx code from the POMP/POMPC project.

   Ronan.Keryell@cri.ensmp.fr

   */

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <stdlib.h>
/* Some silly AIX stuff: */
#if defined(_AIX)
#define _NO_PROTO
/* signal definition is wrong: void (*signal(int, void (*)(int)))(int) */
#else
#include <alloca.h>
#endif

#include <signal.h>

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xos.h>
#include "rasterfile.h"
#include "gr.h"

#define BITMAPDEPTH 8
#define SMALL 1
#define OK 0

enum {
    palette_Y_size = 10
};

Display *display;
int is_X = 1;
int screen;
int shmid;
int small;
int dbx;
char *bm=0;
#define X_MAX_SIZE 1152
#define Y_MAX_SIZE 900
struct sh_header *sh_h;
int W_X,W_Y;
Window win;
GC gc,gcdraw,gccol;
XImage *xi=0;
int mouse;
int is_mouse;
int x_mouse,y_mouse,b_mouse;

/* To store the button mask before a mouse button event: */
int state_mouse;

int depth;
int clipvalue;
int startvalue;
int repvalue;
int pal;
XSetWindowAttributes xswa;
Colormap cmap;
XFontStruct *font_info;
char *save_data;

/* Use to save the current window for redisplay purpose: */
Pixmap save_pixmap;

char **Argv;
int is_usr2 = 0;
int want_to_wait;
char *convert_byte_to_pixel();
#ifdef vax
#define SWAI(X,Y) ((char *) &X)[0] = ((char *) &Y)[3];\
	((char *) &X)[1] = ((char *) &Y)[2];\
	((char *) &X)[2] = ((char *) &Y)[1];\
	((char *) &X)[3] = ((char *) &Y)[0];
#else
#define SWAI(X,Y) X = Y;
#endif

/*extern*/ char *pc_version;
#ifdef mips
char *pc_version = "????";
#endif

int is_X;

#define BITMAPDEPTH 8
#define SMALL 1
#define OK 0
int screen;
int shmid = 0;
int small;
int dbx = 0;
#define X_MAX_SIZE 1152
#define Y_MAX_SIZE 900
struct sh_header *sh_h = 0;
int W_X=512,W_Y=512;
int mouse = 0;
int is_mouse = 0;
int x_mouse,y_mouse,b_mouse;
int depth;
int clipvalue = -1;
int startvalue = 0;
int repvalue = 0;
int pal = 0;
u_char red[256];
u_char green[256];
u_char blue[256];
char **Argv;
int want_to_wait = 1;
char *nb_screen = 0;
char *nb_comp_screen = 0;
int nb_x = 0;
int nb_y = 0;
void *nb_image = 0;
int nb_map[256];

char *data;
char *databis;
int bytesperline;
unsigned int width, height;


XImage *myCreateImage(char *pdata,
		      int ZX,
		      int ZY);
static void analyse_event(XEvent * report);

static void
close_wm()
{
	XUnloadFont(display, font_info->fid);
	XFreeGC(display, gc);
	XCloseDisplay(display);
}


void quit_proc()
{
	if(sh_h) {
		kill(getppid(),SIGINT);
		shmctl(shmid,IPC_RMID,0);
		shmid = 0;
		sh_h = 0;
	}
	close_wm();
}


static int
wave(n)
{
	int p;
	p = n & 0xfff;
	if(p < 1365) return 64;
	if(p < 2048) return ((3*((((p - 1365) * 6) >> 4) & 0xff))>>2)+64;
	if(p < 3413) return 255;
	return ((3*((((4095 - p) * 6) >> 4) & 0xff))>>2) + 64;
}


static void
set_colormap(red,green,blue)
char *red,*green,*blue;
{
	XColor colorcell[256];
	int i;

	for(i=0;i<256;i++) {
		colorcell[i].pixel = i;
		colorcell[i].red = red[i] << 8;
		colorcell[i].green = green[i] << 8;
		colorcell[i].blue = blue[i] << 8;
		colorcell[i].flags = DoRed | DoGreen | DoBlue;
	}
	XStoreColors(display,cmap,colorcell,256);
}


static void
update_panel(pal,repvalue,startvalue,clipvalue)
int pal,repvalue,startvalue,clipvalue;
{
}


static int
slow_byte2bit(c)
int c;
{
	int b;
	c += startvalue;
	c &= 0xff;
	b = (c >> 7) & 1;
	switch(pal) {
	case 0 :
	case 1 :
		b = (c >> (7 - repvalue)) & 1;
		break;
	case 2 :
		b = (c ^ (c >> (7 - repvalue))) & 1;
		break;
	case 3 :
		break;
	}
	if(clipvalue >= 0) {
		b ^= (clipvalue == c);
	}
	return b & 1;
}


int
byte2bit(c)
int c;
{
	return nb_map[c & 255];
}


static void
set_cmap(pal1,repvalue1,startvalue1,clipvalue1)
{
	int i,j,k,mask;
	double x;
	u_char r[256];
	u_char g[256];
	u_char b[256];

	pal = pal1;
	repvalue = repvalue1;
	startvalue = startvalue1;
	clipvalue = clipvalue1;

	switch(pal) {
	case 0 :
		for ( i = 0 ; i < 256 ; i++ ) {
			j = i;
			if(repvalue < 0) j >>= -repvalue;
			else j <<= repvalue;
			j += startvalue;
			j &= 0x1ff;
			r[i] = b[i] = g[i] = j;
		}
		if(clipvalue >=0 && clipvalue < 256) {
			r[clipvalue] = 255;
			g[clipvalue] = b[clipvalue] = 0;
		}
		break;
	case 1 :
	case 2 :
		for ( i = 0 ; i < 256 ; i++ ) {
			j = i;
			j = j << 4;
			if(repvalue < 0) j >>= -repvalue;
			else j <<= repvalue;
			j += startvalue << 4;
			if(pal == 1 || i & 1) {
				r[i] = wave(j+2731);
				b[i] = wave(j);
				g[i] = wave(j+1365);
			} else {
				r[i] = (3 * wave(j+2731)) >> 2;
				b[i] = (3 * wave(j)) >> 2;
				g[i] = (3 * wave(j+1365)) >> 2;
			}
		}
		if(clipvalue >=0 && clipvalue < 256) {
			r[clipvalue] = g[clipvalue] = b[clipvalue] = 0;
		}
		break;
	case 3 :
		{
			int P1r,P2r,P3r;
			int P1g,P2g,P3g;
			int P1b,P2b,P3b;
			double aa,bb,cc;
			double re,gr,bl;

			P1r = 80;
			P1g = 80;
			P1b = 220;
			P2r = 220;
			P2g = 220;
			P2b = 160;
			P3r = 220;
			P3g = 80;
			P3b = 80;
			P2r = P2r * 2 - ((P1r + P3r)>>1);
			P2g = P2g * 2 - ((P1g + P3g)>>1);
			P2b = P2b * 2 - ((P1b + P3b)>>1);
			for(i=0;i<256;i++) {
				j = (i - startvalue) & 0xff;
				j -= 128;
				if(repvalue < 0) j >>= -repvalue;
				else j <<= repvalue;
				x = (j + 128.0)/256.0;
				if(x < 0.0) x = 0.0;
				if(x > 1.0) x = 1.0;
				aa = 1.0 - x;
				aa = aa * aa;
				bb = 2.0 * x * (1.0 - x);
				cc = x * x;
				re = aa * P1r + bb * P2r + cc * P3r;
				gr = aa * P1g + bb * P2g + cc * P3g;
				bl = aa * P1b + bb * P2b + cc * P3b;
				if(re < 0.0) re = 0.0;
				if(re > 255.0) re = 255.0;
				if(bl < 0.0) bl = 0.0;
				if(bl > 255.0) bl = 255.0;
				if(gr < 0.0) gr = 0.0;
				if(gr > 255.0) gr = 255.0;
				b[i] = bl;
				r[i] = re;
				g[i] = gr;
			}
		}
		if(clipvalue >=0 && clipvalue < 256) {
			r[clipvalue] = g[clipvalue] = b[clipvalue] = 0;
		}
		break;
	case 4 :
		j=0;
		mask = ~((-1) << -repvalue);
		for ( i = 0 ; i < 256 ; i++ ) {
			k = (j + startvalue) & 255;
			r[i] = red[k];
			b[i] = blue[k];
			g[i] = green[k];
			if(repvalue >= 0) {
				j = (j + (1 << repvalue)) & 255;
			} else {
				if((i & mask) == mask) j++;
			}
		}
		if(clipvalue >=0 && clipvalue < 256) {
			r[clipvalue] = -(r[clipvalue] < 127);
			g[clipvalue] = -(g[clipvalue] < 127);
			b[clipvalue] = -(b[clipvalue] < 127);
		}
		break;
	}
	if(depth == 8) {
		set_colormap(r,g,b);
		update_panel(pal1,repvalue1,startvalue1,clipvalue1);
	} else {
		switch(pal) {
		case 0:
		case 1:
		case 2:
		case 3:
			for(i=0;i<256;i++) {
				nb_map[i] = slow_byte2bit(i);
			}
			break;
		default :
			for(i=0;i<256;i++) {
				nb_map[i] = (r[i] + g[i] + b[i] > 384);
			}
			break;
		}
	}
}


void
myPutImage(XImage * p,
	   int sx,
	   int sy,
	   int ZX,
	   int ZY)
{
    /* + palette_Y_size for the palette room: */
	XPutImage(display,save_pixmap,gc,p,0,0,sx,sy + palette_Y_size,ZX,ZY);
	XPutImage(display,win,gc,p,0,0,sx,sy + palette_Y_size,ZX,ZY);
}


static void
refresh_nb()
{
	int i,j;
	char *source,*dest;
	int pix,n;

	if(!nb_screen) return;
	source = nb_screen;
	dest = nb_comp_screen;
	for(j=0;j<nb_y;j++) {
		pix = 0;
		n = 0;
		for(i=0;i<nb_x;i++) {
			pix <<= 1;
			pix |= byte2bit(*source++);
			if(++n == 8) {
				*dest++ = ~pix;
				n = 0;
				pix = 0;
			}
		}
		if(n) {
			pix <<= 8 - n;
			*dest++ = ~pix;
		}
	}
	nb_image = myCreateImage(nb_comp_screen, nb_x,nb_y);
	myPutImage(nb_image,0,0,nb_x,nb_y);
}


/* The stuff that sends an image on the screen: */
static void
image()
{
    char * data_array;
    int X_data_array_size;
    int Y_data_array_size;
    int X_offset;
    int Y_offset;
    int X_zoom_ratio;
    int Y_zoom_ratio;
    char * pdata, * im;
    XImage * p;

    X_data_array_size = sh_h->p1;
    Y_data_array_size = sh_h->p2;
    X_offset = sh_h->p3;
    Y_offset = sh_h->p4;
    X_zoom_ratio = sh_h->p5;
    Y_zoom_ratio = sh_h->p6;
    
    /* Allocate the image: */
    data_array = (char *) alloca(X_data_array_size*X_zoom_ratio*Y_data_array_size*Y_zoom_ratio);
    
    im = ((char *)(sh_h+1));
    pdata = data_array;
    pdata = convert_byte_to_pixel(X_data_array_size, Y_data_array_size,
				  im, pdata, X_zoom_ratio, Y_zoom_ratio);

    p = myCreateImage(pdata, X_data_array_size*X_zoom_ratio, Y_data_array_size*Y_zoom_ratio);
    myPutImage(p, X_offset, Y_offset, X_data_array_size*X_zoom_ratio, Y_data_array_size*Y_zoom_ratio);
    
    /* Deallocate the image: */
    p->data = NULL;
    XDestroyImage(p);
}


static void
my_cursor_cible()
{
}


static void
my_cursor_normal()
{
}


static void
draw_frame()
{
    int x0, y0, x1, y1;
    char * name;
    int color, title_color, background_color;

    x0 = sh_h->p1-1;
    y0 = sh_h->p2-1;
    x1 = sh_h->p3;
    y1 = sh_h->p4;
    color = sh_h->p5;
    title_color = sh_h->p6;
    background_color = sh_h->p7;

    y0 += palette_Y_size;
    y1 += palette_Y_size;
    name = (char *)(sh_h + 1);

    /* Draw the frame: */
    if (color < 0)
	/* Select an GXor BitBlt operation: */
	XSetFunction(display, gcdraw, GXxor);
    else
	XSetFunction(display, gcdraw, GXcopy);

    XSetForeground(display, gcdraw, color);
    XDrawRectangle(display,win,gcdraw,x0,y0,x1-x0,y1-y0);
    XDrawRectangle(display,save_pixmap,gcdraw,x0,y0,x1-x0,y1-y0);
    /* Restore the default copy BitBlt operation: */
    XSetFunction(display, gcdraw, GXcopy);

    if (strlen(name) > 0) {
	if (background_color >= 0) {
	    /* Erase the background of the title: */
	    XSetForeground(display, gcdraw, background_color);
	    XFillRectangle(display,save_pixmap,gc,x0,y1,x1-x0,20);
	    XFillRectangle(display,win,gc,x0,y1,x1-x0,20);
	}
	XSetForeground(display, gcdraw, title_color);
	XDrawString(display,win,gcdraw,x0+5,y1+15,name,strlen(name));
	XDrawString(display,save_pixmap,gcdraw,x0+5,y1+15,name,strlen(name));
	XDrawRectangle(display,win,gcdraw,x0,y1,x1-x0,20);
	XDrawRectangle(display,save_pixmap,gcdraw,x0,y1,x1-x0,20);
    }
    XFlush(display);
}


static void
nb_draw_palette()
{
	int i;
	if(depth != 1) return;
	for(i=0;i<256;i++) {
		XSetForeground(display, gccol, 1-byte2bit(i));
		XSetBackground(display, gccol, 1-byte2bit(i));
		XFillRectangle(display,save_pixmap,gccol,2*i,0,2,palette_Y_size);
		XFillRectangle(display,win,gccol,2*i,0,2,palette_Y_size);
	}
	XFlush(display);
}


static void
myscroll(y)
int y;
{
	XCopyArea(display,save_pixmap,save_pixmap,gc,0,y+palette_Y_size,W_X,W_Y-palette_Y_size-y,0,palette_Y_size);
	XCopyArea(display,win,win,gc,0,y+palette_Y_size,W_X,W_Y-palette_Y_size-y,0,palette_Y_size);
	XFillRectangle(display,save_pixmap,gc,0,W_Y-y,W_X,y);
	XFillRectangle(display,win,gc,0,W_Y-y,W_X,y);
}


static int
always_true(Display * display,
	    XEvent * event,
	    char * pointer)
{
	return 1;
}


static void
draw_palette()
{
	int i;
	if(depth == 1) {
		nb_draw_palette();
		return;
	}
	for(i=0;i<256;i++) {
		XSetForeground(display, gccol, i);
		XSetBackground(display, gccol, i);
		XFillRectangle(display,save_pixmap,gccol,2*i,0,2,palette_Y_size);
		XFillRectangle(display,win,gccol,2*i,0,2,palette_Y_size);
	}
	XFlush(display);
}


static void
before_unlocking()
{
	XEvent report;
	if(XCheckIfEvent(display,&report,always_true,0)) {
		analyse_event(&report);
	}
}


static void
wake_up()
{
	char *p;

	if(!sh_h || !sh_h->lock) return;
	switch(sh_h->flag) {
	case GR_START :
		sh_h->p1 = depth;
		break;
	case GR_CLOSE :
		shmctl(shmid,IPC_RMID,0);
		shmid = 0;
		sh_h = 0;
		break;
	case GR_CMAP :
		set_cmap(sh_h->p1,sh_h->p2,sh_h->p3,sh_h->p4);
		refresh_nb();
		break;
	case GR_BUF_CLIP:
	case GR_BUF:
		image();
		break;
	case GR_MOUSE:
		mouse = 1;
		my_cursor_cible();
		return;
	case GR_ISMOUSE:
		/* Reply any previous mouse event: */
		sh_h->p1 = x_mouse;
		sh_h->p2 = y_mouse;
		sh_h->p3 = b_mouse;
		sh_h->p4 = state_mouse;
		before_unlocking();
		sh_h->lock = 0;
		b_mouse = 0;
		kill(getppid(),SIGUSR2);
		return;
	case GR_PUT_FRAME:
		draw_frame();
		break;
	case GR_SET_COLOR:
		p = ((char *)(sh_h+1));
		memcpy(red,p,256);
		memcpy(green,p+256,256);
		memcpy(blue,p+512,256);
		set_cmap(4,0,0,-1);
		nb_draw_palette();
		refresh_nb();
		/*
		set_colormap(red,green,blue);
		*/
		break;
	case GR_SCROLL:
		myscroll(sh_h->p1);
		break;
	default :
		fprintf(stderr,"%s : unknown command %d\n",Argv[0],sh_h->flag);
	}
	before_unlocking();
	sh_h->lock = 0;
	kill(getppid(),SIGUSR2);
}


static void
analyse_event(XEvent * report)
{
	static int start = 1;
	char pc_buffer[20];
	int bufsize = 20;
	int charcount;
	KeySym keysym;
	XComposeStatus compose;

	switch(report->type) {
	case Expose:
		if(start) {
			if(bm) {
				if(depth==1) {
					databis = (char *)malloc(bytesperline* height);
					databis = convert_byte_to_pixel(width,height,data,databis,1,1);
					xi = XCreateImage(display, DefaultVisual(display,screen), depth,
					ZPixmap,0, databis, W_X,W_Y,16,bytesperline);
					save_pixmap = XCreatePixmap(display,win,W_X,W_Y,depth);
				} else {
					xi = XCreateImage(display, DefaultVisual(display,screen), depth,
						ZPixmap,0, data, W_X,W_Y,16,W_X);
					save_pixmap = XCreatePixmap(display,win,W_X,W_Y,depth);
				}
				XPutImage(display,save_pixmap,gc,xi,0,0,0,0,W_X,W_Y);
			} else {
				save_pixmap = XCreatePixmap(display,win,W_X,W_Y,depth);
				XFillRectangle(display,save_pixmap,gc,0,0,W_X,W_Y);
				XFillRectangle(display,win,gc,0,0,W_X,W_Y);
				set_cmap(0,0,0,-1);
				draw_palette();
				XFlush(display);
			}	
			start = 0;
		}
		XCopyArea(display,save_pixmap,win,gc,0,0,W_X,W_Y,0,0);
		if(shmid) {
			wake_up();
		}
		break;
	case KeyPress :
		*pc_buffer = 0;
		charcount =  XLookupString(&report->xkey, pc_buffer, bufsize, &keysym, &compose);
		if(charcount == 1) {
			switch(*pc_buffer) {
			case '1' :
				pal = 0;
				goto  palette;
			case '2' :
				pal = 1;
				goto  palette;
			case '3' :
				pal = 2;
				goto  palette;
			case '4' :
				pal = 3;
				goto  palette;
			case '0' :
				pal = 4;
				goto palette;
			case '+' :
			case '=' :
				repvalue++;
				goto  palette;
			case '-' :
			case '_' :
				repvalue--;
				goto  palette;
			case 'h' :
				startvalue++;
				goto  palette;
			case 'l' :
				startvalue--;
				goto  palette;
			case 'H' :
				startvalue++;
				clipvalue--;
				goto  palette;
			case 'L' :
				startvalue--;
				clipvalue++;
				goto  palette;
			case 'j' :
				clipvalue--;
				goto  palette;
			case 'k' :
				clipvalue++;
palette :
				if(repvalue < -2) repvalue = -2;
				if(repvalue > 6) repvalue = 6;
				if(clipvalue == -2) clipvalue = 255;
				if(clipvalue == 256) clipvalue = -1;
				startvalue &= 0xff;
				set_cmap(pal,repvalue,startvalue,clipvalue);
				nb_draw_palette();
				refresh_nb();
				break;
			case '?' :
			    fprintf(stderr,"XPOMP_set_color_map( , %d, %d, %d, %d);\n", pal, repvalue, startvalue, clipvalue);
				break;
			case 'q' :
			case '\004' :
				quit_proc();
				exit(0);
			default : break;
			}
		}
		break;
	case ButtonPress:
		switch(report->xbutton.button) {
		case Button1 :
		case Button3 :
		case Button2 :
			if(mouse) {
			    /* The application is waiting on the mouse: */
				mouse = 0;
				sh_h->p1 = report->xbutton.x;
				sh_h->p2 = report->xbutton.y - palette_Y_size;
				sh_h->p3 = report->xbutton.button;
				sh_h->p4 = report->xbutton.state;
				my_cursor_normal();
				sh_h->lock = 0;
				kill(getppid(),SIGUSR2);
			} else {
				x_mouse = report->xbutton.x;
				y_mouse = report->xbutton.y - palette_Y_size;
				b_mouse = report->xbutton.button;
				state_mouse = report->xbutton.state;
			}
			break;
		}
		break;
	default :
		;
	}
}


static void
open_shmem(shmid)
{
	char *p;
	int i;

#ifdef vax
	p = (char *) shmat(shmid,0x00400000,SHM_RND);
#else
	p = (char *) shmat(shmid,0,0);
#endif
	sh_h = (struct sh_header *) p;
	if(p == (char *) -1) {
		perror("pompx : shmat");
		i = 0;
		i = 1/i;
		exit(-1);
	}
}


char *convert_byte_to_pixel(X,Y,source,dest,zoomx,zoomy)
int X,Y,zoomx,zoomy;
char *source;
char *dest;
{
	int ZX;
	char *line,pix;
	int i,j,k,n;
	char *cdest;

	cdest = dest;
	if(0 && depth != 8) {
		ZX = (X * zoomx + 15)/16 * 2;
		for(j=0;j<Y;j++) {
			line = dest;
			pix = 0;
			n = 0;
			for(i=0;i<X;i++) {
				for(k=0;k<zoomx;k++) {
					pix <<= 1;
					pix |= byte2bit(*source);
					if(++n == 8) {
						*dest++ = ~pix;
						n = 0;
						pix = 0;
					}
				}
				source++;
			}
			if(n) {
				pix <<= 8 - n;
				*dest++ = ~pix;
			}
			for(i=1;i<zoomy;i++) {
				memcpy(dest,line,ZX);
				dest += ZX;
			}
		}
	} else {
		if(zoomx == 1 && zoomy == 1) return source; 
		ZX = X * zoomx;
		for(j=0;j<Y;j++) {
			line = dest;
			for(i=0;i<X;i++) {
				for(k=0;k<zoomx;k++) {
					*dest++ = *source;
				}
				source++;
			}
			for(i=1;i<zoomy;i++) {
				memcpy(dest,line,ZX);
				dest += ZX;
			}
		}
	}
	return cdest;
}


static void
open_nb_screen(x,y)
int x;
int y;
{
	if(depth == 8) return;
	x = (x + 15)/16 * 16;
	nb_x = x;
	nb_y = y;
	nb_screen = (char *) malloc(x*y);
	x = depth * x/8;
	nb_comp_screen = (char *) malloc(x*y);
}


#if defined(_AIX)
static void
usr2(int sig,
     int code,
     struct sigcontext * scp)
#elif defined(__linux)
static void
usr2(int sig)
#else
static void
usr2(int sig,
     int code,
     struct sigcontext * scp,
     char * addr)
#endif
{
	static in_usr2 =  0;

	if(in_usr2) {
		fprintf(stderr,"pompx : bug! 2 calls to usr2!\n");
		return;
	}

	in_usr2 = 1;
	if(want_to_wait) is_usr2 = 1;
	else {
		wake_up();
	}
	in_usr2 = 0;
	
#if defined(_AIX) || defined(__svr4__)
	/* AIX signals looks like SVR2 unreliable ones. Need to
           reinstall the handler! Sound the same with Solaris 2. */
	signal(SIGUSR2, usr2);
#endif
}


#if defined(_AIX)
static void
intr(int sig,
     int code,
     struct sigcontext * scp)
#elif defined(__linux)
static void
intr(int sig)
#else
static void
intr(int sig,
     int code,
     struct sigcontext * scp,
     char * addr)
#endif
{
}


static void
get_GC(Window win,
       GC * gc,
       GC * gcdraw,
       GC * gccol,
       XFontStruct * font_info)
{
    unsigned long valuemask = 0;
    XGCValues values;
    unsigned int line_width = 1;
    int line_style = LineSolid;
    int cap_style =  CapNotLast;
    int join_style = JoinBevel;

    *gc = XCreateGC(display, win, valuemask, &values);
    *gcdraw = XCreateGC(display, win, valuemask, &values);
    *gccol = XCreateGC(display, win, valuemask, &values);
    /*
       XSetFont(display, *gc, font_info->fid);
       */
    XSetForeground(display, *gc, 1);
    XSetBackground(display, *gc, 1);
    XSetFillRule(display, *gc, EvenOddRule);
    XSetFillStyle(display, *gc, FillSolid);
    XSetClipMask(display, *gc, None);
    XSetSubwindowMode(display, *gc, IncludeInferiors);
    XSetFunction(display, *gc, GXcopy);
    XSetLineAttributes(display, *gc, line_width, line_style, cap_style,
		       join_style);
    
    XCopyGC(display, *gc, -1, *gccol);    
    XCopyGC(display, *gc, -1, *gcdraw);

    /* Well... The following is indeed undone in draw_frame() */
    if(depth != 8) {
	XSetForeground(display, *gcdraw, 0);
	XSetBackground(display, *gcdraw, 0);
	XSetFunction(display, *gcdraw, GXcopy);
    } else {
	XSetForeground(display, *gcdraw, 255);
	XSetBackground(display, *gcdraw, 255);
	XSetFunction(display, *gcdraw, GXor);
    }
    /* Add a clipping rectangle to avoid draw_frame() to override the
       palette: */
    {
	XRectangle r;
	r.x = 0;
	r.y = palette_Y_size;
	r.width = W_X;
	r.height = W_Y - palette_Y_size;
	XSetClipRectangles(display, *gcdraw, 0, 0, &r, 1, YXSorted);
    }


    /* Palette GC: */
    XSetForeground(display, *gccol, 0);
    XSetBackground(display, *gccol, 255);
    XSetFunction(display, *gccol, GXcopy);
    XSetFillStyle(display, *gccol, FillSolid);
}


static void
load_font(font_info)
XFontStruct **font_info;
{
	char *fontname = "9x15";

	if((*font_info = XLoadQueryFont(display,fontname)) == 0) {
		fprintf(stderr,"pompx : Cannot open 9x15 font\n");
		exit(-1);
	}
}


XImage *
myCreateImage(char *pdata,
	      int ZX,
	      int ZY)
{
	int bytesperline;
	if(depth == 1)  {
		bytesperline = ((ZX + 15) & ~15) >> 3;
	} else {
		bytesperline = ZX;
	}
	return XCreateImage(display, DefaultVisual(display,screen), depth,
		ZPixmap,0, pdata, ZX,ZY,16,bytesperline);
}


int
main(argc,argv)
int argc;
char *argv[];
{
	struct rasterfile H;
	int x=0, y=0;
	unsigned int display_width, display_height;
	unsigned int border_width = 4;
	char *window_name  = "xPOMP - CRI ENSMP - http://www.cri.ensmp.fr/pips";
	char *icon_name = "xPOMP";
	Pixmap icon_pixmap = {0};
	XSizeHints size_hints;
	XEvent report;
	char *display_name = 0;
	int xmask,n,f = -1;
	Visual *visual;
	XColor colorcell[256];
	int i,size;
	char col[256];


	signal(SIGUSR2,usr2);
	signal(SIGINT,intr);
	Argv = argv;
	for(i=0;i<256;i++) {
		colorcell[i].pixel = i;
		colorcell[i].red = i << 8;
		colorcell[i].green = i << 8;
		colorcell[i].blue = i << 8;
		colorcell[i].flags = DoRed | DoGreen | DoBlue;
	}
	/*
	for(i=0;i<argc;i++) {
		fprintf(stderr,"%s ",argv[i]);
	}
	fprintf(stderr,"\n");
	*/
	for(i=1;i<argc;i++) {
		if(argv[i][0] == '-') {
			switch(argv[i][1]) {
			case 'm' :
				i++;
				if(i >= argc) break;
				shmid = atoi(argv[i]);
				open_shmem(shmid);
				break;
			case 's' :
				small = 1;
				break;
			case 'd' :
				dbx = 1;
				break;
			case 'D' :
				i++;
				if(i >= argc) break;
				display_name = argv[i];
				break;
			case 'w' :
				i++;
				if(i == argc) break;
				W_X = atoi(argv[i]);
				i++;
				if(i == argc) break;
				W_Y = atoi(argv[i]);
				break;
			case 'V' :
				fprintf(stderr,"%s",pc_version);
				exit(0);
			}
		} else {
			bm = argv[i];
			W_X = X_MAX_SIZE;
			W_Y = Y_MAX_SIZE;
		}
	}
	if(bm) {
		f = open(bm,O_RDONLY);
		if(f == -1) {
			perror(bm);
			exit(-1);
		}
		n = read(f,&H,sizeof(H));
		SWAI(n,H.ras_magic);
		if(n != RAS_MAGIC) {
			fprintf(stderr,"%s is not a rasterfile\n",bm);
			exit(-1);
		}
		SWAI(size,H.ras_length);
		data = (char *) malloc(size);
		SWAI(size,H.ras_maptype);
		if(size == RMT_EQUAL_RGB) {
			SWAI(size,H.ras_maplength);
			size = size/3;
			read(f,col,size);
			for(i=0;i<size;i++) {
				colorcell[i].red = col[i] << 8;
			}
			read(f,col,size);
			for(i=0;i<size;i++) {
				colorcell[i].green = col[i] << 8;
			}
			read(f,col,size);
			for(i=0;i<size;i++) {
				colorcell[i].blue = col[i] << 8;
			}
		} else {
			SWAI(size,H.ras_maplength);
			read(f,data,size);
		}	
		SWAI(size,H.ras_length)
		read(f,data,size);
		SWAI(W_X,H.ras_width);
		SWAI(W_Y,H.ras_height);
	}
	if(!(display = XOpenDisplay(display_name))) {
		fprintf(stderr,"pompx: cannot connect to X server %s\n",XDisplayName(display_name));
		exit(-1);
	}
	screen = DefaultScreen(display);
	display_width = DisplayWidth(display, screen);
	display_height = DisplayHeight(display, screen);
	depth = DefaultDepth(display,screen);
	if(!bm) W_Y += palette_Y_size;
	width = W_X;
	height = W_Y;
	if(depth != 8) {
		bytesperline = ((W_X + 15) & ~15) >> 3;
	} else {
		bytesperline = W_X;
	}

	xmask = 0;

	xswa.background_pixmap = 0;
	xmask |= CWBackPixmap;
	xswa.background_pixel = 0;
	xmask |= CWBackPixel;
	xswa.border_pixel = 0;
	xmask |= CWBorderPixel;
	visual = DefaultVisual(display,screen);
	xmask = 0;

	win  = XCreateWindow(display, RootWindow(display,screen),
		x,y, width,height,border_width,depth,InputOutput,visual,xmask,
		&xswa);
	if(depth > 1) {
		cmap = XCreateColormap(display, RootWindow(display,screen),
			visual,AllocAll);
		set_cmap(0,0,0,-1);
		XSetWindowColormap(display,win,cmap);
		/*
		XInstallColormap(display,cmap);
		*/
	}
	size_hints.flags = PPosition | PSize | PMinSize;
	size_hints.x = x;
	size_hints.y = y;
	size_hints.width = width;
	size_hints.height = height;
	size_hints.min_width = 512;
	size_hints.min_height = 522;

	XSetStandardProperties(display, win, window_name, icon_name, icon_pixmap,
	argv, argc, &size_hints);

	XSelectInput(display, win, ExposureMask | KeyPressMask | ButtonPressMask |
		StructureNotifyMask);

	load_font(&font_info);

	get_GC(win, &gc, &gcdraw, &gccol, font_info);

	open_nb_screen(width,height-palette_Y_size);
	XMapWindow(display, win);
	XSync(display,False);
	/*
	XFlush(display);
	*/
	while(1) {
		want_to_wait = 1;
		if(is_usr2) {
			is_usr2 = 0;
			wake_up();
		}
		if(!XCheckIfEvent(display,&report,always_true,0)) {
			want_to_wait = 0;
			if(is_usr2) {
				is_usr2 = 0;
				wake_up();
			}
			XNextEvent(display, &report);
			want_to_wait = 1;
		}
		analyse_event(&report);
	}
}
