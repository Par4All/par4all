enum gr_flag {
GR_START,
GR_BUF,
GR_MOUSE,
GR_ISMOUSE,
GR_CMAP,
GR_PUT_FRAME,
GR_SET_COLOR,
GR_SCROLL,
GR_CLOSE,
GR_BUF_CLIP
};
struct sh_header{
int lock:1;
int activite:1;
enum gr_flag flag;
int bufsize;
int id;
int p1;
int p2;
int p3;
int p4;
int p5;
int p6;
int p7;
int p8;
int p9;
int p10;
char *buf1;
char *buf2;
};
