/* code taken from  
 * http://fast-edge.googlecode.com
 * and adapted to c99
 */

/*
        FAST-EDGE
        Copyright (c) 2009 Benjamin C. Haynor

        Permission is hereby granted, free of charge, to any person
        obtaining a copy of this software and associated documentation
        files (the "Software"), to deal in the Software without
        restriction, including without limitation the rights to use,
        copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the
        Software is furnished to do so, subject to the following
        conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
        EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
        OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
        NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
        HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
        WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
        FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
        OTHER DEALINGS IN THE SOFTWARE.
*/


struct image {
    int width;
    int height;
    unsigned char pixel_data;
};
#define min(a,b) ((a)>(b) ? (b) : (a))
#define max(a,b) ((a)>(b) ? (a) : (b))

typedef float type_t;
//typedef unsigned  char type_t;

void dilate_1d_h(int width, int height, type_t img[width][height], type_t img_out[width][height]) {
    int x, y;
    for (y = 2 ; y < height -2 ; ++y) {
        for (x = 2; x < width - 2; x++) {
            img_out[y][x] = max(max(max(max(img[y][x-2], img[y][x-1]), img[y][x]), img[y][x+1]), img[y][x+2]);  
        }
    }
}

void dilate_1d_v(struct image * img, struct image * img_out) {
    int x, y;
    for (y = 2 ; y < height -2 ; ++y) {
        for (x = 2; x < width - 2; x++) {
            img_out[y][x] = max(max(max(max(img[y-2][x], img[y-1][x]), img[y][x]), img[y+1][x]), img[y+2][x]);  
        }
    }
}

void erode_1d_h(int width, int height, type_t img[width][height], type_t img_out[width][height]) {
    int x, y;
    for (y = 2 ; y < height -2 ; ++y) {
        for (x = 2; x < width - 2; x++) {
            img_out[y][x] = min(min(min(min(img[y][x-2], img[y][x-1]), img[y][x]), img[y][x+1]), img[y][x+2]);  
        }
    }
}

void erode_1d_v(struct image * img, struct image * img_out) {
    int x, y;
    for (y = 2 ; y < height -2 ; ++y) {
        for (x = 2; x < width - 2; x++) {
            img_out[y][x] = min(min(min(min(img[y-2][x], img[y-1][x]), img[y][x]), img[y+1][x]), img[y+2][x]);  
        }
    }
}

void erode(int width, int height,type_t img_in[width][height], type_t img_scratch[width][height], type_t img_out[width][height]) {
        erode_1d_h(height,width,img_in, img_scratch);
        erode_1d_v(height,width,img_scratch, img_out);
}

void dilate(int width, int height, type_t img_in[width][height], type_t img_scratch[width][height], type_t img_out[width][height]) {
        dilate_1d_h(height,width,img_in, img_scratch);
        dilate_1d_v(height,width,img_scratch, img_out);
}

