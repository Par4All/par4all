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
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "xpomp_graphic.h"

enum {
    X_data_array_size = 100,
    Y_data_array_size = 100
};

int
main(void)
{
    int i, x, y;
    XPOMP_display d2, d3;
    unsigned char data[X_data_array_size][Y_data_array_size];

    XPOMP_display d = XPOMP_open_display(590, 290);
    printf("Display = %d\n", d);

    d2 = XPOMP_open_display(590, 290);
    printf("Display = %d\n", d2);

    /* 
    x = 100;
    y = 100;
    xpompfopendisplay_(&x, &y, &d3);
    printf("Display = %d\n", d3);
    */

    for(x = 0; x < X_data_array_size; x++)
	for(y = 0; y < Y_data_array_size; y++)
	    data[x][y] = 256*(y + Y_data_array_size*x)/(X_data_array_size*Y_data_array_size);

    XPOMP_flash(d, (unsigned char *) data, X_data_array_size, Y_data_array_size, 20, 30, 3, 2);

    {
	float float_data[X_data_array_size][Y_data_array_size];
	double X, Y, dist;
	for(x = 0; x < X_data_array_size; x++) {
	    X = (x - X_data_array_size/2)/10.0;
	    for(y = 0; y < Y_data_array_size; y++) {
		Y = (y - Y_data_array_size/2)/10.0;
		dist = X*X + Y*Y;
		float_data[x][y] = X+Y;
		float_data[x][y] = sin(dist)/(dist + 0.00001);
	    }
	}
	XPOMP_show_float(d, (float *)float_data, X_data_array_size, Y_data_array_size, 0, 0, 1, 1, -1, 2);
    }

    /* Display with a random colormap: */
    {
	unsigned char red[256];
	unsigned char green[256];
	unsigned char blue[256];

	for (i = 0; i < 256; i++) {
	    int r = rand();
	    red[i] = r & 255;
	    green[i] = (r >> 8) & 255;
	    blue[i] = (r >> 16) & 255;
	}
	(void) XPOMP_set_user_color_map(d, red, green, blue);
    }

    for(i = 4; i >= 0; i--) {
	system("sleep 1");
	(void) XPOMP_set_color_map(d, i, 0, 0, -1);
	XPOMP_draw_frame(d, "", 0, 0,
			 X_data_array_size/2 - 10*i, Y_data_array_size/2 - 5*i,
			 X_data_array_size/2 + 10*i, Y_data_array_size/2 + 5*i,
			 -128);
	(void) XPOMP_scroll(d, i);
    }
    (void) XPOMP_scroll(d, -10);

    system("sleep 1");
    for (i = 255; i >= -1; i--) {
	(void) XPOMP_set_color_map(d, 1, 0, 0, i);
	printf("%d ", i);
    }

    while(1) {
	int button, state;
	unsigned char color;
	button = XPOMP_wait_mouse(d, &x, &y, &state);
	printf("XPOMP_wait_mouse: X=%d, Y=%d, button=%d, state=%d\n",
	       x, y, button, state);
	switch(button) {
	case 1:
	    color = 0;
	    break;
	case 2:
	    color = 100;
	    break;
	case 3:
	    color = 200;
	    break;
	}
	/* Display a point at mouse position: */
	XPOMP_flash(d, &color, 1, 1, x, y, 3, 3);
	printf("Point displayed with color %ud at (%d,%d).\n", color, x, y);
	XPOMP_draw_frame(d, "Here", 10, 200, x - 20, y - 20, x + 20, y + 20, 100);
    }
    
    /*
    XPOMP_close_display(d);
    return 0;
    */
}
