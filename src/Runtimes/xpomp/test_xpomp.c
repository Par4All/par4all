#include <stdlib.h>
#include <hpfc_graphic.h>

enum {
    X_data_array_size = 100,
    Y_data_array_size = 100
};

int
main()
{
    int i, x, y;
    unsigned char data[X_data_array_size][Y_data_array_size];

    HPFC_display d = HPFC_open_display(590, 290);
    printf("Display = %d\n", d);

    for(x = 0; x < X_data_array_size; x++)
	for(y = 0; y < Y_data_array_size; y++)
	    data[x][y] = 256*(y + Y_data_array_size*x)/(X_data_array_size*Y_data_array_size);

    HPFC_flash(d, (unsigned char *) data, X_data_array_size, Y_data_array_size, 20, 30, 3, 2);

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
	HPFC_show_float(d, (float *)float_data, X_data_array_size, Y_data_array_size, 0, 0, 1, 1, -1, 2);
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
	(void) HPFC_set_user_color_map(d, red, green, blue);
    }

    for(i = 4; i >= 0; i--) {
	system("sleep 1");
	(void) HPFC_set_color_map(d, i, 0, 0, -1);
    }

    system("sleep 1");
    for (i = 255; i >= -1; i--) {
	(void) HPFC_set_color_map(d, 1, 0, 0, i);
	printf("%d ", i);
    }

    while(1) {
	int button;
	unsigned char color;
	button = HPFC_wait_mouse(d, &x, &y);
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
	HPFC_flash(d, &color, 1, 1, x, y, 3, 3);
	printf("Point displayed with color %ud at (%d,%d).\n", color, x, y);
    }
    
    /*
    HPFC_close_display(d);
    */
    return 0;
}
