#include <stdlib.h>
#include <string.h>
#include <gtk/gtk.h>
#include <gdk-pixbuf/gdk-pixdata.h>

#include "graphics.h"

static GtkWidget* window = NULL;
static GtkWidget* image = NULL;
static GdkPixdata pix_data;

void graphic_destroy(void) {
  gtk_main_quit();
  window = NULL;
}

static void init(int argc, char **argv) {
  gtk_init(&argc, &argv);

  pix_data.magic = GDK_PIXBUF_MAGIC_NUMBER;
  pix_data.height = NCELL;
  pix_data.width = NCELL;
  pix_data.pixdata_type = GDK_PIXDATA_COLOR_TYPE_RGB;
  pix_data.rowstride = NCELL * sizeof(int);
  pix_data.pixel_data = malloc(sizeof(int) * pix_data.height
      * pix_data.rowstride);
  memset(pix_data.pixel_data, 0, sizeof(int) * pix_data.height
      * pix_data.rowstride);



  window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
  gtk_window_set_title(GTK_WINDOW(window), "Output");

  // Here we just set a handler for delete_event that immediately exits GTK.
  g_signal_connect(G_OBJECT(window),
                   "delete_event",
                   G_CALLBACK(graphic_destroy),
                   NULL);

  gtk_container_set_border_width(GTK_CONTAINER(window), 10);
  gtk_widget_show(window); // Pitfall!! Must put this here and NOT at end.

  GtkWidget *vbox = gtk_vbox_new(FALSE, 2);
  gtk_container_add(GTK_CONTAINER(window), vbox);
  gtk_widget_show(vbox);

  image = gtk_drawing_area_new();
  gtk_widget_set_size_request(image, 200, 200);
  gtk_box_pack_start(GTK_BOX(vbox), image, TRUE, TRUE, 0);
  gtk_widget_show(image);

  gtk_widget_show(window);

  gdk_rgb_init();
}

void graphic_draw(int argc, char **argv, int histo[NCELL][NCELL][NCELL]) {
  if(window == NULL) {
    init(argc, argv);
  }

  guchar pixbuf[3 * NCELL * NCELL];

  for (int x = 0; x < NCELL; x++) {
    for (int y = 0; y < NCELL; y++) {
      int sum = 0;
      for (int z = 0; z < NCELL; z++) {
        sum += histo[x][y][z];
      }
      pixbuf[x * NCELL * 3 + y * 3] = sum & 0xFF0000 >> 16;
      pixbuf[x * NCELL * 3 + y * 3 + 1] = sum & 0xFF00 >> 8;
      pixbuf[x * NCELL * 3 + y * 3 + 2] = sum & 0xFF;
    }
  }

  gdk_draw_rgb_image(image->window,
                     *image->style->fg_gc,
                     0,
                     0,
                     NCELL,
                     NCELL,
                     GDK_RGB_DITHER_NONE,
                     pixbuf,
                     NCELL * 3);

  while(gtk_events_pending())
    gtk_main_iteration();
}
