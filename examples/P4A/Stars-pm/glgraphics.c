#include <stdio.h>
#include <GL/glut.h>
#include <pthread.h>
#include <math.h>
#include "varglob.h"
#include "glgraphics.h"
int pthread_kill(pthread_t thread, int sig);
int usleep(int usec);

int idList;

static coord (*pos)[NCELL][NCELL] = NULL;
static int (*histo)[NCELL][NCELL] = NULL;

pthread_t thread1 = 0;
int redisplay_p = 2;

static int yrot = 0;
static int blend = 1;
static int light = 0;
static int init_p = 0;

/* ascii codes for various special keys */
#define ESCAPE 27
#define PAGE_UP 73
#define PAGE_DOWN 81
#define UP_ARROW 72
#define DOWN_ARROW 80
#define LEFT_ARROW 75
#define RIGHT_ARROW 77

void renderScene(void) {
  if(init_p) {
    if(redisplay_p) {
      redisplay_p--;
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

      glColor4f(1.0f, 1.0f, 1.0f, 0.05f);


      if(pos) {
        glBegin(GL_POINTS);
        for (int i = 0; i < NCELL; i++) {
          for (int j = 0; j < NCELL; j++) {
            for (int k = 0; k < NCELL; k++) {
              float x = pos[i][j][k]._[0] - 3;
              float y = pos[i][j][k]._[1] - 3;
              float z = pos[i][j][k]._[2] - 3;
              glVertex3f(x, y, z);
              //        printf("%f %f %f\n",x,y,z);
            }
          }
        }
        glEnd();
      }

      if(histo) {
        float max = 0;
        float delta = 6.0f/128.0f;
        for (int i = 0; i < NCELL; i++) {
          for (int j = 0; j < NCELL; j++) {
            int sum = 0;
            for (int k = 0; k < NCELL; k++) {
              sum +=histo[i][j][k];
            }
            float c = sqrtf(sqrtf((float)sum/11000));
            if(c>max) max=c;
          }
        }
        printf("max : %f\n",max);
        for (int i = 0; i < NCELL; i++) {
          float i_ = ((float)i)*delta-3.0;
          for (int j = 0; j < NCELL; j++) {
            float j_ = ((float)j)*delta-3.0;
            int sum = 0;
            for (int k = 0; k < NCELL; k++) {
              sum +=histo[i][j][k];
            }
            float c = sqrtf(sqrtf((float)sum/max));
            glBegin(GL_POLYGON);
            glColor4f(1.0f,1.0f,1.0f, c);
            glVertex3f(i_,j_,0);
            glVertex3f(i_+delta,j_,0);
            glVertex3f(i_+delta,j_+delta,0);
            glVertex3f(i_,j_+delta,0);
            glEnd();
            if(i_ < 0.5 && i_ > -0.5 && j_ < 0.5 && j_ > -0.5 )
            printf("Drawing from %f %f to %f %f with color %f\n",i_,j_,i_+delta,j_+delta,c);
          }
        }
      }

      glFlush();
    }
    glutPostRedisplay();
  }
}
/* The function called whenever a normal key is pressed. */
void keyPressed(unsigned char key, int x, int y) {
  /* avoid thrashing this procedure */
  usleep(100);

  switch(key) {
    case ESCAPE: // kill everything.
      /* exit the program...normal termination. */
      exit(1);
      break; // redundant.

    case 'b':
    case 'B': // switch the blending
      printf("B/b pressed; blending is: %d\n", blend);
      blend = blend ? 0 : 1; // switch the current value of blend, between 0 and 1.
      if(blend) {
        glEnable(GL_BLEND);
        glDisable(GL_DEPTH_TEST);
      } else {
        glDisable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);
      }
      graphic_glupdate(NULL);
      printf("Blending is now: %d\n", blend);
      break;

    case 'l':
    case 'L': // switch the lighting
      printf("L/l pressed; lighting is: %d\n", light);
      light = light ? 0 : 1; // switch the current value of light, between 0 and 1.
      if(light) {
        glEnable(GL_LIGHTING);
      } else {
        glDisable(GL_LIGHTING);
      }
      printf("Lighting is now: %d\n", light);
      break;

    default:
      printf("Key %d pressed. No action there yet.\n", key);
      break;
  }
}

/* The function called whenever a normal key is pressed. */
void specialKeyPressed(int key, int x, int y) {
  /* avoid thrashing this procedure */
  usleep(100);

  switch(key) {
    case GLUT_KEY_LEFT: // look left
      yrot += 1.5f;
      break;

    case GLUT_KEY_RIGHT: // look right
      yrot -= 1.5f;
      break;

    default:
      printf("Special key %d pressed. No action there yet.\n", key);
      break;
  }
  glLoadIdentity();
  glRotatef(yrot, 0, 1.0f, 0);
  glMatrixMode(GL_MODELVIEW);
  glOrtho(-3, 3, -3, 3, -10.0, 10.0);
  //  glutPostRedisplay();
  graphic_glupdate(NULL);
}

/* The function called when our window is resized (which shouldn't happen, because we're fullscreen) */
GLvoid ReSizeGLScene(GLsizei Width, GLsizei Height) {
  /*
   if(Height == 0) // Prevent A Divide By Zero If The Window Is Too Small
   Height = 1;

   glViewport(0, 0, Width, Height); // Reset The Current Viewport And Perspective Transformation

   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();

   gluPerspective(45.0f, (GLfloat)Width / (GLfloat)Height, 0.1f, 100.0f);*/
  glLoadIdentity();
  glRotatef(yrot, 0, 1.0f, 0);
  glMatrixMode(GL_MODELVIEW);
  glOrtho(-3, 3, -3, 3, -10.0, 10.0);
}

static int argc = 0;
static char **argv = NULL;
static void init() {
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA | GLUT_DEPTH | GLUT_ALPHA);
  glutInitWindowPosition(100, 100);
  glutInitWindowSize(500, 500);
  glutCreateWindow("Stars");
  glClearColor(0, 0, 0, 1);
  glEnable(GL_BLEND);

  glutDisplayFunc(&renderScene);
  /* Register the function called when our window is resized. */
  glutReshapeFunc(&ReSizeGLScene);

  /* Register the function called when the keyboard is pressed. */
  glutKeyboardFunc(&keyPressed);

  /* Register the function called when special keys (arrows, page down, etc) are pressed. */
  glutSpecialFunc(&specialKeyPressed);

  init_p = 1;
}

void *mainloop(void *unused) {
  init();
  renderScene();
  glutMainLoop();
  thread1 = 0;
  return NULL;
}

void graphic_gldraw(int argc_, char **argv_, coord pos_[NCELL][NCELL][NCELL]) {
  argc = argc_;
  argv = argv_;
  pos = pos_;
  histo = NULL;
  if(thread1 == 0) {
    pthread_create(&thread1, NULL, mainloop, (void*)NULL);
  }
}

void graphic_gldraw_histo(int argc_, char **argv_, int histo_[NCELL][NCELL][NCELL]) {
  argc = argc_;
  argv = argv_;
  pos = NULL;
  histo = histo_;
  if(thread1 == 0) {
    pthread_create(&thread1, NULL, mainloop, (void*)NULL);
  }
}

void graphic_gldestroy(void) {
  pthread_kill(thread1, 9);
  thread1 = 0;
}

void graphic_glupdate(coord pos_[NCELL][NCELL][NCELL]) {
  redisplay_p++;
}

