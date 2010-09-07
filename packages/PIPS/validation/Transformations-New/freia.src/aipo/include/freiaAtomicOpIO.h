/***************************************************************
 * Author : Christophe Clienti
 * Date : 10/22/2008
 * Copyright (c) 2008 Thales Research & Technologies (FR)
 ***************************************************************/

#ifdef __cplusplus
extern "C" {
#endif


#ifndef __FREIA_ATOMIC_OP_IO_H__
#define __FREIA_ATOMIC_OP_IO_H__

#include <freiaCommonTypes.h>

  /*!
   * \defgroup freia_aipo_io Basic input/output operations
   * \ingroup freia_aipo
   * @{
   */


  /*!  
    \brief Initialize a freia_dataio instance to manage an input video flow.

    When using PC platform images could read from the disk or from a
    video4linux peripherial. A config file ("fdescio") specifying which device or
    image files must exist in the same directory of the application

    Content of the "fdescio" file
    \code
    # Video flow description

    # Read images from disk
    fdin0 {
      type = pgm;
      dir = video_in_0;
      file = video_in_%08d.pgm;
      start = 33;
    }
    
    # Read images from /dev/video0
    fdin1 {
      type = v4l;
      dir = /dev;
      file = video0;
      start = 0;
    }

    # Store pgm files into video_out_0 directory
    fdout0 {
      type = pgm;
      dir = video_out_0;
      file = video_out_%08d.pgm;
      start = 0;
    }

    # Display images in a window
    fdout1 {
      type = sdl;
      dir = none;
      file = none;
      start = 0;
    }
    \endcode

    \param[out] fdio freia_dataio instance pointer
    \param[in] vidchan video channel
    \return error code
  */
  extern freia_error freia_aipo_open_input(freia_dataio *fdio, uint32_t vidchan);

  /*!  
    \brief Initialize a freia_dataio instance to manage an output video flow.

    \sa freia_aipo_tx_image

    \param[out] fdio freia_dataio instance pointer
    \param[in] vidchan video channel
    \param[in] framewidth width of video frames
    \param[in] frameheight height of video frames
    \param[in] framebpp number of bit per pixel of video frames
    \return error code
  */
  extern freia_error freia_aipo_open_output(freia_dataio *fdio, uint32_t vidchan, uint32_t framewidth, uint32_t frameheight, uint32_t framebpp);


  /*!  
    \brief Close the freia_dataio instance

    \param[in] fdio freia_dataio instance pointer
    \return error code
  */
  extern freia_error freia_aipo_close_input(freia_dataio *fdio);


  /*!  
    \brief Close the freia_dataio instance

    \param[in] fdio freia_dataio instance pointer
    \return error code
  */
  extern freia_error freia_aipo_close_output(freia_dataio *fdio);


  /*!  
    \brief Retrieve an image from the harddisk or from a camera

    \sa freia_aipo_tx_image

    \param[out] imdest destination image 
    \param[in] fdio image IO descriptor considered as input
    \return error code
  */
  extern freia_error freia_aipo_rx_image(freia_data2d *imdest, freia_dataio *fdio);					

  /*!  
    \brief Send or write an image to the harddisk or to screen

    \sa freia_aipo_rx_image

    \param[in] imsrc source image 
    \param[out] fdio image IO descriptor considered as output
    \return error code
  */
  extern freia_error freia_aipo_tx_image(freia_data2d *imsrc, freia_dataio *fdio);
					


  /*!@}*/


#endif

#ifdef __cplusplus
}
#endif
