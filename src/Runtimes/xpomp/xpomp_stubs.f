! $RCSfile: xpomp_stubs.f,v $ (version $Revision$)
! $Date: 1996/09/03 18:13:14 $, 
!
! List of fake functions to have PIPS happy with 
! the same « effects » as the xPOMP graphical library.
! !fcd$ io directive is used by the HPFC compiler to consider
! these functions as IO routines.
! !fcd$ fake directives tells not to compile these functions,
! since they will be provided somewhere else.
!
!     Ronan.Keryell@cri.ensmp.fr
!
      subroutine xpomp_open_display(x, y, d)
      integer x, y, d
!fcd$ io
!fcd$ fake
      print *, x, y
      read *, d
      end
      
      subroutine xpomp_close_display(d)
      integer d
!fcd$ io
!fcd$ fake
      print *, d
      end
      
      subroutine xpomp_get_current_default_display(d)
      integer d
!fcd$ io
!fcd$ fake
      read *, d
      end
      
      subroutine xpomp_set_current_default_display(d, r)
      integer d, r
!fcd$ io
!fcd$ fake
      print *, d
      read *, r
      end
      
      subroutine xpomp_get_depth(d)
      integer d
!fcd$ io
!fcd$ fake
      read *, d
      end
      
      subroutine xpomp_set_color_map(screen,
     &     pal, cycle, start, clip, r)
      integer screen, pal, cycle, start, clip, r
!fcd$ io
!fcd$ fake
      print *, screen, pal, cycle, start, clip
      read *, r
      end
      
      subroutine xpomp_set_user_color_map(screen,
     &     red, green, blue, r)
      integer screen
      character red(256), green(256), blue(256)
      integer r, i
!fcd$ io
!fcd$ fake
      do i = 1, 256
         print *, screen, red(i), green(i), blue(i)
      enddo
      read *, r
      end

      subroutine xpomp_wait_mouse(screen, X, Y, state, r)
      integer screen, X, Y, state, r
!fcd$ io
!fcd$ fake
      print *, screen
      read *, X, Y, state, r
      end

      subroutine xpomp_is_mouse(screen, X, Y, state, r)
      integer screen, X, Y, state, r
!fcd$ io
!fcd$ fake
      print *, screen
      read *, X, Y, state, r
      end
      
      subroutine xpomp_flash(window,
     &     image,
     &     X_data_array_size, Y_data_array_size,
     &     X_offset, Y_offset,
     &     X_zoom_ratio, Y_zoom_ratio,
     &     status)
      integer window
      integer X_data_array_size, Y_data_array_size
      character image(X_data_array_size, Y_data_array_size)
      integer X_offset, Y_offset
      integer X_zoom_ratio, Y_zoom_ratio
      integer status, x, y
!fcd$ io
!fcd$ fake
      print *, window, X_data_array_size, Y_data_array_size,
     &     X_offset, Y_offset,
     &     X_zoom_ratio, Y_zoom_ratio
      do x = 1, X_data_array_size
         do y = 1, Y_data_array_size
            print *, image(x, y)
         enddo
      enddo
      read *, status
      end

      subroutine xpomp_show_real4(screen, image,
     &     X_data_array_size, Y_data_array_size,
     &     X_offset, Y_offset,
     &     X_zoom_ratio, Y_zoom_ratio,
     &     min_value, max_value,
     &     status)
      integer screen
      integer X_data_array_size, Y_data_array_size
      real*4 image(X_data_array_size, Y_data_array_size)
      integer X_offset, Y_offset
      integer X_zoom_ratio, Y_zoom_ratio
      real*8 min_value, max_value
      integer status, x, y
!fcd$ io
!fcd$ fake
      print *, screen, X_data_array_size, Y_data_array_size,
     &     X_offset, Y_offset,
     &     X_zoom_ratio, Y_zoom_ratio
      do x = 1, X_data_array_size
         do y = 1, Y_data_array_size
            print *, image(x, y)
         enddo
      enddo
      read *, status
      end

      subroutine xpomp_show_real8(screen, image,
     &     X_data_array_size, Y_data_array_size,
     &     X_offset, Y_offset,
     &     X_zoom_ratio, Y_zoom_ratio,
     &     min_value, max_value,
     &     status)
      integer screen
      integer X_data_array_size, Y_data_array_size
      real*8 image(X_data_array_size, Y_data_array_size)
      integer X_offset, Y_offset
      integer X_zoom_ratio, Y_zoom_ratio
      real*8 min_value, max_value
      integer status, x, y
!fcd$ io
!fcd$ fake
      print *, screen, X_data_array_size, Y_data_array_size,
     &     X_offset, Y_offset,
     &     X_zoom_ratio, Y_zoom_ratio
      do x = 1, X_data_array_size
         do y = 1, Y_data_array_size
            print *, image(x, y)
         enddo
      enddo
      read *, status
      end
      
      subroutine xpomp_scroll(window, y, result)
      integer window, y, result
!fcd$ io
!fcd$ fake
      print *, window, y
      read *, result
      end
      
      subroutine xpomp_draw_frame(window,
     &     title,
     &     title_color, background_color,
     &     X0, Y0, X1, Y1,
     &     color,
     &     status)
      integer window
      character*(*) title
      integer title_color, background_color
      integer X0, Y0, X1, Y1
      integer color
      integer status
!fcd$ io
!fcd$ fake
      print *, window, title, title_color, background_color,
     &     X_data_array_size, Y_data_array_size,
     &     X0, Y0, X1, Y1, color
      read *, status
      end
      
      subroutine xpomp_show_usage()
!fcd$ io
!fcd$ fake
      print *, 'Some help...'
      end
