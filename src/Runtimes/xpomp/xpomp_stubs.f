c     List of fake functions to have PIPS happy with 
c     the same « effects » as the xPOMP graphical library.
c     !fcd$io directive is used by the HPFC compiler to compile
c     these functions as IO routines.
c
c     Ronan.Keryell@cri.ensmp.fr

      subroutine xpomp_open_display(x, y, d)
      integer x, y, d
!ldf$ -u xpomp_open_display
!fcd$ io
      print *, x, y
      read *, d
      end
      
      subroutine xpomp_close_display(d)
      integer d
!ldf$ -u xpomp_close_display
!fcd$ io
      print *, d
      end
      
      subroutine xpomp_get_current_default_display(d)
      integer d
!ldf$ -u xpomp_get_current_default_display
!fcd$ io
      read *, d
      end
      
      subroutine xpomp_set_current_default_display(d, r)
      integer d, r
!ldf$ -u xpomp_set_current_default_display
!fcd$ io
      print *, d
      read *, r
      end
      
      subroutine xpomp_get_depth(d)
      integer d
!ldf$ -u xpomp_get_depth
!fcd$ io
      read *, d
      end
      
      subroutine xpomp_set_color_map(screen,
     &     pal, cycle, start, clip, r)
      integer screen, pal, cycle, start, clip, r
!ldf$ -u xpomp_set_color_map
!fcd$ io
      print *, screen, pal, cycle, start, clip
      read *, r
      end
      
      subroutine xpomp_set_user_color_map(screen,
     &     red, green, blue, r)
      integer screen
      character red(256), green(256), blue(256)
      integer r, i
!ldf$ -u xpomp_set_user_color_map
!fcd$ io
      do i = 1, 256
         print *, screen, red(i), green(i), blue(i)
      enddo
      read *, r
      end

      subroutine xpomp_wait_mouse(screen, X, Y, r)
      integer screen, X, Y, r
!ldf$ -u xpomp_wait_mouse
!fcd$ io
      print *, screen
      read *, X, Y, r
      end

      subroutine xpomp_is_mouse(screen, X, Y, r)
      integer screen, X, Y, r
!ldf$ -u xpomp_is_mouse
!fcd$ io
      print *, screen
      read *, X, Y, r
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
!ldf$ -u xpomp_flash_
!fcd$ io
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
!ldf$ -u xpomp_show_real4
!fcd$ io
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
!ldf$ -u xpomp_show_real8
!fcd$ io
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
      
      subroutine xpomp_show_usage()
!ldf$ -u xpomp_show_usage
!fcd$ io
      print *, 'Some help...'
      end
