!
! test translation of regions with global exact IO effects.
!
      program io
      real a(10)
      print *, a
      call printa(a)
      call printb(a)
      end

      subroutine printa(x)
      real x(5)
      print *, x
      end
      
      subroutine printb(x)
      real x(1)
      print *, x
      end
