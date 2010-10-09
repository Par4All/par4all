      program w18

C     Check refined transformer for while loop with unknownn entry condition

      integer t, l, x
      external alea
      logical alea

      if(.TRUE.) then
         do while(x.le.9.and.alea())
            t = t + 1
            l = l + 1
            x = x + 1
         enddo
      endif

C     Because of the call to alea(), no information about x is available
      print *, i

      end
      logical function alea()
      read *, x
      if(x.gt.0.) then
         alea = .TRUE.
      else
         alea = .FALSE.
      endif
      return
      end
