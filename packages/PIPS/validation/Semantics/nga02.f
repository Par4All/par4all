      program nga02

C     Extraction de conditions de non-debordement: you want N<=33 before
C     the print statement. It works with the derivative fix-point

      real a(33)

      read *, n

      if(n.lt.5) stop

      do i = 1, n
         if(i.gt.33) stop
         a(i) = 0.
      enddo

      print *, a, n

      end
