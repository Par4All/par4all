      program unstruc07

C     Check that desugared loop might be handled

      real a(100)

      read *, n

      do i = 1, n
         a(i) = 0.
         if(a(i).lt.a(i-1)) go to 200
      enddo

 200  continue
      print *, i, n

      end
