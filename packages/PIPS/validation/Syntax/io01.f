      program io01

C     Check that bug discovered at EDF has been fixed

      l = 3
      write(3+i, 1000) k
 1000 format(i10)
      write(3+i, *) k
      end
