      program type25

C     Check multi-type initialization routine

      character*20 s

      call init(s, x, i)

      print *, s, x, i

      end

      subroutine init(fs, fx, fi)

      character*(*) fs
      integer fi

      fs = "Hello World!"
      fx = 2.0
      fi = 1

      end
