      program type26

C     Check multi-type initialization routine

      character*20 s

      s = "Hello World!"
      x = 2.0
      i = 1

      call iniprint(s, x, i)

      end

      subroutine iniprint(fs, fx, fi)

      character*(*) fs
      integer fi

      print *, fs, fx, fi

      end
