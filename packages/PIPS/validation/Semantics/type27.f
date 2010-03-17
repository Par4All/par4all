      program type27

C     Check multi-type initialization routine

      character*20 s

      call init(s, x, i)

      call iniprint(s, x, i)

      end

      subroutine iniprint(fs, fx, fi)

      character*(*) fs
      integer fi

      print *, fs, fx, fi

      end

      subroutine init(fs, fx, fi)

      character*(*) fs
      integer fi

      fs = "Hello World!"
      fx = 2.0
      fi = 1

      end
