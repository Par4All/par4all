      program format02

c     check that formats in declarations are OK if PRETTYPRINT_ALL_DECLARATIONS
C     is set

 1000 format('Hello world')

      real t(10)

      write(6, 1000)

      do i = 1, 10
         t(i) = 0.
      enddo

      end
