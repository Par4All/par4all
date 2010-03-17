      program format01

c     check that formats in declarations are NOK if Property
c     PRETTYPRINT_ALL_DECLARATIONS is not set

 1000 format('Hello world')

      real t(10)

      write(6, 1000)

      do i = 1, 10
         t(i) = 0.
      enddo

      end
