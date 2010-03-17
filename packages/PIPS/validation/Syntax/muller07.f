      program muller07
      character*3 HEAD

      PARAMETER ( N    = 1 )

      COMMON / CONSTN / HEAD(N)
      HEAD(N)='a'
      print *, HEAD
      end
