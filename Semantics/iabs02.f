      program iabs02

C     Non trivial and non-linear expressions

      read *, n

      i = iabs(n+m/2)

      print *, i

      read *, j

      j = iabs(i*(i+1))

      print *, j

      if(j.ne.3) stop

      k = iabs(j*(i+1))

      print *, k

      end
