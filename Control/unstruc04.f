      program unstruc04

C     Check special case of loop with a unique controle node

      i = 0
 1    if(inc(i).lt.10) go to 1
      print *, i
      end

      integer function inc(i)
      i = i + 1
      inc = i
      end
