      program unstruc16

C     Triplication of a cycle: modified version of unstruc14 to hide the
C     cycle breaking node "j=j+1". This time, the internal cycle on J is
C     triplicated and a complicated embeeding DAG appears. However, the
C     postcondition is not modified as could be expected.

C     Modification: J is now read with N

C     Remark: the intended behavior probably was to start the loop with J
C     equals 1 or 2. The go to targets in the first test are probably wrong.

      read *, n, j
      i = 1
      if(n.gt.0) then
         j = 2
         go to 100
      else
C        J is not (re-)initialized
         go to 200
      endif

 100  continue
      j = 1
 200  continue
 300  continue
      j = j + 1
      if(j.lt.10) go to 300
      if(i+j.gt.105) go to 400
      i = i + 1
      if(i.lt.100) go to 100

 400  continue
C     If N is negative: If the initial value of J is greater than or
C     equal to 104 and N is negative, this statement is reached with I=1
C     and J=J#init+1. Else if the initial value of J is less than 9,
C     I=96 and J=10. Else, J is reinitialized and we end up with I=96 and J=10.

C     If N is strictly positive, we end up with J=10 and I=96 since J is always
C     initialized to 1.

      print *, i, j

      end
