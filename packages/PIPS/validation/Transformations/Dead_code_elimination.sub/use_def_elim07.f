C      Check that effect on static variables are preserved, at least
C      when they escape
C
C     If the line j = j + c is removed, c = c + 1 becomes useless...

       SUBROUTINE COUNT(J)
       integer C
       data C /0/
       save C
       J = J + C
       C = C + 1
       RETURN
       END

       SUBROUTINE LOST_COUNT(J)
       integer C
       data C /0/
       save C
       C = C + 1
       RETURN
       END

       PROGRAM HELLOW
       CALL COUNT(J)
       CALL LOST_COUNT(J)
       STOP
       END


