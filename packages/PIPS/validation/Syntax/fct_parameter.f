      SUBROUTINE DUMMY
      print *, 'bing'
      END

      SUBROUTINE FCT_PARAMETER (TOTO)
      CALL TOTO()
      RETURN
      END

      PROGRAM TEST
      EXTERNAL DUMMY
      CALL FCT_PARAMETER (DUMMY)
      END
