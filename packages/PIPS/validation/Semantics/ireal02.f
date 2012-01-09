C     Bug found in ireal01. The bug disappear if the first statement is
C     removed... or if SEMANTICS_DEBUG_LEVEL is greater than 1...

      program ireal02

      I = MIN(1.2,K)
      LWIDTH=MIN(MAX(IFIELD,72),132)

      end
