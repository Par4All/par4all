C	--- -----------------------------------------------------------
C	--- Fichier d'include de declaration des "common" de la taille
C	--- des zones memoires et des indices des tableaux dans 
C	--- ces zones.						
C	--- -----------------------------------------------------------

C	--- -----------------------------------------------------------
C	CARL	ZIMP, ATG, SUSO, SUSE
C	Nx1	IPCO, IPCE, ICCO, ICCE, NRLO, NRLE, NRGR, NRCO, NRTJ
C	NB1	NRLO, NRLE, NRTJ
C	Nx2	IPTAI, NTAI
C	Nx3	LHAC, NCHA, EHA
C	Nx4	IVOL, IVEL, IVGR, IVCO, IVTM
C	Nx5	IVAC, LITA, ITMF, NXUT
C	Nx6	MVOL, MVEL, MVGR, MVCO, MVTV
C	M_GAIN	LCHBG, LORD, IDEB, NONU, ISUI, INDL, ELM
C	NA7	KZBY, NZBY
C	--- -----------------------------------------------------------
c
c********************
c       character*40 sccs_id/'@(#)indice.inc 1.1 94/05/13\n'/
c********************
c

	Integer*4	I_ATG, I_SUSO, I_SUSE
	Integer*4	I_IPCO, I_IPCE, I_ICCO, I_ICCE
	Integer*4	I_NRLO, I_NRLE, I_NRGR, I_NRCO, I_NRTJ
	Integer*4	IB_NRLO, IB_NRLE, IB_NRTJ
	Integer*4	I_IPTAI, I_NTAI
	Integer*4	I_LHAC, I_NCHA, I_EHA
	Integer*4	I_IVOL, I_IVEL, I_IVGR, I_IVCO, I_IVTM
	Integer*4 	I_LCHBG, I_LORD, I_IDEB, I_NONU, I_ISUI
	Integer*4	I_INDL
	Integer*4	I_ELM
	Integer*4	I_IVAC, I_LITA, I_ITMF, I_NXUT, I_ITMF_P
	Integer*4	I_NXUT_P
	Integer*4	I_KZBY, I_NZBY
	Integer*4	I_TENS

	Common/I_CARL/	I_ATG, I_SUSO, I_SUSE
	Common/I_ESWE/	I_IPCO, I_IPCE, I_ICCO, I_ICCE,
     +			I_NRLO, I_NRLE, I_NRGR, I_NRCO, I_NRTJ
	Common/I_ESWC/	IB_NRLO, IB_NRLE, IB_NRTJ
	Common/I_ESWI/	I_IPTAI, I_NTAI
	Common/I_ESWA/	I_LHAC, I_NCHA, I_EHA
	Common/I_PTMA/	I_IVOL, I_IVEL, I_IVGR, I_IVCO, I_IVTM
	Common/I_GAIN/	I_LCHBG, I_LORD, I_IDEB, I_NONU, I_ISUI,
     +			I_INDL, I_ELM
	Common/I_ESWL/	I_IVAC, I_LITA, I_ITMF, I_NXUT, I_ITMF_P,
     +			I_NXUT_P
	Common/I_LRD/	I_KZBY, I_NZBY
	Common/I_STN/	I_TENS
