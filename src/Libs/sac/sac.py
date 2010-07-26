from pyps import *
import os

def string2file(string, fname):
    f = open(fname, "w")
    f.write(string)
    f.close()

def sac_workspace(sources, **args):
    # add SIMD.[ch] to the project
    string2file(simd_c, "SIMD.c")
    string2file(simd_h, "SIMD.h")
    return workspace(sources + ["SIMD.c"], **args)

def sac(module):

    ws = module._ws
    module.split_update_operator()

    # benchmark.tpips.h begin
    ws.activate("MUST_REGIONS")
    ws.activate("PRECONDITIONS_INTER_FULL")
    ws.activate("TRANSFORMERS_INTER_FULL")
    
    ws.set_property(RICEDG_STATISTICS_ALL_ARRAYS = True)
    ws.activate("RICE_SEMANTICS_DEPENDENCE_GRAPH")

    ws.set_property(SIMD_FORTRAN_MEM_ORGANISATION = False)
    ws.set_property(SAC_SIMD_REGISTER_WIDTH = 128)
    ws.set_property(SIMDIZER_AUTO_UNROLL_SIMPLE_CALCULATION = False)
    ws.set_property(SIMDIZER_AUTO_UNROLL_MINIMIZE_UNROLL = False)
    ws.set_property(PRETTYPRINT_ALL_DECLARATIONS = True)

    module.split_update_operator()
    
    module.if_conversion_init()
    module.if_conversion()
    module.if_conversion_compact()
    #module.use_def_elimination()

    module.partial_eval()
    module.simd_atomizer()

    module.simdizer_auto_unroll()
    module.partial_eval()
    module.clean_declarations()
    module.suppress_dead_code()
    #make DOTDG_FILE
    module.simd_remove_reductions()

    # module.deatomizer()
    # module.partial_eval()
    # module.use_def_elimination()
    # module.display()

    module.print_dot_dependence_graph()
    module.single_assignment()

    module.simdizer()

    # module.use_def_elimination()
    # module.display()

    module.simd_loop_const_elim()
    # setproperty EOLE_OPTIMIZATION_STRATEGY "ICM"
    # module.optimize_expressions()
    # module.partial_redundancy_elimination()

    # module.use_def_elimination()
    module.clean_declarations()
    module.suppress_dead_code()

module.sac = sac


def unincludeSIMD(fname):
    print "removing SIMD.h"
    # in the modulename.c file, undo the inclusion of SIMD.h by deleting
    # everything up to the definition of our function (not as clean as could
    # be, to say the least...)
    f = open(fname, "r")
    while not re.search("dotprod", f.readline()):
        pass
    contents = f.readlines()
    f.close()
    f = open(fname, "w")
    f.writelines(contents)
    f.close()

def addBeginning(fname, *args):
    contents = map((lambda(s): s + "\n" if s[-1] != "\n" else s),
                   args)
    
    f = open(fname, "r")
    contents += f.readlines()
    f.close()
    f = open(fname, "w")
    f.writelines(contents)
    f.close()

def reincludeSIMD(fname):
    print "include SIMD.h"
    addBeginning(fname, '#include "SIMD.h"')

def reincludestdio(fname):
    print "include stdio.h"
    addBeginning(fname, "#include <stdio.h>")

# Shouldn't we allow to easily add functions, in the same way that
# emacs does it with (add-hook HOOK FUN) / (remove-hook HOOK FUN) ?
# That would be easier for us...
def goingToRunWithFactory(old_goingToRunWith, *funs):
    def goingToRunWithAux(s, files, outdir):
        old_goingToRunWith(s, files, outdir)
        for fname in files:
            if re.search(r"SIMD\.c$", fname):
                continue
            for fun in funs:
                fun(fname)
    return goingToRunWithAux

def sac_compile(ws, **args):
    # compile, undoing the inclusion of SIMD.h
    old_goingToRunWith = workspace.goingToRunWith
    workspace.goingToRunWith = goingToRunWithFactory(old_goingToRunWith,
                                                     unincludeSIMD,
                                                     reincludeSIMD,
                                                     reincludestdio)
    ws.compile(**args)
    workspace.goingToRunWith = old_goingToRunWith

# if we want it for all compilations...
# pyps.workspace.compile = sac_compile

def addSSE(fname):
    print "adding sse.h"
    contents = [sse_h]
    f = open(fname)
    for line in f:
        line = re.sub(r"float (v4sf_[^[]+)", r"__m128 \1", line)
        line = re.sub(r"float (v4si_[^[]+)", r"__m128i \1", line)
        line = re.sub(r"v4s[if]_([^,[]+)\[[^]]*\]", r"\1", line)
        line = re.sub(r"v4s[if]_([^ ,[]+)", r"\1", line)
        line = re.sub(r"double (v2df_[^[]+)", r"__m128d \1", line)
        line = re.sub(r"double (v2di_[^[]+)", r"__m128i \1", line)
        line = re.sub(r"v2d[if]_([^,[]+)\[[^]]*\]", r"\1", line)
        line = re.sub(r"v2d[if]_([^ ,[]+)", r"\1", line)
        contents.append(line)
    f.close()
    f = open(fname, "w")
    f.writelines(contents)
    f.close()

def sac_compile_sse(ws, **args):
    # compile, undoing the inclusion of SIMD.h
    old_goingToRunWith = workspace.goingToRunWith
    workspace.goingToRunWith = goingToRunWithFactory(old_goingToRunWith,
                                                     unincludeSIMD,
                                                     addSSE)
    ws.compile(**args)
    workspace.goingToRunWith = old_goingToRunWith


# SIMD.c from validation/SAC/include/SIMD.c r2257
simd_c = """
#define LOGICAL int
#define DMAX(A,B) (A)>(B)?(A):(B)
void SIMD_LOAD_V4SI_TO_V4SF(float a[4], int b[4])
{
    b[0]=a[0];
    b[1]=a[1];
    b[2]=a[2];
    b[3]=a[3];
}
void SIMD_SAVE_V4SF_TO_V4SI(float a[4], int b[4])
{
    a[0]=b[0];
    a[1]=b[1];
    a[2]=b[2];
    a[3]=b[3];
}
void SIMD_SAVE_V2SF_TO_V2DF(double a[2],float b[2])
{
    a[0]=b[0];
    a[1]=b[1];
}
void SIMD_LOAD_V2SF_TO_V2DF(double a[2],float b[2])
{
    b[0]=a[0];
    b[1]=a[1];
}

int
PHI (LOGICAL L, int X1, int X2)
{
    return L ? X1 : X2;
}

void
SIMD_PHIW(int R[4], LOGICAL L[4], int X1[4], int X2[4])
{
    int i;
    for (i=0;i<2;i++)
        R[i]=L[i]?X1[i]:X2[i];
}

void
SIMD_GTD(int R[4], int X1[4], int X2[4])
{
    int i;
    for (i=0;i<4;i++)
        R[i]=X1[i]>X2[i];
}
void
SIMD_LOAD_V4SI (int VEC[4], int BASE[4])
{
    VEC[0] = BASE[0];
    VEC[1] = BASE[1];
    VEC[2] = BASE[2];
    VEC[3] = BASE[3];
}

void
SIMD_LOAD_V4SF (float VEC[4], float BASE[4])
{
    VEC[0] = BASE[0];
    VEC[1] = BASE[1];
    VEC[2] = BASE[2];
    VEC[3] = BASE[3];
}

void
SIMD_LOAD_V2DF (double VEC[2], double BASE[2])
{
    VEC[0] = BASE[0];
    VEC[1] = BASE[1];
}

void
SIMD_LOAD_GENERIC_V2DF (double VEC[2], double X0, double X1)
{
    VEC[0] = X0;
    VEC[1] = X1;
}
void
SIMD_LOAD_GENERIC_V4SI (int VEC[4], int X0, int X1, int X2, int X3)
{
    VEC[0] = X0;
    VEC[1] = X1;
    VEC[2] = X2;
    VEC[3] = X3;
}

void
SIMD_LOAD_GENERIC_V4SF (float VEC[4], float X0, float X1, float X2, float X3)
{
    VEC[0] = X0;
    VEC[1] = X1;
    VEC[2] = X2;
    VEC[3] = X3;
}

void
SIMD_LOAD_CONSTANT_V4SF (float VEC[4], float X0, float X1, float X2, float X3)
{

    VEC[0] = X0;
    VEC[1] = X1;
    VEC[2] = X2;
    VEC[3] = X3;
}

void
SIMD_LOAD_CONSTANT_V4SI (int VEC[4], int X0, int X1, int X2, int X3)
{

    VEC[0] = X0;
    VEC[1] = X1;
    VEC[2] = X2;
    VEC[3] = X3;
}

void
SIMD_SAVE_V4SI (int VEC[4], int BASE[4])
{  
    BASE[0] = VEC[0];
    BASE[1] = VEC[1];
    BASE[2] = VEC[2];
    BASE[3] = VEC[3];
}
void
SIMD_SAVE_V4SF (float VEC[4], float BASE[4])
{  
    BASE[0] = VEC[0];
    BASE[1] = VEC[1];
    BASE[2] = VEC[2];
    BASE[3] = VEC[3];
}
void
SIMD_SAVE_V2DF (double VEC[2], double BASE[2])
{  
    BASE[0] = VEC[0];
    BASE[1] = VEC[1];
}

void
SIMD_MASKED_SAVE_V4SF(float VEC[4], float BASE[3])
{  
    BASE[0] = VEC[0];
    BASE[1] = VEC[1];
    BASE[2] = VEC[2];
}


void
SIMD_SAVE_GENERIC_V2DF (double VEC[2], double X1[1], double X2[1])
{

    X1 [0]= VEC[0];
    X2 [0]= VEC[1];
}
void
SIMD_SAVE_GENERIC_V4SI (int VEC[4], int X1[1], int X2[1],
        int X3[1], int X4[1])
{

    X1 [0]= VEC[0];
    X2 [0]= VEC[1];
    X3 [0]= VEC[2];
    X4 [0]= VEC[3];
}
void
SIMD_SAVE_GENERIC_V4SF (float VEC[4], float X1[1], float X2[1],
        float X3[1], float X4[1])
{

    X1 [0]= VEC[0];
    X2 [0]= VEC[1];
    X3 [0]= VEC[2];
    X4 [0]= VEC[3];
}

void
SIMD_GTPS (LOGICAL DEST[4], float SRC1[4], float SRC2[4])
{
    DEST[0] = SRC1[0] > SRC2[0];
    DEST[1] = SRC1[1] > SRC2[1];
    DEST[2] = SRC1[2] > SRC2[2];
    DEST[3] = SRC1[3] > SRC2[3];
}
void
SIMD_GTPD (LOGICAL DEST[2], double SRC1[2], double SRC2[2])
{
    DEST[0] = SRC1[0] > SRC2[0];
    DEST[1] = SRC1[1] > SRC2[1];
}

void
SIMD_PHIPS (float DEST[4], LOGICAL COND[4], float SRC1[4], float SRC2[4])
{

    if (COND[0])
    {
        DEST[0] = SRC1[0];
    }
    else
    {
        DEST[0] = SRC2[0];
    }
    if (COND[1])
    {
        DEST[1] = SRC1[1];
    }
    else
    {
        DEST[1] = SRC2[1];
    }
    if (COND[2])
    {
        DEST[2] = SRC1[2];
    }
    else
    {
        DEST[2] = SRC2[2];
    }
    if (COND[3])
    {
        DEST[3] = SRC1[3];
    }
    else
    {
        DEST[3] = SRC2[3];
    }
}

void
SIMD_ADDPS (float DEST[4], float SRC1[4], float SRC2[4])
{
    DEST[0] = SRC1[0] + SRC2[0];
    DEST[1] = SRC1[1] + SRC2[1];
    DEST[2] = SRC1[2] + SRC2[2];
    DEST[3] = SRC1[3] + SRC2[3];
}

void
SIMD_SUBPS (float DEST[4], float SRC1[4], float SRC2[4])
{
    DEST[0] = SRC1[0] - SRC2[0];
    DEST[1] = SRC1[1] - SRC2[1];
    DEST[2] = SRC1[2] - SRC2[2];
    DEST[3] = SRC1[3] - SRC2[3];
}
void
SIMD_UMINPS (float DEST[4], float SRC1[4])
{
    DEST[0] =  - SRC1[0];
    DEST[1] =  - SRC1[1];
    DEST[2] =  - SRC1[2];
    DEST[3] =  - SRC1[3];
}

void
SIMD_MULPS (float DEST[4], float SRC1[4], float SRC2[4])
{
    DEST[0] = SRC1[0] * SRC2[0];
    DEST[1] = SRC1[1] * SRC2[1];
    DEST[2] = SRC1[2] * SRC2[2];
    DEST[3] = SRC1[3] * SRC2[3];
}
void
SIMD_DIVPD (double DEST[2], double SRC1[2], double SRC2[2])
{
    DEST[0] = SRC1[0] / SRC2[0];
    DEST[1] = SRC1[1] / SRC2[1];
}
void
SIMD_MULPD (double DEST[2], double SRC1[2], double SRC2[2])
{
    DEST[0] = SRC1[0] * SRC2[0];
    DEST[1] = SRC1[1] * SRC2[1];
}
#ifdef WITH_TRIGO
void
SIMD_SINPD (double DEST[2], double SRC1[2])
{
    DEST[0] = COS(SRC1[0]);
    DEST[1] = COS(SRC1[1]);
}
void
SIMD_COSPD (double DEST[2], double SRC1[2])
{
    DEST[0] = SIN(SRC1[0]);
    DEST[1] = SIN(SRC1[1]);
}
#endif
void
SIMD_ADDPD (double DEST[2], double SRC1[2], double SRC2[2])
{
    DEST[0] = SRC1[0] + SRC2[0];
    DEST[1] = SRC1[1] + SRC2[1];
}
void
SIMD_SUBPD (double DEST[2], double SRC1[2], double SRC2[2])
{
    DEST[0] = SRC1[0] - SRC2[0];
    DEST[1] = SRC1[1] - SRC2[1];
}

void
SIMD_DIVPS (float DEST[4], float SRC1[4], float SRC2[4])
{
    DEST[0] = SRC1[0] / SRC2[0];
    DEST[1] = SRC1[1] / SRC2[1];
    DEST[2] = SRC1[2] / SRC2[2];
    DEST[3] = SRC1[3] / SRC2[3];
}

void
SIMD_MAXPS (float DEST[4], float SRC1[4], float SRC2[4])
{
    DEST[0] = DMAX (SRC1[0], SRC2[0]);
    DEST[1] = DMAX (SRC1[1], SRC2[1]);
    DEST[2] = DMAX (SRC1[2], SRC2[2]);
    DEST[3] = DMAX (SRC1[3], SRC2[3]);
}

void
SIMD_LOAD_V2SI_TO_V2SF(int VEC[2], float TO[2])
{
    TO[0]=VEC[0];
    TO[1]=VEC[1];
}
void
SIMD_SAVE_V2SI_TO_V2SF(int TO[2], float VEC[2])
{
    TO[0]=VEC[0];
    TO[1]=VEC[1];
}


void
SIMD_LOAD_CONSTANT_V2SF (float VEC[2], float HIGH, float LOW)
{

    VEC[0] = LOW;
    VEC[1] = HIGH;
}
void
SIMD_LOAD_CONSTANT_V2SI (int VEC[2], int HIGH, int LOW)
{

    VEC[0] = LOW;
    VEC[1] = HIGH;
}

void
SIMD_LOAD_V2SI (int VEC[2], int BASE[2])
{  
    VEC[0] = BASE[0];
    VEC[1] = BASE[1];
}

void
SIMD_LOAD_GENERIC_V2SI (int VEC[2], int X1, int X2)
{

    VEC[0] = X1;
    VEC[1] = X2;
}

void
SIMD_SAVE_V2SI (int VEC[2], int BASE[2])
{  BASE[0] = VEC[0];
    BASE[1] = VEC[1];
}

void
SIMD_SAVE_GENERIC_V2SI (int VEC[2], int X1[1], int X2[1])
{

    X1 [0]= VEC[0];
    X2 [0]= VEC[1];
}


void
SIMD_SAVE_V2DI (int VEC[2], int BASE[2])
{  BASE[0] = VEC[0];
    BASE[1] = VEC[1];
}

void
SIMD_ADDW (short DEST[8], short SRC1[8], short SRC2[8])
{
    DEST[0] = SRC1[0] + SRC2[0];
    DEST[1] = SRC1[1] + SRC2[1];
    DEST[2] = SRC1[2] + SRC2[2];
    DEST[3] = SRC1[3] + SRC2[3];
    DEST[4] = SRC1[4] + SRC2[4];
    DEST[5] = SRC1[5] + SRC2[5];
    DEST[6] = SRC1[6] + SRC2[6];
    DEST[7] = SRC1[7] + SRC2[7];
}

void
SIMD_SUBW (short DEST[8], short SRC1[8], short SRC2[8])
{
    DEST[0] = SRC1[0] - SRC2[0];
    DEST[1] = SRC1[1] - SRC2[1];
    DEST[2] = SRC1[2] - SRC2[2];
    DEST[3] = SRC1[3] - SRC2[3];
    DEST[4] = SRC1[4] - SRC2[4];
    DEST[5] = SRC1[5] - SRC2[5];
    DEST[6] = SRC1[6] - SRC2[6];
    DEST[7] = SRC1[7] - SRC2[7];
}

void
SIMD_MULW (short DEST[8], short SRC1[8], short SRC2[8])
{
    DEST[0] = SRC1[0] * SRC2[0];
    DEST[1] = SRC1[1] * SRC2[1];
    DEST[2] = SRC1[2] * SRC2[2];
    DEST[3] = SRC1[3] * SRC2[3];
    DEST[4] = SRC1[4] * SRC2[4];
    DEST[5] = SRC1[5] * SRC2[5];
    DEST[6] = SRC1[6] * SRC2[6];
    DEST[7] = SRC1[7] * SRC2[7];
}

void
SIMD_DIVW (short DEST[8], short SRC1[8], short SRC2[8])
{
    DEST[0] = SRC1[0] / SRC2[0];
    DEST[1] = SRC1[1] / SRC2[1];
    DEST[2] = SRC1[2] / SRC2[2];
    DEST[3] = SRC1[3] / SRC2[3];
    DEST[4] = SRC1[4] / SRC2[4];
    DEST[5] = SRC1[5] / SRC2[5];
    DEST[6] = SRC1[6] / SRC2[6];
    DEST[7] = SRC1[7] / SRC2[7];
}

void
SIMD_LOAD_GENERIC_V8HI(short VEC[8], short BASE0, short BASE1, short BASE2, short BASE3, short BASE4, short BASE5, short BASE6, short BASE7)
{  
    VEC[0] = BASE0;
    VEC[1] = BASE1;
    VEC[2] = BASE2;
    VEC[3] = BASE3;
    VEC[4] = BASE4;
    VEC[5] = BASE5;
    VEC[6] = BASE6;
    VEC[7] = BASE7;
}

void
SIMD_LOAD_V8HI (short VEC[8], short BASE[8])
{  
    VEC[0] = BASE[0];
    VEC[1] = BASE[1];
    VEC[2] = BASE[2];
    VEC[3] = BASE[3];
    VEC[4] = BASE[4];
    VEC[5] = BASE[5];
    VEC[6] = BASE[6];
    VEC[7] = BASE[7];
}

void
SIMD_LOAD_V4QI_TO_V4HI (short VEC[4], char BASE[4])
{  VEC[0] = BASE[0];
    VEC[1] = BASE[1];
    VEC[2] = BASE[2];
    VEC[3] = BASE[3];
}


void
SIMD_SAVE_V8HI (short VEC[8], short BASE[8])
{  
    BASE[0] = VEC[0];
    BASE[1] = VEC[1];
    BASE[2] = VEC[2];
    BASE[3] = VEC[3];
    BASE[4] = VEC[4];
    BASE[5] = VEC[5];
    BASE[6] = VEC[6];
    BASE[7] = VEC[7];
}



void
SIMD_PHID (int DEST[4], LOGICAL COND[4], int SRC1[4], int SRC2[4])
{

    if (COND[0])
    {
        DEST[0] = SRC1[0];
    }
    else
    {
        DEST[0] = SRC2[0];
    }
    if (COND[1])
    {
        DEST[1] = SRC1[1];
    }
    else
    {
        DEST[1] = SRC2[1];
    }
    if (COND[2])
    {
        DEST[2] = SRC1[2];
    }
    else
    {
        DEST[2] = SRC2[2];
    }
    if (COND[3])
    {
        DEST[3] = SRC1[3];
    }
    else
    {
        DEST[3] = SRC2[3];
    }
}

void
SIMD_ADDD (int DEST[4], int SRC1[4], int SRC2[4])
{
    DEST[0] = SRC1[0] + SRC2[0];
    DEST[1] = SRC1[1] + SRC2[1];
    DEST[2] = SRC1[2] + SRC2[2];
    DEST[3] = SRC1[3] + SRC2[3];
}

void
SIMD_SUBD (int DEST[4], int SRC1[4], int SRC2[4])
{
    DEST[0] = SRC1[0] - SRC2[0];
    DEST[1] = SRC1[1] - SRC2[1];
    DEST[2] = SRC1[2] - SRC2[2];
    DEST[3] = SRC1[3] - SRC2[3];
}

void
SIMD_MULD (int DEST[4], int SRC1[4], int SRC2[4])
{
    DEST[0] = SRC1[0] * SRC2[0];
    DEST[1] = SRC1[1] * SRC2[1];
    DEST[2] = SRC1[2] * SRC2[2];
    DEST[3] = SRC1[3] * SRC2[3];
}
void
SIMD_DIVD (int DEST[4], int SRC1[4], int SRC2[4])
{
    DEST[0] = SRC1[0] / SRC2[0];
    DEST[1] = SRC1[1] / SRC2[1];
    DEST[2] = SRC1[2] / SRC2[2];
    DEST[3] = SRC1[3] / SRC2[3];
}

void
SIMD_LOAD_CONSTANT_V8QI (char VEC[8], int HIGH, int LOW)
{
    VEC[0] = (char) LOW;
    VEC[1] = (char) (LOW >> 1);
    VEC[2] = (char) (LOW >> 2);
    VEC[3] = (char) (LOW >> 3);
    VEC[4] = (char) HIGH;
    VEC[5] = (char) (HIGH >> 1);
    VEC[6] = (char) (HIGH >> 2);
    VEC[7] = (char) (HIGH >> 3);
}

void
SIMD_LOAD_V8QI (char VEC[8], char BASE[8])
{  VEC[0] = BASE[0];
    VEC[1] = BASE[1];
    VEC[2] = BASE[2];
    VEC[3] = BASE[3];
    VEC[4] = BASE[4];
    VEC[5] = BASE[5];
    VEC[6] = BASE[6];
    VEC[7] = BASE[7];
}

void
SIMD_LOAD_GENERIC_V8QI (char VEC[8], char X1,
        char X2, char X3, char X4, char X5, char X6,
        char X7, char X8)
{
    VEC[0] = X1;
    VEC[1] = X2;
    VEC[2] = X3;
    VEC[3] = X4;
    VEC[4] = X5;
    VEC[5] = X6;
    VEC[6] = X7;
    VEC[7] = X8;
}

void
SIMD_SAVE_V8QI (char VEC[8], char BASE[8])
{  BASE[0] = VEC[0];
    BASE[1] = VEC[1];
    BASE[2] = VEC[2];
    BASE[3] = VEC[3];
    BASE[4] = VEC[4];
    BASE[5] = VEC[5];
    BASE[6] = VEC[6];
    BASE[7] = VEC[7];
}

void
SIMD_SAVE_GENERIC_V8QI (char VEC[8], char *X0,
        char X1[1], char X2[1], char X3[1], char X4[1], char X5[1],
        char X6[1], char X7[1])
{

    X0[0] = VEC[0];
    X1[0] = VEC[1];
    X2[0] = VEC[2];
    X3[0] = VEC[3];
    X4[0] = VEC[4];
    X5[0] = VEC[5];
    X6[0] = VEC[6];
    X7[0] = VEC[7];
}

void
SIMD_ADDB (char DEST[8], char SRC1[8], char SRC2[8])
{
    DEST[0] = SRC1[0] + SRC2[0];
    DEST[1] = SRC1[1] + SRC2[1];
    DEST[2] = SRC1[2] + SRC2[2];
    DEST[3] = SRC1[3] + SRC2[3];
    DEST[4] = SRC1[4] + SRC2[4];
    DEST[5] = SRC1[5] + SRC2[5];
    DEST[6] = SRC1[6] + SRC2[6];
    DEST[7] = SRC1[7] + SRC2[7];
}

void
SIMD_SUBB (char DEST[8], char SRC1[8], char SRC2[8])
{
    DEST[0] = SRC1[0] - SRC2[0];
    DEST[1] = SRC1[1] - SRC2[1];
    DEST[2] = SRC1[2] - SRC2[2];
    DEST[3] = SRC1[3] - SRC2[3];
    DEST[4] = SRC1[4] - SRC2[4];
    DEST[5] = SRC1[5] - SRC2[5];
    DEST[6] = SRC1[6] - SRC2[6];
    DEST[7] = SRC1[7] - SRC2[7];
}

void
SIMD_MULB (char DEST[8], char SRC1[8], char SRC2[8])
{

    DEST[0] = SRC1[0] * SRC2[0];
    DEST[1] = SRC1[1] * SRC2[1];
    DEST[2] = SRC1[2] * SRC2[2];
    DEST[3] = SRC1[3] * SRC2[3];
    DEST[4] = SRC1[4] * SRC2[4];
    DEST[5] = SRC1[5] * SRC2[5];
    DEST[6] = SRC1[6] * SRC2[6];
    DEST[7] = SRC1[7] * SRC2[7];
}

void
SIMD_MOVPS (float DEST[2], float SRC[2])
{
    DEST[0] = SRC[0];
    DEST[1] = SRC[1];
}

void
SIMD_MOVD (int DEST[2], int SRC[2])
{
    DEST[0] = SRC[0];
    DEST[1] = SRC[1];
}

void
SIMD_MOVW (short DEST[4], short SRC[4])
{
    DEST[0] = SRC[0];
    DEST[1] = SRC[1];
    DEST[2] = SRC[2];
    DEST[3] = SRC[3];
}

void
SIMD_MOVB (char DEST[8], char SRC[8])
{

    DEST[0] = SRC[0];
    DEST[1] = SRC[1];
    DEST[2] = SRC[2];
    DEST[3] = SRC[3];
    DEST[4] = SRC[4];
    DEST[5] = SRC[5];
    DEST[6] = SRC[6];
    DEST[7] = SRC[7];
}

void
SIMD_OPPPS (float DEST[2], float SRC[2])
{
    DEST[0] = -SRC[0];
    DEST[1] = -SRC[1];
}

void
SIMD_OPPD (int DEST[2], int SRC[2])
{
    DEST[0] = -SRC[0];
    DEST[1] = -SRC[1];
}

void
SIMD_OPPW (short DEST[4], short SRC[4])
{
    DEST[0] = -SRC[0];
    DEST[1] = -SRC[1];
    DEST[2] = -SRC[2];
    DEST[3] = -SRC[3];
}

void
SIMD_OPPB (char DEST[8], char SRC[8])
{
    DEST[0] = -SRC[0];
    DEST[1] = -SRC[1];
    DEST[2] = -SRC[2];
    DEST[3] = -SRC[3];
    DEST[4] = -SRC[4];
    DEST[5] = -SRC[5];
    DEST[6] = -SRC[6];
    DEST[7] = -SRC[7];
}

void
SIMD_SETPS (float DEST[2], float SRC[2])
{
    DEST[0] = SRC[0];
    DEST[1] = SRC[1];
}

void
SIMD_SETD (int DEST[2], int SRC[2])
{
    DEST[0] = SRC[0];
    DEST[1] = SRC[1];
    DEST[2] = SRC[2];
    DEST[3] = SRC[3];
}

void
SIMD_SETW (short DEST[4], short SRC[4])
{
    DEST[0] = SRC[0];
    DEST[1] = SRC[1];
}

void
SIMD_SETB (char DEST[8], char SRC[8])
{
    DEST[0] = SRC[0];
    DEST[1] = SRC[1];
    DEST[2] = SRC[2];
    DEST[3] = SRC[3];
    DEST[4] = SRC[4];
    DEST[5] = SRC[5];
    DEST[6] = SRC[6];
    DEST[7] = SRC[7];
}

void
SIMD_LOAD_CONSTANT_V2DF(double vec[2],double v0,double v1)
{
    vec[0]=v0;
    vec[1]=v1;
}

#undef LOGICAL
#undef DMAX
"""

simd_h = """
/* SIMD.c */
int PHI(int L, int X1, int X2);
void SIMD_PHIW(int R[4], int L[4], int X1[4], int X2[4]);
void SIMD_GTD(int R[4], int X1[4], int X2[4]);
void SIMD_LOAD_V4SI(int VEC[4], int BASE[4]);
void SIMD_LOAD_V4SF(float VEC[4], float BASE[4]);
void SIMD_LOAD_V2DF(double VEC[2], double BASE[2]);
void SIMD_LOAD_GENERIC_V2DF(double VEC[2], double X0, double X1);
void SIMD_LOAD_GENERIC_V4SI(int VEC[4], int X0, int X1, int X2, int X3);
void SIMD_LOAD_GENERIC_V4SF(float VEC[4], float X0, float X1, float X2, float X3);
void SIMD_LOAD_CONSTANT_V4SF(float VEC[4], float X0, float X1, float X2, float X3);
void SIMD_LOAD_CONSTANT_V4SI(int VEC[4], int X0, int X1, int X2, int X3);
void SIMD_SAVE_V4SI(int VEC[4], int BASE[4]);
void SIMD_SAVE_V4SF(float VEC[4], float BASE[4]);
void SIMD_SAVE_V2DF(double VEC[2], double BASE[2]);
void SIMD_MASKED_SAVE_V4SF(float VEC[4], float BASE[3]);
void SIMD_SAVE_GENERIC_V2DF(double VEC[2], double X1[1], double X2[1]);
void SIMD_SAVE_GENERIC_V4SI(int VEC[4], int X1[1], int X2[1], int X3[1], int X4[1]);
void SIMD_SAVE_GENERIC_V4SF(float VEC[4], float X1[1], float X2[1], float X3[1], float X4[1]);
void SIMD_GTPS(int DEST[4], float SRC1[4], float SRC2[4]);
void SIMD_GTPD(int DEST[2], double SRC1[2], double SRC2[2]);
void SIMD_PHIPS(float DEST[4], int COND[4], float SRC1[4], float SRC2[4]);
void SIMD_ADDPS(float DEST[4], float SRC1[4], float SRC2[4]);
void SIMD_SUBPS(float DEST[4], float SRC1[4], float SRC2[4]);
void SIMD_UMINPS(float DEST[4], float SRC1[4]);
void SIMD_MULPS(float DEST[4], float SRC1[4], float SRC2[4]);
void SIMD_DIVPD(double DEST[2], double SRC1[2], double SRC2[2]);
void SIMD_MULPD(double DEST[2], double SRC1[2], double SRC2[2]);
void SIMD_ADDPD(double DEST[2], double SRC1[2], double SRC2[2]);
void SIMD_SUBPD(double DEST[2], double SRC1[2], double SRC2[2]);
void SIMD_DIVPS(float DEST[4], float SRC1[4], float SRC2[4]);
void SIMD_MAXPS(float DEST[4], float SRC1[4], float SRC2[4]);
void SIMD_LOAD_V2SI_TO_V2SF(int VEC[2], float TO[2]);
void SIMD_SAVE_V2SI_TO_V2SF(int TO[2], float VEC[2]);
void SIMD_LOAD_CONSTANT_V2SF(float VEC[2], float HIGH, float LOW);
void SIMD_LOAD_CONSTANT_V2SI(int VEC[2], int HIGH, int LOW);
void SIMD_LOAD_V2SI(int VEC[2], int BASE[2]);
void SIMD_LOAD_GENERIC_V2SI(int VEC[2], int X1, int X2);
void SIMD_SAVE_V2SI(int VEC[2], int BASE[2]);
void SIMD_SAVE_GENERIC_V2SI(int VEC[2], int X1[1], int X2[1]);
void SIMD_SAVE_V2DI(int VEC[2], int BASE[2]);
void SIMD_ADDW(short DEST[8], short SRC1[8], short SRC2[8]);
void SIMD_SUBW(short DEST[8], short SRC1[8], short SRC2[8]);
void SIMD_MULW(short DEST[8], short SRC1[8], short SRC2[8]);
void SIMD_DIVW(short DEST[2], short SRC1[2], short SRC2[2]);
void SIMD_LOAD_GENERIC_V8HI(short VEC[8], short BASE0, short BASE1, short BASE2, short BASE3, short BASE4, short BASE5, short BASE6, short BASE7);
void SIMD_LOAD_V8HI(short VEC[8], short BASE[8]);
void SIMD_LOAD_V4QI_TO_V4HI(short VEC[4], char BASE[4]);
void SIMD_SAVE_V8HI(short VEC[8], short BASE[8]);
void SIMD_PHID(int DEST[4], int COND[4], int SRC1[4], int SRC2[4]);
void SIMD_ADDD(int DEST[4], int SRC1[4], int SRC2[4]);
void SIMD_SUBD(int DEST[4], int SRC1[4], int SRC2[4]);
void SIMD_MULD(int DEST[4], int SRC1[4], int SRC2[4]);
void SIMD_DIVD(int DEST[4], int SRC1[4], int SRC2[4]);
void SIMD_LOAD_CONSTANT_V8QI(char VEC[8], int HIGH, int LOW);
void SIMD_LOAD_V8QI(char VEC[8], char BASE[8]);
void SIMD_LOAD_GENERIC_V8QI(char VEC[8], char X1, char X2, char X3, char X4, char X5, char X6, char X7, char X8);
void SIMD_SAVE_V8QI(char VEC[8], char BASE[8]);
void SIMD_SAVE_GENERIC_V8QI(char VEC[8], char *X0, char X1[1], char X2[1], char X3[1], char X4[1], char X5[1], char X6[1], char X7[1]);
void SIMD_ADDB(char DEST[8], char SRC1[8], char SRC2[8]);
void SIMD_SUBB(char DEST[8], char SRC1[8], char SRC2[8]);
void SIMD_MULB(char DEST[8], char SRC1[8], char SRC2[8]);
void SIMD_MOVPS(float DEST[2], float SRC[2]);
void SIMD_MOVD(int DEST[2], int SRC[2]);
void SIMD_MOVW(short DEST[4], short SRC[4]);
void SIMD_MOVB(char DEST[8], char SRC[8]);
void SIMD_OPPPS(float DEST[2], float SRC[2]);
void SIMD_OPPD(int DEST[2], int SRC[2]);
void SIMD_OPPW(short DEST[4], short SRC[4]);
void SIMD_OPPB(char DEST[8], char SRC[8]);
void SIMD_SETPS(float DEST[2], float SRC[2]);
void SIMD_SETD(int DEST[2], int SRC[2]);
void SIMD_SETW(short DEST[4], short SRC[4]);
void SIMD_SETB(char DEST[8], char SRC[8]);
void SIMD_LOAD_CONSTANT_V2DF(double vec[2], double v0, double v1);

"""

sse_h = """
#include <xmmintrin.h>

/* extras */
#define MOD(a,b) ((a)%(b))
#define MAX0(a,b) ((a)>(b)?(a):(b))

/* float */
#define SIMD_LOAD_V4SF(vec,arr) vec=_mm_loadu_ps(arr)
#define SIMD_MULPS(vec1,vec2,vec3) vec1=_mm_mul_ps(vec2,vec3)
#define SIMD_ADDPS(vec1,vec2,vec3) vec1=_mm_add_ps(vec2,vec3)
#define SIMD_SAVE_V4SF(vec,arr) _mm_storeu_ps(arr,vec)
#define SIMD_SAVE_GENERIC_V4SF(vec,v0,v1,v2,v3) \
do { \
    float tmp[4]; \
    SIMD_SAVE_V4SF(vec,&tmp[0]);\
    *v0=tmp[0];\
    *v1=tmp[1];\
    *v2=tmp[2];\
    *v3=tmp[3];\
} while (0)
#define SIMD_LOAD_GENERIC_V4SF(vec,v0,v1,v2,v3)\
do { \
    float v[4] = { v0,v1,v2,v3 };\
    SIMD_LOAD_V4SF(vec,&v[0]); \
} while(0)

/* handle padded value, this is a very bad implementation ... */
#define SIMD_MASKED_SAVE_V4SF(vec,arr) do { float tmp[4] ; SIMD_SAVE_V4SF(vec,&tmp[0]); (arr)[0]=tmp[0];(arr)[1]=tmp[1];(arr)[2]=tmp[2]; } while(0)


/* double */
#define SIMD_LOAD_V2DF(vec,arr) vec=_mm_loadu_pd(arr)
#define SIMD_MULPD(vec1,vec2,vec3) vec1=_mm_mul_pd(vec2,vec3)
#define SIMD_ADDPD(vec1,vec2,vec3) vec1=_mm_add_pd(vec2,vec3)
#define SIMD_SAVE_V2DF(vec,arr) _mm_storeu_pd(arr,vec)
#define SIMD_SAVE_GENERIC_V2DF(vec,v0,v1) \
do { \
    double tmp[2]; \
    SIMD_SAVE_V2DF(vec,&tmp[0]);\
    *(v0)=tmp[0];\
    *(v1)=tmp[1];\
} while (0)
#define SIMD_LOAD_GENERIC_V2DF(vec,v0,v1)\
do { \
    double v[2] = { v0,v1};\
    SIMD_LOAD_V2DF(vec,&v[0]); \
} while(0)

/* conversions */
#define SIMD_SAVE_V2SF_TO_V2DF(vec,f) \
    SIMD_SAVE_GENERIC_V2DF(vec,(f),(f)+1)
#define SIMD_LOAD_V2SF_TO_V2DF(vec,f) \
    SIMD_LOAD_GENERIC_V2DF(vec,(f)[0],(f)[1])


"""
