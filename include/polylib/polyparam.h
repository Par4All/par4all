#ifndef _polyparam_H_
#define _polyparam_H_

#if defined(__cplusplus)
extern "C" {
#endif

extern void Compute_PDomains ( Param_Domain *PD, int nb_domains, int
                               working_space );
extern Param_Polyhedron *GenParamPolyhedron(Polyhedron *Pol, Matrix *Rays);
extern void Param_Domain_Free (Param_Domain *PD);
extern void Param_Polyhedron_Free ( Param_Polyhedron *P );
extern void Param_Vertices_Free ( Param_Vertices *PV );
extern void Param_Vertices_Print(FILE *DST, Param_Vertices *PV,
                                   const char **param_names);
extern Polyhedron *PDomainDifference ( Polyhedron *Pol1, Polyhedron
                                       *Pol2, unsigned NbMaxRays );
extern Polyhedron *PDomainIntersection ( Polyhedron *Pol1, Polyhedron
                                         *Pol2, unsigned NbMaxRays );
extern Param_Polyhedron *Polyhedron2Param_Domain ( Polyhedron *Din,
                                                   Polyhedron *Cin, int
                                                   working_space );
extern Param_Polyhedron *Polyhedron2Param_SimplifiedDomain (
   Polyhedron **Din, Polyhedron *Cin, int working_space,
   Polyhedron **CEq, Matrix **CT );
extern Param_Polyhedron *Polyhedron2Param_Vertices ( Polyhedron *Din,
                                                     Polyhedron *Cin, int
                                                     working_space );
extern void Print_Domain(FILE *DST, Polyhedron *D, const char **param_names);
extern void Print_Vertex(FILE *DST, Matrix *V, const char **param_names);
extern Matrix *VertexCT( Matrix *V, Matrix *CT );
void Param_Polyhedron_Scale_Integer(Param_Polyhedron *PP, Polyhedron **P,
				    Value *det, unsigned MaxRays);

#if defined(__cplusplus)
}
#endif

#endif /* _polyparam_H_ */
