;;;; Pattern-matching process for generalized reductions.

(require :init)
(require :ri)
(require :reduction)
(require :util)

(use-package '(:newgen :ri :reduction))

;;; Creation of a pattern instance from a symbolic expression.

(defun create-instance (var sexp) 
    (let ((gexps (sexpression-gexpressions sexp))) 
        (case (length gexps)  
	      (1 (create-simple-instance var (car gexps))) 
	      (2 (let* ((one (first gexps)) 
			(g-one (gexpression-guard one))) 
		     (assert (expression-call-p g-one)) 
		     (create-simple-instance  
		      var  
		      (if (entity-equal-p  
			   (expression-call-function g-one) 
			   (function-entity[] ".NOT.")) 
			      (second gexps) 
			  one))))
	      (otherwise nil))))

(defun create-simple-instance (var gexp)
    (multiple-value-bind
     (op param) (extract-op-param var (gexpression-expression gexp))
     (make-pattern :variable var
		   :condition (gexpression-guard gexp)
		   :parameter param
		   :operator op
		   :indices '(:dont-care))))

;; This only works on calls if the operator arity is 2 (e.g., +, *)
;; We also assume that EXP is normalized (think: I = I+I).
;; We use EXPRESSION[GENSYM], but any would work fine.
;;
(defun extract-op-param (var exp)
    (gen-switch
     (expression-syntax exp)
     ((is-syntax-reference r)
      (if (entity-equal-p var (reference-variable r))
	      (values entity[fun-xy.x] 
		      (make-expression[] 0 "TOP-LEVEL:0"))
	  (values entity[fun-xy.y] exp)))
     ((is-syntax-call c)
      (let* ((f (call-function c)) 
	     (val (entity-initial f))
	     (rests (extract-op-call var exp f)))
	  (cond ((and (endp (call-arguments c))
		      (not (value-code-p val))
		      (not (value-unknown-p val)))
		 (values entity[fun-xy.y] exp))
		(t (assert (not (endp rests)) () "Non-normalized expression")
		   (values (if (in-term-for-op var exp f)
				   f
			       entity[fun-xy.y])
			   (reduce #'(lambda (exp res)
					 (make-expression-call f `(,exp ,res)))
				   (cdr rests)
				   :initial-value (car rests)))))))))

(defun extract-op-call (var exp op)
    (gen-recurse
     exp
     ((expression syntax) syntax)
     ((syntax tag) tag)
     ((reference)
      (if (entity-equal-p (reference-variable reference) var)
	      '()
	  `(,(make-expression-reference (reference-variable reference)
					(reference-indices reference)))))
     ((call arguments)
      (if (entity-equal-p op (call-function call))
	      (apply #'nconc arguments)
	  `(,(make-expression-call (call-function call)
				   (call-arguments call)))))))

(defun in-term-for-op (var exp op)
    (gen-recurse
     exp
     ((expression syntax) syntax)
     ((syntax tag) tag)
     ((reference)
      (entity-equal-p (reference-variable reference) var))
     ((call arguments)
      (if (entity-equal-p op (call-function call))
	      (some #'identity arguments)
	  nil))))


;;; Pattern-matching.

(defun match-pattern (instance index)
    (flet ((expressize (entity)
		       (make-expression-reference entity '())))
	(some 
	 #'(lambda (template)
	       (let* ((pat (template-pattern template))
		      (val (template-value template))
		      (gen (template-generator template))
		      (ivar-exp (expressize (pattern-variable instance)))
		      (pvar-exp (expressize (pattern-variable pat)))
		      (binding
		       (reduce #'match-instance-pattern
			       `((,ivar-exp . ,pvar-exp)
				 (,(pattern-condition instance) .
				  ,(pattern-condition pat))
				 (,(expressize 
				    (pattern-operator instance)) .
				  ,(expressize 
				    (pattern-operator pat)))
				 (,(pattern-parameter instance) .
				  ,(pattern-parameter pat)))
			       :initial-value (init-bindings))))
		   (and (success-binding-p binding)
			(every #'(lambda (ind)
				     (expression-equal-p
				      (get-bindings ind binding)
				      (expressize index)))
			       (pattern-indices pat))
			(let ((new-val (result-match val binding)))
			    (make-template :pattern pat
					   :generator gen
					   :value new-val)))))
	 *reductions*)))

(defun match-instance-pattern (binding instance-pattern)
    (if (success-binding-p binding)
	    (match-expression
	     (car instance-pattern)
	     (cdr instance-pattern)
	     binding)
	binding))

(defun match-expression (instance pattern binding)
    (funcall
     (gen-recurse
      pattern
      ((expression syntax) syntax)
      ((syntax tag) tag)
      ((reference indices)
       #'(lambda (exp)
	     (cond ((and (expression-reference-p exp)
			 (= (length (reference-indices reference))
			    (length (expression-reference-indices exp))))
		    (setf binding
			  (update-bindings 
			   (reference-variable reference)
			   (make-expression-reference
			    (expression-reference-variable exp)
			    '())
			   binding))
		    (every #'funcall 
			   indices 
			   (expression-reference-indices exp)))
		   (t
		    (return-from match-expression failure-binding)))))
      ((call arguments)
       #'(lambda (exp)
	     (cond ((and (expression-call-p exp)
			 (= (length (call-arguments call))
			    (length (expression-call-arguments exp))))
		    (setf binding
			  (update-bindings 
			   (call-function call)
			   (make-expression-reference
			    (expression-call-function exp)
			    '())
			   binding))
		    (every #'funcall 
			   arguments 
			   (expression-call-arguments exp)))
		   (t
		    (return-from match-expression failure-binding))))))
     instance)
    binding)

(defun result-match (exp binding)
    (gen-recurse
     exp
     ((expression syntax) syntax)
     ((syntax tag) tag)
     ((reference indices)
      (let ((exp-ref (get-bindings (reference-variable reference)
				   binding)))
	  (make-expression-reference 
	   (if exp-ref 
		   (expression-reference-variable exp-ref)
	       (reference-variable reference))
	   indices)))
     ((call arguments)
      (let ((exp-ref (get-bindings (call-function call) binding)))
	  (make-expression-call 
	   (if exp-ref
		   (expression-reference-variable exp-ref)
	       (call-function call))
	   arguments)))))

;;; Management of unification bindings

(defun init-bindings () '())

(defvar failure-binding :failed)

(defun success-binding-p (b)
	(not (eq b failure-binding)))

(defun get-bindings (var bindings)
    (cond ((unification-variable-p var)
	   (let ((val (assoc var bindings :test #'equal)))
	       (if val
		       (cdr val)
		   nil)))
	  (t nil)))

(defun update-bindings (var exp bindings)
    (cond ((not (success-binding-p bindings))
	   failure-binding)
	  ((unification-variable-p var)
	   (let ((val (get-bindings var bindings)))
	       (cond (val
		      (if (expression-equal-p val exp)
			      bindings
			  failure-binding))
		     ((and (expression-reference-p exp)
			   (entity-equal-p (expression-reference-variable exp)
					   entity[gensym]))
		      failure-binding)
		     (t 
		      (acons var exp bindings)))))
	  ((entity-equal-p var entity[gensym])
	   failure-binding)
	  (t (if (and (expression-reference-p exp)
		      (entity-equal-p var
				      (expression-reference-variable exp)))
		     bindings
		 failure-binding))))
			  
;;;

(provide :match)

				
					
				    
