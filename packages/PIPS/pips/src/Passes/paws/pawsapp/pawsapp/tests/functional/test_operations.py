from pawsapp.tests import *

class TestOperationsController(TestController):

    def test_perform_preconditions(self):
	test_code = file('pawsapp/tests/functional/codes/test.c').read()
	result_code = file('pawsapp/tests/functional/results/result_operations_preconditions.txt').read().strip()
	resp = self.app.get(url(controller='operations', action='get_directory'))
	assert 'directory' in resp.session
	response = self.app.get(url(controller='operations', action='perform'), 
				params={
					'code': test_code,
					'language': 'C',
					'operation': 'preconditions'
				})
        assert result_code in response

    def test_perform_preconditions_advanced(self):
	test_code = file('pawsapp/tests/functional/codes/test_advanced.c').read()
	result_code = file('pawsapp/tests/functional/results/result_operations_preconditions_advanced.txt').read().strip()
	phases = 'refine_transformers true;'	
	analyses = 'preconditions_intra;summary_precondition;'
	properties = 'SEMANTICS_K_FIX_POINT 2;PRETTYPRINT_ANALYSES_WITH_LF true;SEMANTICS_ANALYZE_SCALAR_BOOLEAN_VARIABLES true;SEMANTICS_ANALYZE_SCALAR_COMPLEX_VARIABLES false;SEMANTICS_ANALYZE_SCALAR_FLOAT_VARIABLES false;SEMANTICS_ANALYZE_SCALAR_STRING_VARIABLES false;SEMANTICS_ANALYZE_SCALAR_INTEGER_VARIABLES true;SEMANTICS_COMPUTE_TRANSFORMERS_IN_CONTEXT false;SEMANTICS_FILTERED_PRECONDITIONS true;SEMANTICS_TRUST_ARRAY_DECLARATIONS false;SEMANTICS_TRUST_ARRAY_REFERENCES false;SEMANTICS_USE_TRANSFORMER_LISTS true;SEMANTICS_USE_LIST_PROJECTION true;'
	resp = self.app.get(url(controller='operations', action='get_directory'))
	assert 'directory' in resp.session
        response = self.app.get(url(controller='operations', action='perform_advanced'), 
				params={
					'code': test_code,
					'language': 'C',
					'operation': 'preconditions',
					'analyses': analyses,					
					'properties': properties,
					'phases': phases
				})
        assert result_code in response

    def test_perform_openmp(self):
	test_code = file('pawsapp/tests/functional/codes/test.c').read()
	result_code = file('pawsapp/tests/functional/results/result_operations_openmp.txt').read().strip()
	resp = self.app.get(url(controller='operations', action='get_directory'))
	assert 'directory' in resp.session
	response = self.app.get(url(controller='operations', action='perform'), 
				params={
					'code': test_code,
					'language': 'C',
					'operation': 'openmp'
				})
	assert result_code in response

    def test_perform_regions(self):
	test_code = file('pawsapp/tests/functional/codes/test.c').read()
	result_code = file('pawsapp/tests/functional/results/result_operations_regions.txt').read().strip()
	resp = self.app.get(url(controller='operations', action='get_directory'))
	assert 'directory' in resp.session
        response = self.app.get(url(controller='operations', action='perform'), 
				params={
					'code': test_code,
					'language': 'C',
					'operation': 'regions'
				})
        assert result_code in response

    def test_perform_in_regions(self):
	test_code = file('pawsapp/tests/functional/codes/test.c').read()
	result_code = file('pawsapp/tests/functional/results/result_operations_in_regions.txt').read().strip()
	resp = self.app.get(url(controller='operations', action='get_directory'))
	assert 'directory' in resp.session
	response = self.app.get(url(controller='operations', action='perform'), 
				params={
					'code': test_code,
					'language': 'C',
					'operation': 'in_regions'
				})
	assert result_code in response

    def test_perform_out_regions(self):
	test_code = file('pawsapp/tests/functional/codes/test.c').read()
	result_code = file('pawsapp/tests/functional/results/result_operations_out_regions.txt').read().strip()
	resp = self.app.get(url(controller='operations', action='get_directory'))
	assert 'directory' in resp.session
	response = self.app.get(url(controller='operations', action='perform'), 
				params={
					'code': test_code,
					'language': 'C',
					'operation': 'out_regions'
				})
	assert result_code in response

    def test_perform_multiple_preconditions(self):
	result_code = file('pawsapp/tests/functional/results/result_operations_multiple_preconditions.txt').read().strip()
	code0 = file('pawsapp/tests/functional/codes/test_multiple1.c').read()
	code1 = file('pawsapp/tests/functional/codes/test_multiple2.c').read()
	lang = 'C'
	first_response = self.app.get(url(controller='operations', action='get_directory'))
	assert 'directory' in first_response.session
	second_response = self.app.get(url(controller='operations', action='get_functions'),
				params={
					'number': '2',
					'code0': code0,
					'code1': code1,
					'lang0': lang,
					'lang1': lang 
				})
	assert 'sources' in second_response.session

        response = self.app.get(url(controller='operations', action='perform_multiple'), 
				params={
					'language': 'C',
					'operation': 'preconditions',
					'function': 'a1'
				})
        assert result_code in response


