from pawsapp.tests import *
import string, os

class TestGraphController(TestController):

    def test_dependence_graph(self):
	test_code = file('pawsapp/tests/functional/codes/test.c').read()
	resp = self.app.get(url(controller='operations', action='get_directory'))
	assert 'directory' in resp.session
	response = self.app.get(url(controller='graph', action='dependence_graph'), 
				params={
					'code': test_code,
					'language': 'C',
				})
	start = string.find(response._body, 'a href=')
	file_name = response._body[start + 8 : string.find(response._body, '"', start + 8)]
	assert 'main.png' in response
	assert os.path.exists('pawsapp/public' + file_name) == True

    def test_dependence_graph_multiple(self):
	test_code = file('pawsapp/tests/functional/codes/test.c').read()
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
	response = self.app.get(url(controller='graph', action='dependence_graph_multi'))
	start = string.find(response._body, 'a href=')
	file_name = response._body[start + 8 : string.find(response._body, '"', start + 8)]
	assert 'main.png' in response
	assert os.path.exists('pawsapp/public' + file_name) == True

