from pawsapp.tests import *

class TestOpenmpController(TestController):

    def test_index(self):
        response = self.app.get(url(controller='tools_openmp', action='index'))
        assert 'Openmp demo page' in response
