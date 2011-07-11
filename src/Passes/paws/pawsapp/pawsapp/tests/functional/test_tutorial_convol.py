from pawsapp.tests import *

class TestTutorialConvolController(TestController):

    def test_index(self):
	result = file('pawsapp/tests/functional/results/result_tutorial_convol.txt').read().strip()
        response = self.app.get(url(controller='tutorial_convol', action='index'))
        assert 'Current demo step:' in response
