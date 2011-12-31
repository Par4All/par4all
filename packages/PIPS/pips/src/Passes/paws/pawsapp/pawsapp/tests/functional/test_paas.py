from pawsapp.tests import *

class TestPassController(TestController):

    def test_index(self):
	response = self.app.get(url(controller='paas', action='index'))
	assert 'PYPS AS WEB SERVICE' in response

