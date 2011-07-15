from pawsapp.tests import *

class TestUserfilesController(TestController):

    def test_upload(self):
        file_name = 'pawsapp/tests/functional/codes/test.c'
	response = self.app.post(url(controller='userfiles', action='upload'), upload_files=[('file', file_name)])
	assert file_name in response
	assert 'int main()' in response

    def test_upload_zip(self):
        file_name = 'pawsapp/tests/functional/codes/test.zip'
	response = self.app.post(url(controller='userfiles', action='upload'), upload_files=[('file', file_name)])
	assert 'initial_parallel.c' in response
	assert 'simple_loop.c' in response
	assert 'int main (void)' in response
	assert 'void a1' in response
