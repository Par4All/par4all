import inspect
import threading
from Queue import Queue

# local thread safe queue used to store actions
call_queue = Queue()

counter=0

# all actions are blocking except these one
asynchronous_calls = set(['apply','capply','push_property','pop_property'])

# now let's try some magic !
pypips_module = __import__('pypips')
current_module = __import__('apypips')

def synchronous_wrapper(fun):
	print "creating new sync fun", fun.__name__
	def anonymous(*args,**kwargs):
		print "waiting for sync"
		call_queue.join()
		print "running", fun.__name__
		return fun(*args,**kwargs)
	return anonymous

def asynchronous_wrapper(fun):
	print "creating new async fun", fun.__name__
	def anonymous(*args,**kwargs):
		global counter
		print "pushing task on queue"
		call_queue.put((fun,args,kwargs,counter))
		counter+=1
	return anonymous

for attr in dir(pypips_module):
	pattr=getattr(pypips_module,attr)
	if inspect.isbuiltin(pattr) and not pattr.__name__ in asynchronous_calls:
		setattr(current_module,attr,synchronous_wrapper(pattr))
	elif inspect.isbuiltin(pattr):
		setattr(current_module,attr,asynchronous_wrapper(pattr))

# ok we managed to do the wrapping, that's automatic code generation !

# this start processing the queue once the module is loaded
def queue_processor(*args,**kwargs):
	queue=kwargs['queue']
	while 1:
		print "waiting for task"
		(fun,fargs,fkwargs,count) = queue.get()
		print "task", count, "received"
		fun(*fargs,**fkwargs)
		queue.task_done()
		print "task",count, "processed"


processor=threading.Thread(name="apypips processor", target=queue_processor,kwargs={'queue':call_queue} )
processor.daemon=True
processor.start()


