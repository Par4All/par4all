import broker

class fftwbroker(broker.Broker):
    def __init__(self):
        super(fftwbroker,self).__init__()
    def get_broker_dirs(self):
        return ['fftw'] + super(fftwbroker,self).get_broker_dirs()

# main script
if __name__ == '__main__':
    broker.parse_args()
