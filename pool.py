import multiprocessing as mp
from typing import List,Generator,Union,Optional
import time, os

class BasicPool:
    def __init__(self, processes = None, sleep_time=60,**kwargs):
        self.running:List[mp.Process] = []
        self.queue:List[Union[mp.Process,Generator[mp.Process]]] = []
        self.processes = processes or os.cpu_count()
        self.sleep_time = sleep_time

    def update(self):
        for process in self.running[:]:
            process.join(0)
            if not process.is_alive():
                process.close()
                self.running.remove(process)
        while len(self.running)<self.processes and self.getNumProcesses() < self.processes:
            process = self._popNextProcess()
            if process is None: # No further processes to be added
                break
            self.running.append(process)
            process.start()

    def _popNextProcess(self):
        if not self.queue:
            return None
        if isinstance(self.queue[0],mp.Process):
            return self.queue.pop(0)

        #Assume it to be a generator
        process = next(self.queue[0],None)
        if process is None:
            self.queue.pop(0)
            return self._popNextProcess()
        return process


    def getNumProcesses(self):
        raise NotImplemented()

    def _createProcess(self,func, args=(),kwargs=None, name=None):
        kwargs = {} if kwargs is None else kwargs
        return mp.Process(name=name, target=func,args=args,kwargs=kwargs)

    def apply_async(self,func, args=(),kwargs=None, name = None):
        process = self._createProcess(func=func, name=name, args=args, kwargs=kwargs)
        self.queue.append(process)
        self.update()
        return process

    '''def imap(self, func, args_gen, max_buffersize=100):
        def process_gen():
            processbuffer = []
            for args in args_gen:
                yield self.apply_async(func,next(args_gen))
        gen = process_gen()
        open_processes = []
        for i in range(max_buffersize):
            np = next(gen,None)
            if np is None:
                break
            open_processes.append(np)

        while open_processes:
            cp = open_processes.pop(0)
            cp.join()
            yield cp.ge'''

    def wait(self):
        self.update()
        while self.running or self.queue:
            time.sleep(self.sleep_time)
            self.update()

class Pool(BasicPool):
    def getNumProcesses(self):
        if not os.path.exists('__temp_files'):
            return 0
        return len(os.listdir('__temp_files'))

class Pool2(BasicPool):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.numProcesses = mp.Value('i',0)

    def getNumProcesses(self):
        return self.numProcesses.value

    def _createProcess(self,func, args=(),kwargs=None, name=None):
        kwargs = {} if kwargs is None else kwargs

        def warppedfunc(func, args, kwargs, counter: mp.Value):
            with counter.get_lock():
                counter.value += 1
            res = func(*args, **kwargs)
            with counter.get_lock():
                counter.value -= 1
            return res

        return mp.Process(name=name, target=warppedfunc, args=(func,args,kwargs), kwargs={'counter':self.numProcesses})

