import time
import gc
import atexit
import logging
import threading
import torch.multiprocessing as mp
import queue
import zmq
import pickle
from contextlib import ExitStack

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.config import Config
from nanovllm.request_type import EngineCoreRequestType
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.utils.zmq_utils import make_zmq_socket
from nanovllm.utils.utils import disable_gc


logger = logging.getLogger("engine_core")
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(process)d - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)


class EngineCore:
    def __init__(self, config: Config, input_address: str, output_address: str):
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        if config.tensor_parallel_size > 1:
            for i in range(1, config.tensor_parallel_size):
                event = ctx.Event()
                process = ctx.Process(target=ModelRunner, args=(config, i, event))
                process.start()
                self.ps.append(process)
                self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events if config.tensor_parallel_size > 1 else [])
        logger.info("Engine core model runner loaded")
        self.scheduler = Scheduler(config)
        self.input_queue = queue.Queue[Sequence]()
        self.output_queue = queue.Queue[Sequence]()
        self.input_address = input_address
        self.output_address = output_address
        logger.info(f"Engine core input address: {self.input_address}")
        logger.info(f"Engine core output address: {self.output_address}")
        self.input_thread = threading.Thread(target=self.process_input_sockets,
                         args=(self.input_address,),
                         daemon=True)
        self.input_thread.start()
        self.output_thread = threading.Thread(
            target=self.process_output_sockets,
            args=(self.output_address,),
            daemon=True)
        self.output_thread.start()

    def exit(self):
        pass
        # self._send_engine_dead()
        # self.model_runner.call("exit")
        # del self.model_runner
        # for p in self.ps:
        #     p.join()
        # logger.info("Engine core model runner exit")

    @staticmethod
    def run_engine(config: Config, input_address: str, output_address: str):
        engine : EngineCore = None
        try:
            engine = EngineCore(config, input_address, output_address)
            engine.busy_loop()
        except Exception as e:
            logger.error(f"Engine core exception: {e}")
        finally:
            if engine is not None:
                engine.exit()

    def _send_engine_dead(self):
        logger.info("Send engine core dead signal")
        exit_seq = Sequence([-1], seq_id=-1)
        self.output_queue.put_nowait([exit_seq])
        self.output_thread.join(timeout=5.0)

    def busy_loop(self):
        shutdown = False
        while True:
            s = time.time()
            shutdown = shutdown or self._process_input_queue()
            m = time.time()
            self._process_engine_step()
            e = time.time()
            duration = e - s
            if duration > 0.08:
                logger.warning(f"Engine core step took too long: {duration:.3f}s")
                print(f"  Input queue processing time: {m - s:.3f}s"
                      f"  Engine step time: {e - m:.3f}s")
            if shutdown:
                break

    def _process_input_queue(self):
        while not self.input_queue.empty():
            seq = self.input_queue.get_nowait()
            if seq.seq_id == -1:
                logger.info("Engine core input thread shutdown")
                return True
            try:
                self.scheduler.add(seq)
            except Exception as e:
                logger.error(f"Failed to add sequence {seq.seq_id} to scheduler: {e}")
                seq.set_error(str(e))
                self.output_queue.put_nowait([seq])
        return False

    @disable_gc()
    def _process_engine_step(self):
        seqs = self.scheduler.schedule()
        if seqs is None or len(seqs) == 0:
            return
        token_ids = self.model_runner.call("run", seqs)
        self.scheduler.postprocess(seqs, token_ids)
        self.output_queue.put_nowait(seqs)

    def process_input_sockets(self, input_address: str):
        """Input socket IO thread."""
        with ExitStack() as stack, zmq.Context() as ctx:
            input_socket = stack.enter_context(make_zmq_socket(ctx,
                                    input_address,
                                    zmq.DEALER,
                                    bind=False))
            poller = zmq.Poller()
            # Send initial message to input socket - this is required
            # before the front-end ROUTER socket can send input messages
            # back to us.
            input_socket.send(b'')
            poller.register(input_socket, zmq.POLLIN)
            logger.info("Engine core input socket connected")

            while True:
                for input_socket, _ in poller.poll():
                    # (RequestType, RequestData)
                    serialized_obj = input_socket.recv(copy=False)
                    obj = pickle.loads(serialized_obj)
                    request_type = obj[0]
                    if (request_type == EngineCoreRequestType.ADD):
                        self.input_queue.put_nowait(obj[1])
                    elif (request_type == EngineCoreRequestType.SHUTDOWN):
                        logger.info("Engine core input thread shutdown")
                        self.input_queue.put_nowait(Sequence([-1], seq_id=-1))
                        break

    def process_output_sockets(self, output_address: str):
        """Output socket IO thread."""
        with ExitStack() as stack, zmq.Context() as ctx:
            socket = stack.enter_context(make_zmq_socket(ctx, output_address, zmq.PUSH, linger=4000))
            logger.info("Engine core output socket connected")

            while True:
                output = self.output_queue.get()
                for seq in output:
                    if seq.seq_id == -1:
                        socket.send(pickle.dumps((EngineCoreRequestType.SHUTDOWN, None), protocol=pickle.HIGHEST_PROTOCOL))
                        logger.info("Engine core output thread closed")
                        break
                serialized_obj = pickle.dumps((EngineCoreRequestType.ADD, output), protocol=pickle.HIGHEST_PROTOCOL)
                socket.send(serialized_obj)
