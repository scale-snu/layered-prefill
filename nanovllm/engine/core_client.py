import logging
import asyncio
import pickle
import queue
from threading import Thread
import weakref
import zmq
import zmq.asyncio
import multiprocessing
import gc

from nanovllm.config import Config
from nanovllm.request_type import EngineCoreRequestType
from nanovllm.engine.engine_core import EngineCore
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.zmq_utils import get_open_zmq_inproc_path, get_open_zmq_ipc_path, make_zmq_socket

logger = logging.getLogger("core_client")
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(process)d - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)

class CoreClient:
    def __init__(self, config: Config):
        # sync_ctx = zmq.Context(io_threads=2)
        self.loop = asyncio.get_event_loop()
        self.ctx = zmq.asyncio.Context.instance(io_threads=2)
        self.outputs_queue = queue.Queue[Sequence]()
        engine_core_process, addresses = launch_engine_core(config)
        logger.info("Engine core inited")
        self.engine_core_process = engine_core_process

        input_address = addresses["input_address"]
        output_address = addresses["output_address"]
        self.input_socket  = make_zmq_socket(
            self.ctx, input_address, zmq.ROUTER, bind=True)
        identity, _ = self.loop.run_until_complete(self.input_socket.recv_multipart())
        logger.info(f"Core engine input socket identity: {identity}")
        self.core_engine_identity = identity
        self.output_socket = make_zmq_socket(
            self.ctx, output_address, zmq.PULL)

        shutdown_path = get_open_zmq_inproc_path()
        self.shutdown_path = shutdown_path

        def process_outputs_socket():
            assert isinstance(self.output_socket, zmq.Socket)
            shutdown_socket = self.ctx.socket(zmq.PAIR)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                shutdown_socket.bind(shutdown_path)
                poller = zmq.Poller()
                poller.register(shutdown_socket, zmq.POLLIN)
                poller.register(self.output_socket, zmq.POLLIN)
                logger.info("Core client output thread started")
                while True:
                    socks = poller.poll()
                    if not socks:
                        continue
                    if len(socks) == 2 or socks[0][0] == shutdown_socket:
                        # shutdown signal, exit thread.
                        logger.info("Core client output thread receive shutdown signal")
                        break

                    obj = loop.run_until_complete(self.output_socket.recv(copy=False))
                    request_type, seqs = pickle.loads(obj)
                    if request_type == EngineCoreRequestType.SHUTDOWN:
                        logger.info("Core client output thread receive shutdown signal")
                        self.engine_core_process.join()
                        self.engine_core_process = None
                        break
                    elif request_type == EngineCoreRequestType.ADD:
                        self.outputs_queue.put_nowait(seqs)
            finally:
                # Close sockets.
                shutdown_socket.close(linger=0)
                self.output_socket.close(linger=0)

        gc.collect()
        gc.freeze()

        self.output_queue_thread = Thread(target=process_outputs_socket,
                                        name="EngineCoreOutputQueueThread",
                                        daemon=True)
        self.output_queue_thread.start()
        self._finalizer = weakref.finalize(self, self.close)

    def close(self):
        self._shut_down_core_engine()
        self.input_socket.close()
        logger.info("Core client input socket closed")
        if self.shutdown_path and self.output_queue_thread:
            with self.ctx.socket(zmq.PAIR) as shutdown_sender:
                shutdown_sender.connect(self.shutdown_path)
                # Send shutdown signal.
                shutdown_sender.send(b'')
        self.output_queue_thread.join()
        logger.info("Core client output thread is shut down")

    def add_request(self, seq: Sequence):
        self.input_socket.send_multipart([self.core_engine_identity, pickle.dumps((EngineCoreRequestType.ADD, seq), protocol=pickle.HIGHEST_PROTOCOL)], copy=False)

    def _shut_down_core_engine(self):
        logger.info("Shutdown core engine")
        if self.engine_core_process is not None:
            self.input_socket.send_multipart([self.core_engine_identity, pickle.dumps((EngineCoreRequestType.SHUTDOWN, None), protocol=pickle.HIGHEST_PROTOCOL)], copy=False)
            self.engine_core_process.join()

    def get_output(self) -> Sequence:
        try:
            output = self.outputs_queue.get(block=False)
        except queue.Empty:
            return None
        return output

    def is_rest(self):
        return not self.outputs_queue.empty()

    def is_alive(self):
        return self.engine_core_process is not None

    @staticmethod
    def make_core_client(config: Config):
        return CoreClient(config)

def launch_engine_core(config: Config):
    input_address = get_open_zmq_ipc_path(config)
    output_address = get_open_zmq_ipc_path(config)
    import torch
    torch.multiprocessing.set_start_method('spawn')
    process = multiprocessing.Process(target=EngineCore.run_engine,
                    name=f"EngineCore",
                    kwargs={
                        "config": config,
                        "input_address": input_address,
                        "output_address": output_address,
                    })
    process.start()
    logger.info("Engine core process started")
    return process, {"input_address": input_address, "output_address": output_address}
