from uuid import uuid4
import zmq
from typing import Any, Union, Optional

from nanovllm.config import Config

def get_open_zmq_ipc_path(config: Config) -> str:
    base_rpc_path = config.rpc_base_path if config.rpc_base_path else "/tmp"
    return f"ipc://{base_rpc_path}/{uuid4()}"

def get_open_zmq_inproc_path() -> str:
    return f"inproc://{uuid4()}"

def make_zmq_socket(
    ctx: Union[zmq.asyncio.Context, zmq.Context],
    path: str,
    socket_type: Any,
    bind: Optional[bool] = None,
    linger: Optional[int] = None,
) -> Union[zmq.Socket, zmq.asyncio.Socket]:
    socket = ctx.socket(socket_type)

    if bind is None:
        bind = socket_type not in (zmq.PUSH, zmq.SUB, zmq.XSUB)

    if socket_type in (zmq.PULL, zmq.DEALER, zmq.ROUTER):
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.RCVBUF, -1)

    if socket_type in (zmq.PUSH, zmq.DEALER, zmq.ROUTER):
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.SNDBUF, -1)

    if linger is not None:
        socket.setsockopt(zmq.LINGER, linger)

    if bind:
        socket.bind(path)
    else:
        socket.connect(path)

    return socket