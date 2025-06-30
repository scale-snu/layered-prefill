import ssl
from dataclasses import dataclass, field
from typing import Optional

from nanovllm.config import Config


@dataclass
class APIServerConfig(Config):
    log_level: str = field(
        default="debug", metadata={"help": "Logging level for the API server."}
    )
    host: str = field(
        default="localhost",
        metadata={"help": "Hostname or IP address to bind the server to."},
    )
    port: int = field(
        default=8000, metadata={"help": "Port number to run the server on."}
    )
