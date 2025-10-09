import logging
from dataclasses import fields
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.core_client import CoreClient


logger = logging.getLogger("llm_engine")
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(process)d - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.rquest_ids = set()
        self.core_client = CoreClient.make_core_client(config)
        logger.info("LLMEngine init")

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.rquest_ids.add(seq.seq_id)
        self.core_client.add_request(seq)

    def step(self):
        seq = self.core_client.get_output()
        self.rquest_ids.remove(seq.seq_id)
        return (seq.seq_id, seq.completion_token_ids)

    def is_finished(self):
        return len(self.rquest_ids) == 0

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
    ) -> list[str]:
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        while not self.is_finished() and (self.core_client.is_alive() or self.core_client.is_rest()):
            seq_id, token_ids = self.step()
            outputs[seq_id] = token_ids
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
