import os
from enum import Enum
from typing import Optional, List

import math
import torch

from abc import ABC, abstractmethod
from dataclasses import dataclass
from opentelemetry import trace

from text_embeddings_server.pb import embed_pb2
from text_embeddings_server.pb.embed_pb2 import Embedding, Score

tracer = trace.get_tracer(__name__)
PAD_SEQUENCE_TO_MULTIPLE_OF = int(os.environ.get("PAD_SEQUENCE_TO_MULTIPLE_OF", 128))


def round_up(number, k):
    return (number + k - 1) // k * k


class Task(Enum):
    RETRIEVAL_QUERY = "retrieval.query"
    RETRIEVAL_PASSAGE = "retrieval.passage"
    SEPARATION = "separation"
    CLASSIFICATION = "classification"
    TEXT_MATCHING = "text-matching"


def proto_to_embed_tasks(pb_task) -> Optional[Task]:
    task = None
    if pb_task == embed_pb2.Task.RETRIEVAL_QUERY:
        task = Task.RETRIEVAL_QUERY
    elif pb_task == embed_pb2.Task.RETRIEVAL_PASSAGE:
        task = Task.RETRIEVAL_PASSAGE
    elif pb_task == embed_pb2.Task.SEPARATION:
        task = Task.SEPARATION
    elif pb_task == embed_pb2.Task.CLASSIFICATION:
        task = Task.CLASSIFICATION
    elif pb_task == embed_pb2.Task.TEXT_MATCHING:
        task = Task.TEXT_MATCHING
    return task


class Batch(ABC):
    @classmethod
    @abstractmethod
    def from_pb(cls, pb: embed_pb2.EmbedRequest, device: torch.device) -> "Batch":
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError


@dataclass
class PaddedBatch(Batch):
    input_ids: torch.Tensor
    token_type_ids: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: torch.Tensor
    task: List[Optional[Task]]
    dimensions: List[Optional[int]]

    @classmethod
    @tracer.start_as_current_span("from_pb")
    def from_pb(
        cls, pb: embed_pb2.EmbedRequest, device: torch.device, max_input_length: int
    ) -> "PaddedBatch":
        if pb.max_length > max_input_length:
            raise RuntimeError(f"input length exceeds model config's max_input_length")

        batch_size = len(pb.cu_seq_lengths) - 1
        if device.type == "hpu":
            # To better utilize HPU, we need to do batch/seq_len bucketing
            max_length = round_up(pb.max_length, PAD_SEQUENCE_TO_MULTIPLE_OF)
            max_length = min(max_length, max_input_length)
            new_bs = 2 ** math.ceil(math.log2(batch_size))
        else:
            new_bs = batch_size
            max_length = pb.max_length
        # Allocate padded tensors all at once
        all_tensors = torch.zeros([4, new_bs, max_length], dtype=torch.int32)

        for i, start_index in enumerate(pb.cu_seq_lengths[:-1]):
            end_index = pb.cu_seq_lengths[i + 1]
            input_length = end_index - start_index

            all_tensors[0, i, :input_length] = torch.tensor(
                pb.input_ids[start_index:end_index], dtype=torch.int32
            )
            all_tensors[1, i, :input_length] = torch.tensor(
                pb.token_type_ids[start_index:end_index], dtype=torch.int32
            )
            all_tensors[2, i, :input_length] = torch.tensor(
                pb.position_ids[start_index:end_index], dtype=torch.int32
            )
            all_tensors[3, i, :input_length] = 1

        # Move padded tensors all at once
        all_tensors = all_tensors.to(device)
        task = [
            proto_to_embed_tasks(t) for t in pb.task
        ]
        dimensions = [
            None if dimension == 0 else dimension for dimension in pb.dimensions
        ]

        return PaddedBatch(
            input_ids=all_tensors[0],
            token_type_ids=all_tensors[1],
            position_ids=all_tensors[2],
            attention_mask=all_tensors[3],
            task=task,
            dimensions=dimensions,
        )

    def __len__(self):
        return len(self.input_ids)


@dataclass
class FlashBatch(Batch):
    input_ids: torch.Tensor
    token_type_ids: torch.Tensor
    position_ids: torch.Tensor

    cu_seqlens: torch.Tensor
    max_s: int
    size: int

    @classmethod
    @tracer.start_as_current_span("from_pb")
    def from_pb(
        cls, pb: embed_pb2.EmbedRequest, device: torch.device, max_input_length: int
    ) -> "FlashBatch":
        batch_input_ids = torch.tensor(pb.input_ids, dtype=torch.int32, device=device)
        batch_token_type_ids = torch.tensor(
            pb.token_type_ids, dtype=torch.int32, device=device
        )
        batch_position_ids = torch.tensor(
            pb.position_ids, dtype=torch.int32, device=device
        )

        cu_seqlens = torch.tensor(pb.cu_seq_lengths, dtype=torch.int32, device=device)

        return FlashBatch(
            input_ids=batch_input_ids,
            token_type_ids=batch_token_type_ids,
            position_ids=batch_position_ids,
            cu_seqlens=cu_seqlens,
            max_s=pb.max_length,
            size=len(cu_seqlens) - 1,
        )

    def __len__(self):
        return self.size