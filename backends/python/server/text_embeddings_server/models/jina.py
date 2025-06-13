from pathlib import Path
from typing import Type, List, Optional

import torch
from opentelemetry import trace
from transformers import AutoModel

from text_embeddings_server.models import Model
from text_embeddings_server.models.types import PaddedBatch, Task
from text_embeddings_server.pb.embed_pb2 import Embedding, Score

tracer = trace.get_tracer(__name__)


def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    This mean-pooling code has been fetched from the official `jina-embeddings-v3` Hugging Face repository.

    Recap:
    Mean pooling takes all token embeddings from the model's output
    and averages them at the sentence or paragraph level.
    This approach has been shown to produce high-quality sentence embeddings.
    """
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class JinaModel(Model):

    def __init__(self, model_path: Path, device: torch.device, dtype: torch.dtype):
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(dtype).to(device)
        super(JinaModel, self).__init__(model=model, dtype=dtype, device=device)

    @property
    def batch_type(self) -> Type[PaddedBatch]:
        return PaddedBatch

    def convert_task_to_adapter_mask(self, tasks: List[Optional[Task]], default_task: Task = Task.TEXT_MATCHING) -> torch.Tensor:
        adaptation_map = getattr(self.model, '_adaptation_map')
        adapter_mask = torch.full((len(tasks),), adaptation_map[default_task.value], dtype=torch.int32, device=self.device)
        for index, task in enumerate(tasks):
            if not task:
                task = default_task
            adapter_mask[index] = adaptation_map[task.value]
        return adapter_mask

    @tracer.start_as_current_span("embed")
    def embed(self, batch: PaddedBatch) -> List[Embedding]:
        adapter_mask = self.convert_task_to_adapter_mask(batch.task)
        with torch.no_grad():
            token_embeds = self.model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                adapter_mask=adapter_mask)[0]

        # NOTE: Convert to f32 to avoid overflow on Rust backend.
        token_embeds = token_embeds.float()

        embeddings = mean_pooling(token_embeds, batch.attention_mask)
        embeddings = embeddings.tolist()

        # NOTE: Dimension truncation would have done on Rust backend.
        return [Embedding(values=embedding) for embedding in embeddings]

    @tracer.start_as_current_span("predict")
    def predict(self, batch: PaddedBatch) -> List[Score]:
        pass
