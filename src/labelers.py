import torch

from EventStream.data.pytorch_dataset import PytorchBatch
from EventStream.transformer.zero_shot_labeler import Labeler


class A1cGreaterThan7Labeler(Labeler):
    def __call__(self, batch: PytorchBatch, input_seq_len: int) -> tuple[torch.LongTensor, torch.BoolTensor]:
        static_indices = batch.static_indices
        static_measurement_indices = batch.static_measurement_indices
        
        a1c_greater_than_7_idx = self.config.vocab_idxmap['static']['A1cGreaterThan7']
        
        pred_a1c_greater_than_7 = torch.any(
            (static_measurement_indices == a1c_greater_than_7_idx) & (static_indices == 1),
            dim=1
        ).long()
        
        unknown_pred = torch.all(static_measurement_indices != a1c_greater_than_7_idx, dim=1)
        
        return pred_a1c_greater_than_7.unsqueeze(1), unknown_pred