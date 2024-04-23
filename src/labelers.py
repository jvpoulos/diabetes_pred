import torch

from EventStream.data.pytorch_dataset import PytorchBatch
from EventStream.transformer.model_output import get_event_types
from EventStream.transformer.zero_shot_labeler import Labeler


def masked_idx_in_set(
    indices_T: torch.LongTensor, indices_set: set[int], mask: torch.BoolTensor
) -> torch.BoolTensor:
    return torch.where(mask, torch.any(torch.stack([(indices_T == i) for i in indices_set], 0), dim=0), False)


class A1cGreaterThan7Labeler(Labeler):
    def __call__(self, batch: PytorchBatch, input_seq_len: int) -> tuple[torch.LongTensor, torch.BoolTensor]:
        static_indices = batch.static_indices
        static_measurement_indices = batch.static_measurement_indices
        
        a1c_greater_than_7_idx = self.config.measurements_idxmap["A1cGreaterThan7"]
        
        pred_a1c_greater_than_7 = torch.any(
            (static_measurement_indices == a1c_greater_than_7_idx) & (static_indices == 1),
            dim=1
        ).long()
        
        unknown_pred = ~torch.any(static_measurement_indices == a1c_greater_than_7_idx, dim=1)
        
        return pred_a1c_greater_than_7.unsqueeze(1), unknown_pred