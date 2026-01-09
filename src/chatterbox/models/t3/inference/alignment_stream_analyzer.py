# Copyright (c) 2025 Resemble AI
# Author: John Meade, Jeremy Hsu
# MIT License
import logging
import torch
from dataclasses import dataclass

logger = logging.getLogger(__name__)

LLAMA_ALIGNED_HEADS = [(12, 15), (13, 11), (9, 2)]


@dataclass
class AlignmentAnalysisResult:
    # was this frame detected as being part of a noisy beginning chunk with potential hallucinations?
    false_start: bool
    # was this frame detected as being part of a long tail with potential hallucinations?
    long_tail: bool
    # was this frame detected as repeating existing text content?
    repetition: bool
    # was the alignment position of this frame too far from the previous frame?
    discontinuity: bool
    # has inference reached the end of the text tokens? eg, this remains false if inference stops early
    complete: bool
    # approximate position in the text token sequence. Can be used for generating online timestamps.
    position: int


class AlignmentStreamAnalyzer:
    """
    HARD-DISABLED alignment analyzer.

    Reason:
      - Transformers SDPA attention does not support output_attentions=True
      - The original analyzer relies on attention weights + hooks to estimate alignment
      - For Mac/MPS (and recent Transformers defaults), that combination breaks.

    This implementation is a no-op drop-in replacement that preserves the public API:
      - __init__ accepts the same args
      - step(logits, next_token=...) returns logits unchanged
    """

    def __init__(self, tfmr, queue, text_tokens_slice, alignment_layer_idx=9, eos_idx=0):
        # Keep attributes that other code might expect to exist
        self.text_tokens_slice = text_tokens_slice
        self.eos_idx = eos_idx

        i, j = text_tokens_slice
        self.alignment = torch.zeros(0, max(j - i, 0))
        self.curr_frame_pos = 0
        self.text_position = 0

        self.started = True
        self.started_at = 0

        self.complete = False
        self.completed_at = None

        # Token repetition tracking (kept because it is independent of attention)
        self.generated_tokens = []

        # Mark disabled so step() can branch cheaply if you ever re-enable later
        self.disabled = True

        # Maintain the field for compatibility; it's unused here
        self.last_aligned_attns = [None for _ in LLAMA_ALIGNED_HEADS]

    def _add_attention_spy(self, tfmr, buffer_idx, layer_idx, head_idx):
        # Intentionally disabled: no hooks, no output_attentions
        return

    def step(self, logits, next_token=None):
        """
        No-op: returns logits unchanged.
        """
        # Optional: keep lightweight repetition tracking without forcing EOS.
        if next_token is not None:
            if isinstance(next_token, torch.Tensor):
                token_id = next_token.item() if next_token.numel() == 1 else next_token.view(-1)[0].item()
            else:
                token_id = int(next_token)
            self.generated_tokens.append(token_id)
            if len(self.generated_tokens) > 8:
                self.generated_tokens = self.generated_tokens[-8:]

        self.curr_frame_pos += 1
        return logits