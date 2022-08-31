# Copyright      2022  Xiaomi Corp.        (authors: Yunus Emre Ozkose)
#
# See LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
from typing import List

import numpy as np
import onnxruntime
import torch
from torch.nn.utils.rnn import pad_sequence

LOG_EPS = math.log(1e-10)


class GreedySearchOfflineOnnx:
    def __init__(self, blank_id, unk_id, context_size):
        self.blank_id = blank_id
        self.unk_id = unk_id
        self.context_size = context_size

    @torch.no_grad()
    def process(
        self,
        encoder: onnxruntime.InferenceSession,
        decoder: onnxruntime.InferenceSession,
        joiner: onnxruntime.InferenceSession,
        joiner_encoder_proj: onnxruntime.InferenceSession,
        joiner_decoder_proj: onnxruntime.InferenceSession,
        features: List[torch.Tensor],
        device="cpu",
    ) -> List[List[int]]:
        """
        Args:
          model:
            RNN-T model decoder model

          features:
            A list of 2-D tensors. Each entry is of shape
            (num_frames, feature_dim).
        Returns:
          Return a list-of-list containing the decoding token IDs.
        """
        features_length = torch.tensor(
            [f.size(0) for f in features],
            dtype=torch.int64,
        )
        features = pad_sequence(
            features,
            batch_first=True,
            padding_value=LOG_EPS,
        )

        features = features.to(device)
        features_length = features_length.to(device)

        encoder_out_length, encoder_out = encoder.run(
            None,
            {
                encoder.get_inputs()[0]
                .name: features.numpy()
                .astype(np.float32),
                encoder.get_inputs()[1].name: features_length.numpy(),
            },
        )
        encoder_out_length, encoder_out = torch.from_numpy(
            encoder_out_length
        ), torch.from_numpy(encoder_out)

        hyp_tokens = self.greedy_search(
            encoder_out=encoder_out,
            encoder_out_length=encoder_out_length.cpu(),
            decoder=decoder,
            joiner=joiner,
            joiner_encoder_proj=joiner_encoder_proj,
            joiner_decoder_proj=joiner_decoder_proj,
            device=device,
        )
        return hyp_tokens

    def greedy_search(
        self,
        encoder_out: torch.Tensor,
        encoder_out_length: torch.Tensor,
        decoder,
        joiner,
        joiner_encoder_proj,
        joiner_decoder_proj,
        device,
    ) -> List[List[int]]:
        """Greedy search in batch mode. It hardcodes --max-sym-per-frame=1.
        Args:
        model:
            The transducer model.
        encoder_out:
            Output from the encoder. Its shape is (N, T, C), where N >= 1.
        encoder_out_lens:
            A 1-D tensor of shape (N,), containing number of valid frames in
            encoder_out before padding.
        Returns:
        Return a list-of-list of token IDs containing the decoded results.
        len(ans) equals to encoder_out.size(0).
        """
        assert encoder_out.ndim == 3
        assert encoder_out.size(0) >= 1, encoder_out.size(0)
        packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
            input=encoder_out,
            lengths=encoder_out_length,
            batch_first=True,
            enforce_sorted=False,
        )

        device = encoder_out.device
        batch_size_list = packed_encoder_out.batch_sizes.tolist()
        N = encoder_out.size(0)
        assert torch.all(encoder_out_length > 0), encoder_out_length
        assert N == batch_size_list[0], (N, batch_size_list)

        hyps = [[self.blank_id] * self.context_size for _ in range(N)]

        decoder_input = torch.tensor(
            hyps,
            device=device,
            dtype=torch.int64,
        )  # (N, context_size)
        decoder_out = decoder.run(
            None,
            {
                decoder.get_inputs()[0]
                .name: decoder_input.numpy()
                .astype(np.int64)
            },
        )[0]
        decoder_out = torch.from_numpy(decoder_out)

        decoder_out = joiner_decoder_proj.run(
            None,
            {
                joiner_decoder_proj.get_inputs()[0]
                .name: decoder_out.detach()
                .squeeze(0)
                .numpy()
                .astype(np.float32)
            },
        )[0]
        decoder_out = torch.from_numpy(decoder_out)

        encoder_out = joiner_encoder_proj.run(
            None,
            {
                joiner_encoder_proj.get_inputs()[0]
                .name: packed_encoder_out.data.detach()
                .numpy()
                .astype(np.float32)
            },
        )[0]
        encoder_out = torch.from_numpy(encoder_out)

        offset = 0

        for batch_size in batch_size_list:
            start = offset
            end = offset + batch_size
            current_encoder_out = encoder_out.data[start:end]
            current_encoder_out = current_encoder_out.unsqueeze(1).unsqueeze(1)
            # current_encoder_out's shape: (batch_size, 1, 1, encoder_out_dim)
            offset = end

            decoder_out = decoder_out[:batch_size]

            logits = joiner.run(
                None,
                {
                    joiner.get_inputs()[0]
                    .name: current_encoder_out.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32),
                    joiner.get_inputs()[1]
                    .name: decoder_out.unsqueeze(1)
                    .unsqueeze(1)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32),
                },
            )[0]
            logits = torch.from_numpy(logits)
            # logits'shape (batch_size, 1, 1, vocab_size)

            logits = logits.squeeze(1).squeeze(1)  # (batch_size, vocab_size)
            assert logits.ndim == 2, logits.shape
            y = logits.argmax(dim=1).tolist()
            emitted = False
            for i, v in enumerate(y):
                if v not in (self.blank_id, self.unk_id):
                    hyps[i].append(v)
                    emitted = True
            if emitted:
                # update decoder output
                decoder_input = [
                    h[-self.context_size:] for h in hyps[:batch_size]
                ]
                decoder_input = torch.tensor(
                    decoder_input,
                    device=device,
                    dtype=torch.int64,
                )
                decoder_out = decoder.run(
                    None,
                    {
                        decoder.get_inputs()[0]
                        .name: decoder_input.numpy()
                        .astype(np.int64)
                    },
                )[0]
                decoder_out = torch.from_numpy(decoder_out)

                decoder_out = joiner_decoder_proj.run(
                    None,
                    {
                        joiner_decoder_proj.get_inputs()[0]
                        .name: decoder_out.detach()
                        .squeeze(0)
                        .numpy()
                        .astype(np.float32)
                    },
                )[0]
                decoder_out = torch.from_numpy(decoder_out)

        sorted_ans = [h[self.context_size:] for h in hyps]
        ans = []
        unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
        for i in range(N):
            ans.append(sorted_ans[unsorted_indices[i]])

        return ans
