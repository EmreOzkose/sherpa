#!/usr/bin/env python3
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
"""
A standalone script for offline ASR recognition.

It loads an onnx model, decodes the given wav files, and exits.

Usage:
    ./onnx_offline_asr.py --help

For BPE based models (e.g., LibriSpeech):

    ./offline_asr.py \
        --onnx-model-filename /path/to/all_in_one.onnx \
        --bpe-model-filename /path/to/bpe.model \
        --decoding-method greedy_search \
        ./foo.wav \
        ./bar.wav \
        ./foobar.wav

For character based models (e.g., aishell):

    ./offline.py \
        --onnx-model-filename /path/to/all_in_one.onnx \
        --token-filename /path/to/lang_char/tokens.txt \
        --decoding-method greedy_search \
        ./foo.wav \
        ./bar.wav \
        ./foobar.wav

"""  # noqa
import argparse
import logging
import os
from typing import List, Optional, Union

import k2
import kaldifeat
import onnx
import onnx_graphsurgeon as gs
import onnxruntime
import sentencepiece as spm
import torch
import torchaudio
from onnx_beam_search import GreedySearchOfflineOnnx

from sherpa import add_beam_search_arguments


def get_args():
    beam_search_parser = add_beam_search_arguments()
    parser = argparse.ArgumentParser(
        parents=[beam_search_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--onnx-model-filename",
        type=str,
        required=True,
        help="""The all-in-one onnx model. You can use
        egs/librispeech/ASR/pruned_transducer_stateless3/onnx_export.py \
        --onnx=1 from icefall
        to generate this model.
        """,
    )

    parser.add_argument(
        "--bpe-model-filename",
        type=str,
        help="""The BPE model
        You can find it in the directory egs/librispeech/ASR/data/lang_bpe_xxx
        from icefall,
        where xxx is the number of BPE tokens you used to train the model.
        Note: Use it only when your model is using BPE. You don't need to
        provide it if you provide `--token-filename`
        """,
    )

    parser.add_argument(
        "--token-filename",
        type=str,
        help="""Filename for tokens.txt
        You can find it in the directory
        egs/aishell/ASR/data/lang_char/tokens.txt from icefall.
        Note: You don't need to provide it if you provide `--bpe-model`
        """,
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="The expected sample rate of the input sound files",
    )

    parser.add_argument(
        "sound_files",
        type=str,
        nargs="+",
        help="The input sound file(s) to transcribe. "
        "Supported formats are those supported by torchaudio.load(). "
        "For example, wav and flac are supported. "
        "The sample rate has to equal to `--sample-rate`.",
    )

    return (
        parser.parse_args(),
        beam_search_parser.parse_known_args()[0],
    )


def read_sound_files(
    filenames: List[str],
    expected_sample_rate: int,
) -> List[torch.Tensor]:
    """Read a list of sound files into a list 1-D float32 torch tensors.
    Args:
      filenames:
        A list of sound filenames.
      expected_sample_rate:
        The expected sample rate of the sound files.
    Returns:
      Return a list of 1-D float32 torch tensors.
    """
    ans = []
    for f in filenames:
        wave, sample_rate = torchaudio.load(f)
        assert sample_rate == expected_sample_rate, (
            f"expected sample rate: {expected_sample_rate}. "
            f"Given: {sample_rate}"
        )
        # We use only the first channel
        ans.append(wave[0])
    return ans


class OfflineAsrOnnx(object):
    def __init__(
        self,
        onnx_model_filename: str,
        bpe_model_filename: Optional[str],
        token_filename: Optional[str],
        num_active_paths: int,
        sample_rate: int = 16000,
        device: Union[str, torch.device] = "cpu",
        beam_search_params: dict = {},
    ):
        """
        Args:
          nn_model_filename:
            Path to the torch script model.
          bpe_model_filename:
            Path to the BPE model. If it is None, you have to provide
            `token_filename`.
          token_filename:
            Path to tokens.txt. If it is None, you have to provide
            `bpe_model_filename`.
          num_active_paths:
            Used only when decoding_method is modified_beam_search.
            It specifies number of active paths for each utterance. Due to
            merging paths with identical token sequences, the actual number
            may be less than "num_active_paths".
          sample_rate:
            Expected sample rate of the feature extractor.
          device:
            The device to use for computation.
          beam_search_params:
            Dictionary containing all the parameters for beam search.
        """
        """
        self.model = RnntConformerModel(
            filename=nn_model_filename,
            device=device,
            optimize_for_inference=False,
        )
        """
        self.model_onnx = onnx.load(onnx_model_filename)

        self.encoder = self.get_onnx_session(
            self.model_onnx,
            ["encoder/x", "encoder/x_lens"],
            ["encoder/encoder_out", "encoder/encoder_out_lens"],
        )
        self.decoder = self.get_onnx_session(
            self.model_onnx,
            ["decoder/y"],
            ["decoder/decoder_out"],
        )
        self.joiner = self.get_onnx_session(
            self.model_onnx,
            [
                "joiner/encoder_out",
                "joiner/decoder_out",
                "joiner/project_input",
            ],
            ["joiner/logit"],
        )
        self.joiner_encoder_proj = self.get_onnx_session(
            self.model_onnx,
            ["joiner_encoder_proj/encoder_out"],
            ["joiner_encoder_proj/encoder_proj"],
        )
        self.joiner_decoder_proj = self.get_onnx_session(
            self.model_onnx,
            ["joiner_decoder_proj/decoder_out"],
            ["joiner_decoder_proj/decoder_proj"],
        )

        if bpe_model_filename:
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(bpe_model_filename)
        else:
            self.token_table = k2.SymbolTable.from_file(token_filename)

        self.feature_extractor = self._build_feature_extractor(
            sample_rate=sample_rate,
            device=device,
        )

        blank_id, unk_id, context_size = self.extract_parameters()
        decoding_method = beam_search_params["decoding_method"]
        if decoding_method == "greedy_search":
            self.beam_search = GreedySearchOfflineOnnx(
                blank_id, unk_id, context_size
            )
        else:
            raise ValueError(
                f"Decoding method {decoding_method} is not supported."
            )

        self.device = device

    def _build_feature_extractor(
        self,
        sample_rate: int = 16000,
        device: Union[str, torch.device] = "cpu",
    ) -> kaldifeat.OfflineFeature:
        """Build a fbank feature extractor for extracting features.

        Args:
          sample_rate:
            Expected sample rate of the feature extractor.
          device:
            The device to use for computation.
        Returns:
          Return a fbank feature extractor.
        """
        opts = kaldifeat.FbankOptions()
        opts.device = device
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = False
        opts.frame_opts.samp_freq = sample_rate
        opts.mel_opts.num_bins = 80

        fbank = kaldifeat.Fbank(opts)

        return fbank

    def decode_waves(self, waves: List[torch.Tensor]) -> List[List[str]]:
        """
        Args:
          waves:
            A list of 1-D torch.float32 tensors containing audio samples.
            wavs[i] contains audio samples for the i-th utterance.

            Note:
              Whether it should be in the range [-32768, 32767] or be normalized
              to [-1, 1] depends on which range you used for your training data.
              For instance, if your training data used [-32768, 32767],
              then the given waves have to contain samples in this range.

              All models trained in icefall use the normalized range [-1, 1].
        Returns:
          Return a list of decoded results. `ans[i]` contains the decoded
          results for `wavs[i]`.
        """
        waves = [w.to(self.device) for w in waves]
        features = self.feature_extractor(waves)

        tokens = self.beam_search.process(
            encoder=self.encoder,
            decoder=self.decoder,
            joiner=self.joiner,
            joiner_encoder_proj=self.joiner_encoder_proj,
            joiner_decoder_proj=self.joiner_decoder_proj,
            features=features,
            device=self.device,
        )

        if hasattr(self, "sp"):
            results = self.sp.decode(tokens)
        else:
            results = [[self.token_table[i] for i in hyp] for hyp in tokens]
            results = ["".join(r) for r in results]

        return results

    def get_onnx_session(
        self,
        onnx_graph: onnx.ModelProto,
        input_op_names: list,
        output_op_names: list,
        non_verbose=False,
    ):
        graph = gs.import_onnx(onnx_graph)

        # Extraction of input OP and output OP
        graph_node_inputs = [
            graph_nodes
            for graph_nodes in graph.nodes
            for graph_nodes_input in graph_nodes.inputs
            if graph_nodes_input.name in input_op_names
        ]
        graph_node_outputs = [
            graph_nodes
            for graph_nodes in graph.nodes
            for graph_nodes_output in graph_nodes.outputs
            if graph_nodes_output.name in output_op_names
        ]

        # Init graph INPUT/OUTPUT
        graph.inputs.clear()
        graph.outputs.clear()

        # Update graph INPUT/OUTPUT
        graph.inputs = [
            graph_node_input
            for graph_node in graph_node_inputs
            for graph_node_input in graph_node.inputs
            if graph_node_input.shape
        ]
        graph.outputs = [
            graph_node_output
            for graph_node in graph_node_outputs
            for graph_node_output in graph_node.outputs
        ]

        # Cleanup
        graph.cleanup().toposort()

        # clean repeated inputs for Joiner
        # somehow, after import_onnx(),
        # there are 2 joiner/encoder_out and joiner/decoder_out
        if "joiner" in graph.inputs[0].name:
            graph.inputs = graph.inputs[:2]

        # Shape Estimation
        extracted_graph = None
        try:
            extracted_graph = onnx.shape_inference.infer_shapes(
                gs.export_onnx(graph)
            )
        except Exception:
            extracted_graph = gs.export_onnx(graph)
            if not non_verbose:
                print(
                    "WARNING: "
                    + "The input shape of the next OP does not match"
                    + "the output shape. "
                    + "Be sure to open the .onnx file to verify"
                    + "the certainty of the geometry."
                )

        onnx.save(extracted_graph, "tmp.onnx")
        onnx.checker.check_model(extracted_graph)
        sess = onnxruntime.InferenceSession("tmp.onnx")
        os.remove("tmp.onnx")
        return sess

    def extract_parameters(self):
        for node_i in self.model_onnx.graph.node:
            if node_i.name == "constants_lm":
                att_dict = {att.name: att.i for att in node_i.attribute}
                return (
                    att_dict["blank_id"],
                    att_dict["unk_id"],
                    att_dict["context_size"],
                )


@torch.no_grad()
def main():
    args, beam_search_parser = get_args()
    beam_search_params = vars(beam_search_parser)
    logging.info(vars(args))

    onnx_model_filename = args.onnx_model_filename
    bpe_model_filename = args.bpe_model_filename
    token_filename = args.token_filename
    num_active_paths = args.num_active_paths
    sample_rate = args.sample_rate
    sound_files = args.sound_files

    decoding_method = beam_search_params["decoding_method"]
    assert decoding_method in (
        "greedy_search",
        "modified_beam_search",
    ), decoding_method

    if decoding_method == "modified_beam_search":
        assert num_active_paths >= 1, num_active_paths

    if bpe_model_filename:
        assert token_filename is None, (
            "You need to provide either --bpe-model-filename or "
            "--token-filename parameter. But not both."
        )

    if token_filename:
        assert bpe_model_filename is None, (
            "You need to provide either --bpe-model-filename or "
            "--token-filename parameter. But not both."
        )

    assert bpe_model_filename or token_filename, (
        "You need to provide either --bpe-model-filename or "
        "--token-filename parameter. But not both."
    )

    device = torch.device("cpu")
    # if torch.cuda.is_available():
    #    device = torch.device("cuda", 0)

    logging.info(f"device: {device}")

    offline_asr = OfflineAsrOnnx(
        onnx_model_filename=onnx_model_filename,
        bpe_model_filename=bpe_model_filename,
        token_filename=token_filename,
        num_active_paths=num_active_paths,
        sample_rate=sample_rate,
        device=device,
        beam_search_params=beam_search_params,
    )

    waves = read_sound_files(
        filenames=sound_files,
        expected_sample_rate=sample_rate,
    )

    logging.info("Decoding started.")

    hyps = offline_asr.decode_waves(waves)

    s = "\n"
    for filename, hyp in zip(sound_files, hyps):
        s += f"{filename}:\n{hyp}\n\n"
    logging.info(s)

    logging.info("Decoding done.")


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# See https://github.com/pytorch/pytorch/issues/38342
# and https://github.com/pytorch/pytorch/issues/33354
#
# If we don't do this, the delay increases whenever there is
# a new request that changes the actual batch size.
# If you use `py-spy dump --pid <server-pid> --native`, you will
# see a lot of time is spent in re-compiling the torch script model.
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._set_graph_executor_optimize(False)
"""
// Use the following in C++
torch::jit::getExecutorMode() = false;
torch::jit::getProfilingMode() = false;
torch::jit::setGraphExecutorOptimize(false);
"""

if __name__ == "__main__":
    torch.manual_seed(20220609)
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"  # noqa
    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
