"""Generate a single-file, standard-library-only submission from a DPN."""

from __future__ import annotations

import argparse
import base64
import copy
import sys
import textwrap
import zlib
from pathlib import Path

import torch

from .model import DPN, ReversiTransformerConfig


_SUBMISSION_TEMPLATE = r'''# Generated Reversi DPN submission. Do not edit by hand.
import base64
import math
import operator
import struct
import sys
import zlib

SIZE = 8
PASS = 64
DMODEL = __DMODEL__
HEADS = __HEADS__
LAYERS = __LAYERS__
TENSORS = __TENSORS__
WEIGHTS = """__WEIGHTS__"""
DIRECTIONS = ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1))

try:
    _dot = math.sumprod  # Python 3.12+: optimized and accurately accumulated in C.
except AttributeError:
    def _dot(left, right):
        return sum(map(operator.mul, left, right))


def legal_actions(board, player):
    opponent = -player
    result = []
    for action, stone in enumerate(board):
        if stone:
            continue
        row, col = divmod(action, SIZE)
        for dr, dc in DIRECTIONS:
            r, c = row + dr, col + dc
            seen = False
            while 0 <= r < SIZE and 0 <= c < SIZE and board[r * SIZE + c] == opponent:
                seen = True
                r += dr
                c += dc
            if seen and 0 <= r < SIZE and 0 <= c < SIZE and board[r * SIZE + c] == player:
                result.append(action)
                break
    return result


def _linear_one(vector, weight, bias, output_size):
    """PyTorch Linear for one vector; each weight row is one output."""
    return [
        bias[output] + _dot(vector, weight[output])
        for output in range(output_size)
    ]


def _linear(rows, weight, bias, output_size):
    return [_linear_one(row, weight, bias, output_size) for row in rows]


def _layer_norm(rows, weight, bias):
    width = len(weight)
    inverse_width = 1.0 / width
    normalized_rows = []
    for row in rows:
        mean = sum(row) * inverse_width
        variance = 0.0
        for value in row:
            difference = value - mean
            variance += difference * difference
        scale = 1.0 / math.sqrt(variance * inverse_width + 1e-5)
        normalized_rows.append([
            (row[index] - mean) * scale * weight[index] + bias[index]
            for index in range(width)
        ])
    return normalized_rows


def _add(left, right):
    return [
        [a + b for a, b in zip(left_row, right_row)]
        for left_row, right_row in zip(left, right)
    ]


class PurePythonDPN:
    """A tiny Transformer runtime requiring only the Python standard library."""

    def __init__(self):
        packed = base64.b85decode("".join(WEIGHTS.split()))
        raw = zlib.decompress(packed)
        weights = {}
        offset = 0
        for name, shape in TENSORS:
            count = math.prod(shape)
            values = struct.unpack_from("<%de" % count, raw, offset)
            if len(shape) == 2:
                rows, columns = shape
                weights[name] = tuple(
                    values[row * columns:(row + 1) * columns]
                    for row in range(rows)
                )
            else:
                weights[name] = values
            offset += count * 2
        if offset != len(raw):
            raise ValueError("invalid embedded weight payload")
        self.w = weights

        row_weight = weights["encoder.row_embedding.weight"]
        col_weight = weights["encoder.col_embedding.weight"]
        self.position = []
        for cell in range(64):
            row = row_weight[cell // 8]
            column = col_weight[cell % 8]
            self.position.append([
                row[index] + column[index]
                for index in range(DMODEL)
            ])

    def attention(self, rows, prefix):
        weights = self.w
        qkv = _linear(
            rows,
            weights[prefix + ".in_proj_weight"],
            weights[prefix + ".in_proj_bias"],
            DMODEL * 3,
        )
        q = [row[:DMODEL] for row in qkv]
        k = [row[DMODEL:DMODEL * 2] for row in qkv]
        v = [row[DMODEL * 2:] for row in qkv]
        sequence_length = len(rows)
        head_size = DMODEL // HEADS
        scale = head_size ** -0.5
        context = [[0.0] * DMODEL for _ in rows]

        for head in range(HEADS):
            start = head * head_size
            stop = start + head_size
            queries = [row[start:stop] for row in q]
            keys = [row[start:stop] for row in k]
            # Transposing this small head once makes every context dot product
            # run inside the C implementations of map/operator.mul/sum.
            value_columns = list(zip(*(row[start:stop] for row in v)))
            for query_index, query in enumerate(queries):
                scores = [
                    _dot(query, key) * scale
                    for key in keys
                ]
                maximum = max(scores)
                probabilities = [math.exp(score - maximum) for score in scores]
                denominator = sum(probabilities)
                probabilities = [probability / denominator for probability in probabilities]
                context[query_index][start:stop] = [
                    _dot(probabilities, column)
                    for column in value_columns
                ]

        return _linear(
            context,
            weights[prefix + ".out_proj.weight"],
            weights[prefix + ".out_proj.bias"],
            DMODEL,
        )

    def __call__(self, board, player, legal):
        board = list(board)
        legal_set = set(legal)
        weights = self.w
        projection = weights["encoder.cell_projection.weight"]
        projection_bias = weights["encoder.cell_projection.bias"]
        cells = []
        for cell, stone in enumerate(board):
            features = (stone == 0, stone == player, stone == -player, cell in legal_set)
            position = self.position[cell]
            cells.append([
                projection_bias[output]
                + projection[output][0] * features[0]
                + projection[output][1] * features[1]
                + projection[output][2] * features[2]
                + projection[output][3] * features[3]
                + position[output]
                for output in range(DMODEL)
            ])

        cls_token = weights["encoder.cls_token"]
        empty_embedding = weights["encoder.empty_count_embedding.weight"]
        empty_row = empty_embedding[board.count(0)]
        cls = [
            cls_token[index] + empty_row[index]
            for index in range(DMODEL)
        ]
        rows = [cls] + cells

        for layer in range(LAYERS):
            root = "encoder.transformer.layers.%d" % layer
            normalized = _layer_norm(
                rows,
                weights[root + ".norm1.weight"],
                weights[root + ".norm1.bias"],
            )
            rows = _add(rows, self.attention(normalized, root + ".self_attn"))
            normalized = _layer_norm(
                rows,
                weights[root + ".norm2.weight"],
                weights[root + ".norm2.bias"],
            )
            hidden = _linear(
                normalized,
                weights[root + ".linear1.weight"],
                weights[root + ".linear1.bias"],
                len(weights[root + ".linear1.bias"]),
            )
            for row_index, row in enumerate(hidden):
                hidden[row_index] = [
                    0.5 * value * (
                        1.0 + math.tanh(0.7978845608028654 * (value + 0.044715 * value * value * value))
                    )
                    for value in row
                ]
            rows = _add(
                rows,
                _linear(
                    hidden,
                    weights[root + ".linear2.weight"],
                    weights[root + ".linear2.bias"],
                    DMODEL,
                ),
            )

        rows = _layer_norm(
            rows,
            weights["encoder.transformer.norm.weight"],
            weights["encoder.transformer.norm.bias"],
        )
        cell_weight = weights["cell_head.weight"][0]
        cell_bias = weights["cell_head.bias"][0]
        logits = [
            cell_bias + _dot(row, cell_weight)
            for row in rows[1:]
        ]
        pass_weight = weights["pass_head.weight"][0]
        pass_logit = weights["pass_head.bias"][0] + _dot(rows[0], pass_weight)
        return logits + [pass_logit]


# Compatibility for callers of submissions produced by the earlier exporter.
NumpyDPN = PurePythonDPN
MODEL = None


def choose_action(board, player, legal):
    global MODEL
    if not legal:
        return PASS
    if MODEL is None:
        MODEL = PurePythonDPN()
    logits = MODEL(board, player, legal)
    return max(legal, key=lambda action: logits[action])


def main():
    player = 1
    for line in sys.stdin:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "READY":
            player = 1 if parts[1] == "FIRST" else -1
            print("OK", flush=True)
        elif parts[0] == "TURN":
            board = [0 if char == "." else (1 if char == "B" else -1) for char in parts[2]]
            legal = legal_actions(board, player)
            action = choose_action(board, player, legal)
            if action == PASS:
                print("PASS", flush=True)
            else:
                row, col = divmod(action, 8)
                print("MOVE", row + 1, col + 1, flush=True)


if __name__ == "__main__":
    main()
'''


def _serialize_weights(model: DPN) -> tuple[str, str]:
    """Return source metadata and a compressed little-endian FP16 payload."""
    chunks: list[bytes] = []
    metadata: list[tuple[str, tuple[int, ...]]] = []
    for name, tensor in model.state_dict().items():
        half = tensor.detach().to(dtype=torch.float16).contiguous()
        # uint8.tolist() avoids making NumPy an exporter dependency as well.
        chunks.append(bytes(half.view(torch.uint8).flatten().tolist()))
        metadata.append((name, tuple(half.shape)))
    if sys.byteorder != "little":
        raise RuntimeError("submission export currently requires a little-endian host")
    packed = zlib.compress(b"".join(chunks), level=9)
    encoded = base64.b85encode(packed).decode("ascii")
    return repr(tuple(metadata)), encoded


def export_dpn_submission(model: DPN, output_path: str | Path) -> Path:
    """Write one standalone source file containing the DPN and its weights."""
    model = copy.deepcopy(model).eval().cpu()
    metadata, encoded = _serialize_weights(model)
    wrapped_weights = "\n".join(textwrap.wrap(encoded, width=120))
    source = (
        _SUBMISSION_TEMPLATE.replace("__DMODEL__", str(model.config.d_model))
        .replace("__HEADS__", str(model.config.num_heads))
        .replace("__LAYERS__", str(model.config.num_layers))
        .replace("__TENSORS__", metadata)
        .replace("__WEIGHTS__", wrapped_weights)
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(source, encoding="utf-8", newline="\n")
    return output_path


def export_checkpoint(checkpoint_path: str | Path, output_path: str | Path) -> Path:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ReversiTransformerConfig(**checkpoint.get("model_config", {}))
    model = DPN(config)
    model.load_state_dict(checkpoint["dpn"])
    return export_dpn_submission(model, output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("Reversi/submission.py"))
    args = parser.parse_args()
    path = export_checkpoint(args.checkpoint, args.output)
    print(path)


if __name__ == "__main__":
    main()
