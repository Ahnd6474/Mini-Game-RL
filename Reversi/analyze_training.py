"""Summarize a measured training run as a compact Markdown report."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .training_plot import read_jsonl, write_training_curves


def _metric_record(
    records: list[dict[str, object]], key: str, *, maximize: bool
) -> dict[str, object]:
    candidates = [record for record in records if isinstance(record.get(key), (int, float))]
    selector = max if maximize else min
    return selector(candidates, key=lambda record: float(record[key]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    metrics_path = args.run_dir / "metrics.jsonl"
    records = read_jsonl(metrics_path)
    trained = [record for record in records if int(record["iteration"]) > 0]
    initial, final = records[0], records[-1]
    best_validation = _metric_record(
        trained, "dvn_validation_total", maximize=False
    )
    best_cell = _metric_record(
        trained, "dvn_validation_cell_accuracy", maximize=True
    )
    best_winner = _metric_record(
        trained, "dvn_validation_decisive_accuracy", maximize=True
    )
    best_tree = _metric_record(records, "tree_score", maximize=True)
    throughput = statistics.mean(float(record["positions_per_second"]) for record in trained)
    total_positions = sum(int(record["positions"]) for record in trained)
    tree_points = [
        f"iter {int(record['iteration'])}: {100 * float(record['tree_score']):.1f}%"
        for record in records
        if "tree_score" in record
    ]

    post_path = args.run_dir / "post_training_eval.json"
    post = json.loads(post_path.read_text(encoding="utf-8")) if post_path.exists() else []
    post_lines = []
    for result in post:
        label = Path(result["checkpoint"]).stem
        post_lines.append(
            f"- {label}, depth {result['tree_depth']}: "
            f"{result['tree_wins']}승 {result['tree_draws']}무 {result['tree_losses']}패, "
            f"score {100 * result['tree_score']:.1f}%, 평균 돌 차 {result['tree_mean_disc_margin']:.2f}"
        )
    deployment_note = (
        "확대 대국이 없으므로 소규모 평가의 best-tree DPN을 우선 선택한다."
    )
    if post:
        deepest = max(int(result["tree_depth"]) for result in post)
        deepest_results = [
            result for result in post if int(result["tree_depth"]) == deepest
        ]
        expanded_best = max(
            deepest_results,
            key=lambda result: (
                float(result["tree_score"]),
                float(result["tree_mean_disc_margin"]),
            ),
        )
        deployment_note = (
            f"확대 depth-{deepest} 평가에서는 `{Path(expanded_best['checkpoint']).stem}`가 "
            f"{100 * float(expanded_best['tree_score']):.1f}%로 가장 높으므로 이 DPN을 선택한다."
        )

    ppo_records = [record for record in trained if "dpn_approx_kl" in record]
    ppo_section = ""
    if ppo_records:
        ppo_section = f"""
## PPO 안정성

- 평균 approximate KL: {statistics.mean(float(record['dpn_approx_kl']) for record in ppo_records):.5f}
- 최대 approximate KL: {max(float(record['dpn_approx_kl']) for record in ppo_records):.5f}
- 평균 clip fraction: {100 * statistics.mean(float(record['dpn_clip_fraction']) for record in ppo_records):.2f}%
- 평균 수행 epoch: {statistics.mean(float(record['dpn_epochs_completed']) for record in ppo_records):.2f}

KL과 clip fraction이 함께 기록되므로 대국 성능 변화가 실제 정책 과대 업데이트 때문인지 구분할 수 있다.
"""

    initial_total = float(initial["dvn_validation_total"])
    final_total = float(final["dvn_validation_total"])
    report = f"""# DVN/DPN 학습 분석

## 실행 요약

- iteration: {int(final['iteration'])}
- 학습 게임: {sum(int(record['games']) for record in trained):,}
- 학습 상태: {total_positions:,}
- 평균 처리량: {throughput:,.0f} positions/s
- 총 학습 경과: {float(final['elapsed_seconds']):.1f}s
- CUDA 최대 메모리: {max(int(record.get('cuda_peak_memory_mb', 0)) for record in trained)}MB

## DVN

- 고정 검증 total loss: {initial_total:.4f} → {final_total:.4f} ({100 * (final_total / initial_total - 1):+.1f}%)
- 최저 검증 loss: {float(best_validation['dvn_validation_total']):.4f} (iteration {int(best_validation['iteration'])})
- 최종 칸 정확도: {100 * float(initial['dvn_validation_cell_accuracy']):.2f}% → {100 * float(final['dvn_validation_cell_accuracy']):.2f}%
- 최고 칸 정확도: {100 * float(best_cell['dvn_validation_cell_accuracy']):.2f}% (iteration {int(best_cell['iteration'])})
- 승패 정확도: {100 * float(initial['dvn_validation_decisive_accuracy']):.2f}% → {100 * float(final['dvn_validation_decisive_accuracy']):.2f}%
- 최고 승패 정확도: {100 * float(best_winner['dvn_validation_decisive_accuracy']):.2f}% (iteration {int(best_winner['iteration'])})
- value 상관계수: {float(initial['dvn_validation_value_correlation']):.3f} → {float(final['dvn_validation_value_correlation']):.3f}

DVN은 분명한 예측 신호를 학습했지만, 후반부에는 현재 정책의 self-play 훈련 loss와 고정 random-policy 검증 loss가 분리된다. iteration {int(best_validation['iteration'])} 이후 검증 성능이 개선되지 않으므로, 다음 학습에서는 과거 상태 replay를 현재 rollout과 섞는 편이 좋다.

## DPN

- 소규모 고정 트리 평가: {', '.join(tree_points)}
- 최고 점수: {100 * float(best_tree['tree_score']):.1f}% (iteration {int(best_tree['iteration'])})
- 최종 점수: {100 * float(final['tree_score']):.1f}%

policy loss는 advantage가 곱해진 목적함수라 절댓값 자체가 품질 지표가 아니며, 고정 상대 대국 점수를 선택 기준으로 써야 한다. {deployment_note}

{ppo_section}

## 확대 대국

{chr(10).join(post_lines) if post_lines else '- 아직 실행하지 않음'}

## 다음 학습 권장 변경

1. 체크포인트 선택용 트리 대국을 최소 64~128판으로 늘려 16판 평가의 선택 노이즈를 줄인다.
2. entropy가 너무 빨리 감소하면 entropy weight를 높이거나 목표 entropy를 둔다.
3. DVN에는 최근 rollout만 쓰지 말고 이전 여러 iteration의 reservoir replay를 섞는다.
4. 자기대국 상대를 현재 DPN 하나로 고정하지 말고 과거 best-policy pool에서 일부 샘플링한다.
5. 체크포인트 선택은 DPN은 트리 대국 점수, DVN은 고정 검증 loss로 각각 독립 수행한다.
"""
    (args.run_dir / "analysis.md").write_text(report, encoding="utf-8")
    write_training_curves(records, args.run_dir / "learning_curves.svg")
    print(report)


if __name__ == "__main__":
    main()
