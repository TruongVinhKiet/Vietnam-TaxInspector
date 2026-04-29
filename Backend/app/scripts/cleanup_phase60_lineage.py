from __future__ import annotations

import json

from sqlalchemy import text

from app.database import SessionLocal


TARGET_VERSION = "ops-learned-2026-04-28"


def main() -> None:
    with SessionLocal() as db:
        updates = {}

        updates["deployment_rollouts_entity"] = db.execute(
            text(
                """
                UPDATE deployment_rollouts
                SET model_version = :target,
                    rollout_metadata = (COALESCE(rollout_metadata::jsonb, '{}'::jsonb) || CAST(:meta_patch AS jsonb))::json
                WHERE model_name = 'entity_risk_fusion'
                  AND model_version = 'entity-fusion-phase60-v1'
                """
            ),
            {
                "target": TARGET_VERSION,
                "meta_patch": json.dumps({"learned_artifact_version": TARGET_VERSION}),
            },
        ).rowcount

        updates["deployment_rollouts_collections"] = db.execute(
            text(
                """
                UPDATE deployment_rollouts
                SET model_version = :target,
                    rollout_metadata = (COALESCE(rollout_metadata::jsonb, '{}'::jsonb) || CAST(:meta_patch AS jsonb))::json
                WHERE model_name = 'collections_uplift'
                  AND model_version = 'collections-uplift-phase60-v1'
                """
            ),
            {
                "target": TARGET_VERSION,
                "meta_patch": json.dumps({"learned_artifact_version": TARGET_VERSION}),
            },
        ).rowcount

        updates["ccr_entity"] = db.execute(
            text(
                """
                UPDATE champion_challenger_results
                SET challenger_version = :target
                WHERE model_name = 'entity_risk_fusion'
                  AND challenger_version = 'entity-fusion-phase60-v1'
                """
            ),
            {"target": TARGET_VERSION},
        ).rowcount

        updates["ccr_collections"] = db.execute(
            text(
                """
                UPDATE champion_challenger_results
                SET challenger_version = :target
                WHERE model_name = 'collections_uplift'
                  AND challenger_version = 'collections-uplift-phase60-v1'
                """
            ),
            {"target": TARGET_VERSION},
        ).rowcount

        updates["eval_entity"] = db.execute(
            text(
                """
                UPDATE evaluation_slices
                SET model_version = :target
                WHERE model_name = 'entity_risk_fusion'
                  AND model_version = 'entity-fusion-phase60-v1'
                """
            ),
            {"target": TARGET_VERSION},
        ).rowcount

        updates["eval_collections"] = db.execute(
            text(
                """
                UPDATE evaluation_slices
                SET model_version = :target
                WHERE model_name = 'collections_uplift'
                  AND model_version = 'collections-uplift-phase60-v1'
                """
            ),
            {"target": TARGET_VERSION},
        ).rowcount

        updates["calibration_entity"] = db.execute(
            text(
                """
                UPDATE calibration_bins
                SET model_version = :target
                WHERE model_name = 'entity_risk_fusion'
                  AND model_version = 'entity-fusion-phase60-v1'
                """
            ),
            {"target": TARGET_VERSION},
        ).rowcount

        updates["fusion_predictions"] = db.execute(
            text(
                """
                UPDATE entity_risk_fusion_predictions
                SET model_version = :target
                WHERE model_version = 'entity-fusion-phase60-v1'
                """
            ),
            {"target": TARGET_VERSION},
        ).rowcount

        updates["nba_predictions"] = db.execute(
            text(
                """
                UPDATE nba_predictions
                SET model_version = :target
                WHERE model_version = 'collections-uplift-phase60-v1'
                """
            ),
            {"target": TARGET_VERSION},
        ).rowcount

        db.commit()
        print("[OK] Phase60 lineage cleanup completed")
        print(json.dumps(updates, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
