import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from lib.llm_client import ValidationRetryError
from phase_09_ambiguity_pruning import (
    CandidateDiscoveryResult,
    CandidateGroup,
    ScopeSpec,
    TaxonomyPruneDecision,
    apply_pruning_to_tree,
    build_parent_by_id,
    build_prompt_index,
    render_scope_tree,
    validate_candidate_discovery_result,
    validate_path_taxonomy_tree,
)


def test_phase09_indexes_phase08_path_style_tree():
    metas = [
        {
            "meta": "adverse_events",
            "paths": [
                {"path": "cardiovascular", "description": "Cardiovascular events"},
                {"path": "cardiovascular/arrhythmias", "description": "Arrhythmias"},
                {"path": "neurological", "description": "Neurological events"},
            ],
        },
        {
            "meta": "conditions",
            "paths": [
                {"path": "cardiovascular", "description": "Cardiovascular conditions"},
            ],
        },
    ]

    refs, ordered_by_meta, ids_by_meta = build_prompt_index(metas)

    assert len(ids_by_meta["adverse_events"]) == 3
    assert len(ids_by_meta["conditions"]) == 1
    assert all(node_id.startswith("N") for node_id in refs)

    tree_text = render_scope_tree(
        ScopeSpec(scope_id="in_meta", scope_type="in_meta", metas=("adverse_events",)),
        refs,
        ordered_by_meta,
    )

    assert "[N001] cardiovascular :: Cardiovascular events" in tree_text
    assert "  - [N002] cardiovascular/arrhythmias :: Arrhythmias" in tree_text
    assert "adverse_events" not in ids_by_meta["adverse_events"]


def test_phase09_rejects_path_tree_with_missing_parent_before_llm_calls():
    tree = {
        "metas": [
            {
                "meta": "adverse_events",
                "paths": [
                    {"path": "cardiovascular/arrhythmias", "description": "Arrhythmias"},
                ],
            }
        ]
    }

    with pytest.raises(ValueError, match="Missing parent path for adverse_events: cardiovascular/arrhythmias -> cardiovascular"):
        validate_path_taxonomy_tree(tree)


def test_candidate_discovery_validation_carries_failed_result_for_repair_feedback():
    result = CandidateDiscoveryResult(
        candidates=[CandidateGroup(concern="meta name leak", node_ids=["adverse_events", "N001"])]
    )

    with pytest.raises(ValidationRetryError) as exc_info:
        validate_candidate_discovery_result(result, allowed_node_ids={"N001", "N002"})

    assert "adverse_events" in str(exc_info.value)
    assert exc_info.value.failed_result is result


def test_phase09_prunes_path_canonical_tree_and_keeps_paths_output():
    tree = {
        "domain_context": "x",
        "target": "adverse_events",
        "model": "qwen/qwen3.5-27b",
        "max_children": 5,
        "metas": [
            {
                "meta": "adverse_events",
                "paths": [
                    {"path": "cardiovascular", "description": "Cardiovascular"},
                    {"path": "cardiovascular/arrhythmias", "description": "Arrhythmias"},
                    {"path": "cardiovascular/myocardial_infarction", "description": "MI"},
                    {"path": "neurological", "description": "Neurological"},
                ],
            },
            {
                "meta": "conditions",
                "paths": [
                    {"path": "cardiovascular", "description": "Cardiovascular conditions"},
                ],
            },
        ],
    }
    validate_path_taxonomy_tree(tree)
    refs, _ordered, _ids = build_prompt_index(tree["metas"])
    arrhythmias_id = next(
        node_id
        for node_id, ref in refs.items()
        if ref.meta == "adverse_events" and ref.code == "cardiovascular/arrhythmias"
    )
    neurological_id = next(
        node_id
        for node_id, ref in refs.items()
        if ref.meta == "adverse_events" and ref.code == "neurological"
    )

    pruned_tree, removed_ids = apply_pruning_to_tree(
        tree,
        refs,
        [
            TaxonomyPruneDecision(
                concern="example",
                keep_node_id=neurological_id,
                remove_node_ids=[arrhythmias_id],
            )
        ],
        model="qwen/qwen3.5-27b",
    )

    assert arrhythmias_id in removed_ids
    adverse_meta = pruned_tree["metas"][0]
    assert set(adverse_meta) == {"meta", "paths"}
    assert [row["path"] for row in adverse_meta["paths"]] == [
        "cardiovascular",
        "cardiovascular/myocardial_infarction",
        "neurological",
    ]
    assert "roots" not in adverse_meta
    assert "nodes" not in adverse_meta
    validate_path_taxonomy_tree(pruned_tree)


def test_path_prompt_refs_preserve_parent_child_relationships():
    metas = [
        {
            "meta": "adverse_events",
            "paths": [
                {"path": "cardiovascular", "description": "Cardiovascular"},
                {"path": "cardiovascular/arrhythmias", "description": "Arrhythmias"},
            ],
        }
    ]

    refs, _ordered, _ids = build_prompt_index(metas)
    parent_by_id = build_parent_by_id(refs)
    child_id = next(node_id for node_id, ref in refs.items() if ref.code == "cardiovascular/arrhythmias")
    parent_id = next(node_id for node_id, ref in refs.items() if ref.code == "cardiovascular")

    assert parent_by_id[child_id] == parent_id
    assert refs[parent_id].child_node_ids == [child_id]
