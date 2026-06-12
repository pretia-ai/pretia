"""Comprehensive tests for per-workflow input generators.

Cover base allocation logic, dirty injection, determinism, per-workflow
structural invariants, save/load round-trip, and workflow-specific
behaviors (W13 weights, W5 modality, W19 session depth, W17 claims).
"""

from __future__ import annotations

import importlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import pytest

# Ensure the project root is on sys.path so ``inputs.generators`` is importable.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from inputs.generators._base import (  # noqa: E402
    GROUND_TRUTH_WEIGHTS,
    PROFILING_WEIGHTS,
    BaseInputGenerator,
    GeneratedInput,
)

# ---------------------------------------------------------------------------
# Inline concrete subclass for base-class tests
# ---------------------------------------------------------------------------


class _StubGenerator(BaseInputGenerator):
    """Minimal concrete generator for testing BaseInputGenerator logic."""

    workflow_id = "TEST"
    dirty_types = ["typos"]

    def generate_single(
        self,
        tier,
        profile,
        rng,
        idx,
        is_dirty=False,
        dirty_type=None,
    ):
        return GeneratedInput(
            id=self.make_id(profile, tier, idx),
            workflow="TEST",
            profile=profile,
            tier=tier,
            token_count=100,
            is_dirty=is_dirty,
            dirty_type=dirty_type,
            structural_descriptor={"stub": True},
            input_data={"input": f"test {tier} {idx}"},
        )


# ===================================================================
# 1. BaseInputGenerator tests
# ===================================================================


class TestBaseAllocations:
    """Verify tier allocation, dirty injection, determinism, and ID format."""

    def test_profiling_tier_allocation(self):
        """50 profiling inputs -> 20 easy, 18 medium, 10 hard, 2 edge."""
        gen = _StubGenerator(seed=42)
        inputs = gen.generate_batch("profiling", 50)
        counts = Counter(i.tier for i in inputs)
        assert counts["easy"] == 20
        assert counts["medium"] == 18
        assert counts["hard"] == 10
        assert counts["edge"] == 2

    def test_ground_truth_tier_allocation(self):
        """500 GT inputs -> 275 easy, 125 medium, 60 hard, 25 edge, 15 extreme."""
        gen = _StubGenerator(seed=42)
        inputs = gen.generate_batch("ground_truth", 500)
        counts = Counter(i.tier for i in inputs)
        assert counts["easy"] == 275
        assert counts["medium"] == 125
        assert counts["hard"] == 60
        assert counts["edge"] == 25
        assert counts["extreme"] == 15

    def test_deterministic_with_seed(self):
        """Same seed produces identical tier assignments."""
        gen_a = _StubGenerator(seed=42)
        gen_b = _StubGenerator(seed=42)
        tiers_a = [i.tier for i in gen_a.generate_batch("profiling", 50)]
        tiers_b = [i.tier for i in gen_b.generate_batch("profiling", 50)]
        assert tiers_a == tiers_b

    def test_different_seeds_different_results(self):
        """Different seeds produce different tier assignments."""
        gen_a = _StubGenerator(seed=42)
        gen_b = _StubGenerator(seed=99)
        tiers_a = [i.tier for i in gen_a.generate_batch("profiling", 50)]
        tiers_b = [i.tier for i in gen_b.generate_batch("profiling", 50)]
        assert tiers_a != tiers_b

    def test_dirty_input_count_profiling(self):
        """Profiling n=50 should produce 2-3 dirty inputs (~5%)."""
        gen = _StubGenerator(seed=42)
        inputs = gen.generate_batch("profiling", 50)
        dirty_count = sum(1 for i in inputs if i.is_dirty)
        assert 2 <= dirty_count <= 3

    def test_dirty_input_count_ground_truth(self):
        """GT n=500 should produce 20-30 dirty inputs (~5%)."""
        gen = _StubGenerator(seed=42)
        inputs = gen.generate_batch("ground_truth", 500)
        dirty_count = sum(1 for i in inputs if i.is_dirty)
        assert 20 <= dirty_count <= 30

    def test_dirty_inputs_spread_across_tiers(self):
        """Dirty inputs should appear in at least 2 different tiers."""
        gen = _StubGenerator(seed=42)
        inputs = gen.generate_batch("ground_truth", 500)
        dirty_tiers = {i.tier for i in inputs if i.is_dirty}
        assert len(dirty_tiers) >= 2

    def test_output_has_all_fields(self):
        """Every GeneratedInput must have all required fields set."""
        gen = _StubGenerator(seed=42)
        inputs = gen.generate_batch("profiling", 10)
        required_attrs = [
            "id",
            "workflow",
            "profile",
            "tier",
            "token_count",
            "is_dirty",
            "dirty_type",
            "structural_descriptor",
            "input_data",
        ]
        for inp in inputs:
            for attr in required_attrs:
                assert hasattr(inp, attr), f"Missing attribute: {attr}"
                # id, workflow, profile, tier should be non-empty strings
                if attr in ("id", "workflow", "profile", "tier"):
                    assert isinstance(getattr(inp, attr), str)
                    assert getattr(inp, attr) != ""

    def test_make_id_format(self):
        """IDs follow the pattern w{NN}_{prof|gt}_{tier}_{NNN}."""
        gen = _StubGenerator(seed=42)
        gen.workflow_id = "W01"
        assert gen.make_id("profiling", "easy", 0) == "w01_prof_easy_000"
        assert gen.make_id("profiling", "medium", 5) == "w01_prof_medium_005"
        assert gen.make_id("ground_truth", "hard", 42) == "w01_gt_hard_042"
        assert gen.make_id("ground_truth", "extreme", 100) == "w01_gt_extreme_100"

    def test_make_id_prof_vs_gt_prefix(self):
        """Profiling IDs use 'prof', ground truth IDs use 'gt'."""
        gen = _StubGenerator(seed=42)
        gen.workflow_id = "W05"
        prof_id = gen.make_id("profiling", "easy", 7)
        gt_id = gen.make_id("ground_truth", "easy", 7)
        assert "_prof_" in prof_id
        assert "_gt_" in gt_id


# ===================================================================
# 2. Per-workflow generator tests (parametrized across all 13)
# ===================================================================

ALL_GENERATORS = [
    ("W01", "inputs.generators.w01_support_simple", "W01SupportSimpleGenerator"),
    ("W02", "inputs.generators.w02_support_complex", "W02SupportComplexGenerator"),
    ("W04", "inputs.generators.w04_compliance_review", "ComplianceReviewGenerator"),
    ("W05", "inputs.generators.w05_multimodal_extraction", "MultimodalExtractionGenerator"),
    ("W09", "inputs.generators.w09_sales_outreach", "W09SalesOutreachGenerator"),
    ("W12", "inputs.generators.w12_extraction_deepseek", "W12ExtractionDeepSeekGenerator"),
    ("W13", "inputs.generators.w13_routing_agent", "W13RoutingAgentGenerator"),
    ("W14", "inputs.generators.w14_simple_rag_queries", "SimpleRagQueryGenerator"),
    ("W15", "inputs.generators.w15_multihop_rag_queries", "MultihopRagQueryGenerator"),
    ("W16", "inputs.generators.w16_map_reduce", "W16MapReduceGenerator"),
    ("W17", "inputs.generators.w17_claims_agent", "W17ClaimsAgentGenerator"),
    ("W18", "inputs.generators.w18_long_document", "W18LongDocumentGenerator"),
    ("W19", "inputs.generators.w19_multi_turn", "MultiTurnGenerator"),
    ("W11", "inputs.generators.w11_support_qwen", "W11SupportQwenGenerator"),
]


@pytest.fixture(params=ALL_GENERATORS, ids=[g[0] for g in ALL_GENERATORS])
def generator_and_inputs(request):
    """Instantiate each generator with seed=42 and produce profiling n=50."""
    wf_id, mod_path, cls_name = request.param
    mod = importlib.import_module(mod_path)
    cls = getattr(mod, cls_name)
    gen = cls(seed=42)
    inputs = gen.generate_batch("profiling", 50)
    return gen, inputs


class TestWorkflowGenerators:
    """Cross-cutting invariants that every generator must satisfy."""

    def test_produces_correct_count(self, generator_and_inputs):
        """Batch should contain exactly the requested number of inputs."""
        _gen, inputs = generator_and_inputs
        assert len(inputs) == 50

    def test_all_inputs_have_required_fields(self, generator_and_inputs):
        """Every input must have the core fields populated."""
        _gen, inputs = generator_and_inputs
        for inp in inputs:
            assert isinstance(inp.id, str) and inp.id
            assert isinstance(inp.workflow, str) and inp.workflow
            assert isinstance(inp.profile, str) and inp.profile
            assert isinstance(inp.tier, str) and inp.tier
            assert isinstance(inp.token_count, int)
            assert isinstance(inp.structural_descriptor, dict)
            assert isinstance(inp.input_data, dict)

    def test_tier_labels_valid(self, generator_and_inputs):
        """All tier labels must be from the known set."""
        _gen, inputs = generator_and_inputs
        valid_tiers = {"easy", "medium", "hard", "edge", "extreme"}
        for inp in inputs:
            assert inp.tier in valid_tiers, f"Invalid tier '{inp.tier}' in {inp.id}"

    def test_extreme_only_in_ground_truth(self, generator_and_inputs):
        """Profiling batches should never contain the 'extreme' tier."""
        _gen, inputs = generator_and_inputs
        extreme_count = sum(1 for i in inputs if i.tier == "extreme")
        assert extreme_count == 0, "Profiling batch should not contain extreme-tier inputs"

    def test_id_format(self, generator_and_inputs):
        """IDs should start with w{NN}_prof_ for profiling batches."""
        gen, inputs = generator_and_inputs
        wf_num = gen.workflow_id.upper().replace("W", "")
        expected_prefix = f"w{wf_num}_prof_"
        for inp in inputs:
            assert inp.id.startswith(expected_prefix), (
                f"Expected prefix '{expected_prefix}', got id '{inp.id}'"
            )


# ===================================================================
# 3. W13 special tier weights
# ===================================================================


class TestW13Weights:
    """W13 uses non-standard tier weights: heavy easy, no edge in profiling."""

    @pytest.fixture
    def w13_gen(self):
        mod = importlib.import_module("inputs.generators.w13_routing_agent")
        return mod.W13RoutingAgentGenerator(seed=42)

    def test_w13_profiling_weights(self, w13_gen):
        """Profiling n=50 -> 35 easy, 10 medium, 5 hard (70/20/10)."""
        inputs = w13_gen.generate_batch("profiling", 50)
        counts = Counter(i.tier for i in inputs)
        assert counts["easy"] == 35
        assert counts["medium"] == 10
        assert counts["hard"] == 5
        # No edge tier in profiling
        assert counts.get("edge", 0) == 0

    def test_w13_ground_truth_weights(self, w13_gen):
        """GT n=500 -> 275 easy, 125 medium, 75 hard, 25 edge."""
        inputs = w13_gen.generate_batch("ground_truth", 500)
        counts = Counter(i.tier for i in inputs)
        assert counts["easy"] == 275
        assert counts["medium"] == 125
        assert counts["hard"] == 75
        assert counts["edge"] == 25

    def test_w13_no_extreme_in_gt(self, w13_gen):
        """W13 GT has no extreme tier (not in its weight config)."""
        inputs = w13_gen.generate_batch("ground_truth", 500)
        extreme_count = sum(1 for i in inputs if i.tier == "extreme")
        assert extreme_count == 0


# ===================================================================
# 4. W5 modality drift
# ===================================================================


class TestW5Modality:
    """W5 enforces modality ratios at the batch level."""

    @pytest.fixture
    def w5_gen(self):
        mod = importlib.import_module("inputs.generators.w05_multimodal_extraction")
        return mod.MultimodalExtractionGenerator(seed=42)

    def test_w5_profiling_modality_ratio(self, w5_gen):
        """Profiling should have ~70% text modality (within tolerance)."""
        inputs = w5_gen.generate_batch("profiling", 50)
        modalities = Counter(i.structural_descriptor["modality"] for i in inputs)
        text_ratio = modalities.get("text", 0) / len(inputs)
        # Target is 70% for non-edge; with edge inputs being image/mixed,
        # the overall text ratio will be slightly below 70%.
        assert 0.55 <= text_ratio <= 0.80, f"Text ratio {text_ratio:.2f} outside expected range"

    def test_w5_ground_truth_modality_ratio(self, w5_gen):
        """GT should have ~40% text modality (within tolerance)."""
        inputs = w5_gen.generate_batch("ground_truth", 50)
        modalities = Counter(i.structural_descriptor["modality"] for i in inputs)
        text_ratio = modalities.get("text", 0) / len(inputs)
        assert 0.25 <= text_ratio <= 0.50, f"Text ratio {text_ratio:.2f} outside expected range"

    def test_w5_modality_field_present(self, w5_gen):
        """Every W5 input should have modality in structural_descriptor."""
        inputs = w5_gen.generate_batch("profiling", 50)
        for inp in inputs:
            assert "modality" in inp.structural_descriptor
            assert inp.structural_descriptor["modality"] in {"text", "image", "mixed"}


# ===================================================================
# 5. W19 session depth
# ===================================================================


class TestW19SessionDepth:
    """W19 generates 8-turn conversations with session depth drift."""

    @pytest.fixture
    def w19_gen(self):
        mod = importlib.import_module("inputs.generators.w19_multi_turn")
        return mod.MultiTurnGenerator(seed=42)

    def test_w19_profiling_substantive_turns(self, w19_gen):
        """Profiling mean substantive turn count should be <= 5.5."""
        inputs = w19_gen.generate_batch("profiling", 50)
        counts = [i.structural_descriptor["substantive_turn_count"] for i in inputs]
        mean = statistics.mean(counts)
        assert mean <= 5.5, f"Mean substantive turns {mean:.2f} exceeds 5.5"

    def test_w19_ground_truth_substantive_turns(self, w19_gen):
        """GT mean substantive turn count should be >= 6.5."""
        inputs = w19_gen.generate_batch("ground_truth", 50)
        counts = [i.structural_descriptor["substantive_turn_count"] for i in inputs]
        mean = statistics.mean(counts)
        assert mean >= 6.5, f"Mean substantive turns {mean:.2f} below 6.5"

    def test_w19_always_8_turns(self, w19_gen):
        """Every W19 input must have exactly 8 turns."""
        for profile in ("profiling", "ground_truth"):
            inputs = w19_gen.generate_batch(profile, 50)
            for inp in inputs:
                turns = inp.input_data["turns"]
                assert len(turns) == 8, f"{inp.id}: expected 8 turns, got {len(turns)}"

    def test_w19_turns_non_empty(self, w19_gen):
        """No turn should be empty."""
        inputs = w19_gen.generate_batch("profiling", 50)
        for inp in inputs:
            for i, turn in enumerate(inp.input_data["turns"]):
                assert turn, f"{inp.id} turn {i} is empty"


# ===================================================================
# 6. W17 claims template tests
# ===================================================================


class TestW17Claims:
    """W17 generates structured JSON claims with known pipeline triggers."""

    @pytest.fixture
    def w17_gen(self):
        mod = importlib.import_module("inputs.generators.w17_claims_agent")
        return mod.W17ClaimsAgentGenerator(seed=42)

    def test_w17_all_providers_represented(self, w17_gen):
        """All three providers (United Healthcare, Aetna, Cigna) should appear."""
        inputs = w17_gen.generate_batch("profiling", 50)
        providers = {i.input_data.get("provider") for i in inputs if "provider" in i.input_data}
        assert "United Healthcare" in providers
        assert "Aetna" in providers
        assert "Cigna" in providers

    def test_w17_easy_tier_short_circuits(self, w17_gen):
        """Easy-tier claims should trigger intake short-circuit
        (inactive member or missing itemized bill)."""
        inputs = w17_gen.generate_batch("profiling", 50)
        easy_inputs = [i for i in inputs if i.tier == "easy"]
        assert len(easy_inputs) > 0, "No easy-tier inputs generated"
        for inp in easy_inputs:
            d = inp.input_data
            is_inactive = d.get("member_status") == "inactive"
            missing_bill = d.get("itemized_bill") is None
            assert is_inactive or missing_bill, (
                f"{inp.id}: easy-tier claim should short-circuit "
                f"(inactive={is_inactive}, missing_bill={missing_bill})"
            )

    def test_w17_edge_has_flags(self, w17_gen):
        """Edge-tier claims should have high amount (>5000) or mismatched codes."""
        mod = importlib.import_module("inputs.generators.w17_claims_agent")
        matched_set = {pair for pair in mod._MATCHED_CODE_PAIRS}

        inputs = w17_gen.generate_batch("profiling", 50)
        edge_inputs = [i for i in inputs if i.tier == "edge"]
        assert len(edge_inputs) > 0, "No edge-tier inputs generated"
        for inp in edge_inputs:
            d = inp.input_data
            high_amount = d.get("claimed_amount", 0) > 5000
            code_pair = (d.get("diagnosis_code"), d.get("procedure_code"))
            mismatched = code_pair not in matched_set
            assert high_amount or mismatched, (
                f"{inp.id}: edge claim should be flagged "
                f"(amount={d.get('claimed_amount')}, mismatched={mismatched})"
            )

    def test_w17_claim_json_structure(self, w17_gen):
        """Every W17 input_data should have the expected claim keys."""
        inputs = w17_gen.generate_batch("profiling", 50)
        required_keys = {
            "claim_id",
            "member_id",
            "member_status",
            "claim_type",
            "provider",
            "diagnosis_code",
            "procedure_code",
            "claimed_amount",
            "service_date",
        }
        for inp in inputs:
            missing = required_keys - set(inp.input_data.keys())
            assert not missing, f"{inp.id}: missing keys {missing}"


# ===================================================================
# 7. W01 determinism and W1/W11 sharing
# ===================================================================


class TestW1Determinism:
    """W01SupportSimpleGenerator with seed=42 produces deterministic results."""

    def test_w01_deterministic_across_instances(self):
        """Two instances with same seed produce identical outputs."""
        mod = importlib.import_module("inputs.generators.w01_support_simple")
        gen_a = mod.W01SupportSimpleGenerator(seed=42)
        gen_b = mod.W01SupportSimpleGenerator(seed=42)
        inputs_a = gen_a.generate_batch("profiling", 20)
        inputs_b = gen_b.generate_batch("profiling", 20)
        ids_a = [i.id for i in inputs_a]
        ids_b = [i.id for i in inputs_b]
        assert ids_a == ids_b
        tiers_a = [i.tier for i in inputs_a]
        tiers_b = [i.tier for i in inputs_b]
        assert tiers_a == tiers_b


class TestW1W11Sharing:
    """W11 must produce identical content to W1 with workflow field changed."""

    def test_w11_content_identical_to_w1(self):
        w1_mod = importlib.import_module("inputs.generators.w01_support_simple")
        w11_mod = importlib.import_module("inputs.generators.w11_support_qwen")
        w1 = w1_mod.W01SupportSimpleGenerator(seed=42).generate_batch("profiling", 50)
        w11 = w11_mod.W11SupportQwenGenerator(seed=42).generate_batch("profiling", 50)
        for a, b in zip(w1, w11, strict=True):
            assert a.input_data == b.input_data

    def test_w11_workflow_field_is_w11(self):
        w11_mod = importlib.import_module("inputs.generators.w11_support_qwen")
        inputs = w11_mod.W11SupportQwenGenerator(seed=42).generate_batch("profiling", 10)
        for inp in inputs:
            assert inp.workflow == "W11"

    def test_w11_tiers_match_w1(self):
        w1_mod = importlib.import_module("inputs.generators.w01_support_simple")
        w11_mod = importlib.import_module("inputs.generators.w11_support_qwen")
        w1 = w1_mod.W01SupportSimpleGenerator(seed=42).generate_batch("profiling", 50)
        w11 = w11_mod.W11SupportQwenGenerator(seed=42).generate_batch("profiling", 50)
        assert [a.tier for a in w1] == [b.tier for b in w11]

    def test_w11_dirty_flags_match_w1(self):
        w1_mod = importlib.import_module("inputs.generators.w01_support_simple")
        w11_mod = importlib.import_module("inputs.generators.w11_support_qwen")
        w1 = w1_mod.W01SupportSimpleGenerator(seed=42).generate_batch("profiling", 50)
        w11 = w11_mod.W11SupportQwenGenerator(seed=42).generate_batch("profiling", 50)
        assert [a.is_dirty for a in w1] == [b.is_dirty for b in w11]


# ===================================================================
# 8. Save and load round-trip
# ===================================================================


class TestSaveLoad:
    """Verify save_batch writes valid JSON and round-trips cleanly."""

    def test_save_and_load_round_trip(self, tmp_path):
        """Generate W01 profiling n=5, save, load, verify round-trip."""
        mod = importlib.import_module("inputs.generators.w01_support_simple")
        gen = mod.W01SupportSimpleGenerator(seed=42)
        inputs = gen.generate_batch("profiling", 5)
        output_dir = tmp_path / "w01_profiling"
        gen.save_batch(inputs, str(output_dir))

        # Load all JSON files
        json_files = list(output_dir.glob("*.json"))
        assert len(json_files) == 5

        # Build lookup by ID so ordering does not matter
        originals_by_id = {i.id: i for i in inputs}

        for filepath in json_files:
            loaded = json.loads(filepath.read_text())
            original = originals_by_id[loaded["id"]]
            assert loaded["workflow"] == original.workflow
            assert loaded["profile"] == original.profile
            assert loaded["tier"] == original.tier
            assert loaded["token_count"] == original.token_count
            assert loaded["is_dirty"] == original.is_dirty
            assert loaded["dirty_type"] == original.dirty_type

    def test_save_creates_directory(self, tmp_path):
        """save_batch should create missing parent directories."""
        mod = importlib.import_module("inputs.generators.w01_support_simple")
        gen = mod.W01SupportSimpleGenerator(seed=42)
        inputs = gen.generate_batch("profiling", 2)
        nested_dir = tmp_path / "nested" / "deep" / "output"
        gen.save_batch(inputs, str(nested_dir))
        assert nested_dir.exists()
        assert len(list(nested_dir.glob("*.json"))) == 2

    def test_saved_json_is_valid(self, tmp_path):
        """Every saved file should be valid JSON."""
        mod = importlib.import_module("inputs.generators.w01_support_simple")
        gen = mod.W01SupportSimpleGenerator(seed=42)
        inputs = gen.generate_batch("profiling", 3)
        gen.save_batch(inputs, str(tmp_path))
        for filepath in tmp_path.glob("*.json"):
            data = json.loads(filepath.read_text())
            assert isinstance(data, dict)
            assert "id" in data
            assert "input_data" in data


# ===================================================================
# 9. Agent compatibility: generated inputs have expected structure
# ===================================================================


class TestAgentCompatibility:
    """Verify generated inputs have the structure agents expect."""

    def test_w01_has_customer_message(self):
        """W01 inputs should contain 'customer_message' and 'input' keys."""
        mod = importlib.import_module("inputs.generators.w01_support_simple")
        gen = mod.W01SupportSimpleGenerator(seed=42)
        inputs = gen.generate_batch("profiling", 3)
        for inp in inputs:
            assert "customer_message" in inp.input_data
            assert "input" in inp.input_data
            assert isinstance(inp.input_data["customer_message"], str)

    def test_w17_claim_json_parseable(self):
        """W17 input_data should be serializable to valid JSON."""
        mod = importlib.import_module("inputs.generators.w17_claims_agent")
        gen = mod.W17ClaimsAgentGenerator(seed=42)
        inputs = gen.generate_batch("profiling", 3)
        for inp in inputs:
            # Ensure claim data is valid JSON-serializable
            serialized = json.dumps(inp.input_data)
            reparsed = json.loads(serialized)
            assert reparsed["claim_id"].startswith("CLM-")
            assert reparsed["member_id"].startswith("MEM-")
            assert isinstance(reparsed["claimed_amount"], (int, float))

    def test_w19_has_turns_and_conversation_script(self):
        """W19 input_data should contain 'turns' list and 'conversation_script'."""
        mod = importlib.import_module("inputs.generators.w19_multi_turn")
        gen = mod.MultiTurnGenerator(seed=42)
        inputs = gen.generate_batch("profiling", 3)
        for inp in inputs:
            assert "turns" in inp.input_data
            assert "conversation_script" in inp.input_data
            assert isinstance(inp.input_data["turns"], list)
            assert isinstance(inp.input_data["conversation_script"], list)
            # Conversation script should have turn numbers
            for entry in inp.input_data["conversation_script"]:
                assert "turn" in entry
                assert "user_message" in entry

    def test_w05_has_modality_and_content(self):
        """W5 input_data should contain 'modality', 'content', and 'input'."""
        mod = importlib.import_module("inputs.generators.w05_multimodal_extraction")
        gen = mod.MultimodalExtractionGenerator(seed=42)
        inputs = gen.generate_batch("profiling", 3)
        for inp in inputs:
            assert "modality" in inp.input_data
            assert "content" in inp.input_data
            assert "input" in inp.input_data
            assert inp.input_data["modality"] in {"text", "image", "mixed"}

    def test_w13_has_user_query(self):
        """W13 input_data should contain 'user_query' and 'input'."""
        mod = importlib.import_module("inputs.generators.w13_routing_agent")
        gen = mod.W13RoutingAgentGenerator(seed=42)
        inputs = gen.generate_batch("profiling", 3)
        for inp in inputs:
            assert "user_query" in inp.input_data
            assert "input" in inp.input_data
            assert isinstance(inp.input_data["user_query"], str)


# ===================================================================
# Additional edge-case and invariant tests
# ===================================================================


class TestGeneratedInputDataclass:
    """Verify GeneratedInput dataclass behavior."""

    def test_to_dict_round_trip(self):
        """to_dict() should produce a dict with all fields."""
        inp = GeneratedInput(
            id="test_001",
            workflow="TEST",
            profile="profiling",
            tier="easy",
            token_count=42,
            is_dirty=False,
            dirty_type=None,
            structural_descriptor={"key": "value"},
            input_data={"input": "hello"},
        )
        d = inp.to_dict()
        assert d["id"] == "test_001"
        assert d["workflow"] == "TEST"
        assert d["profile"] == "profiling"
        assert d["tier"] == "easy"
        assert d["token_count"] == 42
        assert d["is_dirty"] is False
        assert d["dirty_type"] is None
        assert d["structural_descriptor"] == {"key": "value"}
        assert d["input_data"] == {"input": "hello"}

    def test_to_dict_contains_exactly_expected_keys(self):
        """to_dict() should contain exactly the 9 expected keys."""
        inp = GeneratedInput(
            id="x",
            workflow="W",
            profile="p",
            tier="t",
            token_count=1,
            is_dirty=False,
            dirty_type=None,
            structural_descriptor={},
            input_data={},
        )
        expected_keys = {
            "id",
            "workflow",
            "profile",
            "tier",
            "token_count",
            "is_dirty",
            "dirty_type",
            "structural_descriptor",
            "input_data",
        }
        assert set(inp.to_dict().keys()) == expected_keys


class TestStyleShift:
    """Verify apply_style_shift behavior."""

    def test_profiling_no_style_shift(self):
        """Style shift should be a no-op for profiling inputs."""
        gen = _StubGenerator(seed=42)
        original = "This is a test sentence with some words."
        result = gen.apply_style_shift(original, "profiling")
        assert result == original

    def test_ground_truth_may_shift(self):
        """Over many calls, GT style shift should sometimes modify text."""
        gen = _StubGenerator(seed=42)
        text = "This is a test sentence with some words in it."
        modified_count = 0
        for _ in range(50):
            result = gen.apply_style_shift(text, "ground_truth")
            if result != text:
                modified_count += 1
        # With 70% application rate, expect roughly 35 modifications
        assert modified_count > 0, "Style shift never modified GT text"


class TestEstimateTokens:
    """Verify rough token estimation."""

    def test_estimate_tokens_basic(self):
        gen = _StubGenerator(seed=42)
        assert gen.estimate_tokens("test") == 1  # 4 chars / 4 = 1
        assert gen.estimate_tokens("hello world!!") == 3  # 13 chars / 4 = 3
        assert gen.estimate_tokens("") == 1  # max(1, 0) = 1

    def test_estimate_tokens_always_positive(self):
        gen = _StubGenerator(seed=42)
        assert gen.estimate_tokens("") >= 1
        assert gen.estimate_tokens("a") >= 1


class TestEmptyDirtyTypes:
    """Generators with no dirty_types should produce zero dirty inputs."""

    def test_no_dirty_with_empty_dirty_types(self):
        class CleanGenerator(BaseInputGenerator):
            workflow_id = "CLEAN"
            dirty_types = []

            def generate_single(self, tier, profile, rng, idx, is_dirty=False, dirty_type=None):
                return GeneratedInput(
                    id=self.make_id(profile, tier, idx),
                    workflow="CLEAN",
                    profile=profile,
                    tier=tier,
                    token_count=50,
                    is_dirty=is_dirty,
                    dirty_type=dirty_type,
                    structural_descriptor={},
                    input_data={"input": "clean"},
                )

        gen = CleanGenerator(seed=42)
        inputs = gen.generate_batch("profiling", 50)
        dirty_count = sum(1 for i in inputs if i.is_dirty)
        assert dirty_count == 0


class TestTierWeightsProperty:
    """Verify default tier_weights returns both profiling and GT configs."""

    def test_default_weights_have_both_profiles(self):
        gen = _StubGenerator(seed=42)
        weights = gen.tier_weights
        assert "profiling" in weights
        assert "ground_truth" in weights

    def test_default_profiling_weights_match_constants(self):
        gen = _StubGenerator(seed=42)
        assert gen.tier_weights["profiling"] == PROFILING_WEIGHTS

    def test_default_gt_weights_match_constants(self):
        gen = _StubGenerator(seed=42)
        assert gen.tier_weights["ground_truth"] == GROUND_TRUTH_WEIGHTS

    def test_profiling_weights_sum_to_one(self):
        total = sum(PROFILING_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9

    def test_ground_truth_weights_sum_to_one(self):
        total = sum(GROUND_TRUTH_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9
