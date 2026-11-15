"""The outer-off pair changes the intended numerical control, not the schedule."""
import pytest

from test_fp4_training_recipe import _build


def options(admission=False):
    return dict(
        outer_off_pair=True, admission=admission, steps=8 if admission else 3000,
        comparison_rmsnorm_backend="torch", sr_rng="triton_philox",
        value_checks="grouped_deferred",
    )


@pytest.mark.parametrize("recipe", ["nvfp4", "amdfp4"])
@pytest.mark.parametrize("admission", [False, True])
def test_outer_off_changes_only_outer_setting_against_existing_control(recipe, admission):
    config = options(admission)
    document, args, manifest = _build(recipe, **config)
    old = {key: value for key, value in config.items() if key != "outer_off_pair"}
    old["supplemental_dp4" if admission else "dp4_3000"] = True
    _, reference, prior = _build(recipe, **old)
    assert args["fp4_outer_scope"] == "none" and reference["fp4_outer_scope"] == "tensor"
    assert {k: v for k, v in args.items() if k != "fp4_outer_scope"} == {
        k: v for k, v in reference.items() if k != "fp4_outer_scope"
    }
    assert manifest["environment"] == prior["environment"]
    assert manifest["full_geometry"] == prior["full_geometry"]
    assert manifest["deosc_settings"] == prior["deosc_settings"]
    assert manifest["validation_steps"] == prior["validation_steps"]
    assert manifest["scope"] == ("outer_off_dp4_admission" if admission else "outer_off_dp4_3000")
    assert manifest["outer_scale_policy"] == "fixed_one"
    assert manifest["outer_scale_value"] == 1.0
    assert manifest["comparison_recipes"] == ["nvfp4", "amdfp4"]
    assert manifest["expected_world_size"] == 4
    assert manifest["gradient_accumulation_steps"] == 2
    assert "outer_off" in document["exp_name"]


@pytest.mark.parametrize("override", [
    {"recipe": "mxfp4"}, {"steps": 128}, {"steps": 8}, {"admission": True},
    {"supplemental_dp4": True}, {"supplemental_dp4_prefix": True}, {"dp4_3000": True},
    {"recompute": False}, {"comparison_rmsnorm_backend": "native"},
    {"sr_rng": "torch_cpu"}, {"value_checks": "strict"},
])
def test_outer_off_pair_rejects_wrong_matrix_or_changed_controls(override):
    config = {"recipe": "amdfp4", **options(), **override}
    with pytest.raises(ValueError, match="Outer-off pair"):
        _build(**config)
