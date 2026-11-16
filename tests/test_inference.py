import pytest
import torch
from torch import nn

from tileflow import TileConfig, tiled_predict
from tileflow.models import local_model


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


@pytest.mark.parametrize("shape", [(3, 1, 1), (3, 11, 7), (3, 33, 51), (3, 64, 64)])
@pytest.mark.parametrize("overlap,blend", [(0, "uniform"), (7, "hann")])
def test_halo_matches_whole_frame_at_edges_and_seams(shape, overlap, blend):
    image = torch.rand(shape, generator=torch.Generator().manual_seed(17))
    model = local_model(width=5)
    with torch.inference_mode():
        expected = model(image.unsqueeze(0))[0]
    actual = tiled_predict(model, image, TileConfig(16, overlap, 2, 3, blend))
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert not actual.requires_grad


def test_insufficient_context_exhibits_seams_then_halo_fixes_them():
    model = local_model(width=5)
    image = torch.rand((3, 47, 53), generator=torch.Generator().manual_seed(5))
    with torch.inference_mode():
        expected = model(image.unsqueeze(0))[0]
    bad = tiled_predict(model, image, TileConfig(16, 0, 0, 2, "uniform"))
    good = tiled_predict(model, image, TileConfig(16, 0, 2, 2, "uniform"))
    assert (bad - expected).abs().max() > 0.01
    torch.testing.assert_close(good, expected, atol=1e-6, rtol=1e-5)


def test_materialized_and_batch_controls_preserve_outputs():
    model = local_model(width=5)
    image = torch.rand((3, 39, 51))
    reference = tiled_predict(model, image, TileConfig(16, 8, 2, 1))
    for size in (2, 8):
        for materialize in (False, True):
            actual = tiled_predict(
                model, image, TileConfig(16, 8, 2, size), materialize=materialize
            )
            torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tile_size": 0},
        {"overlap": 256},
        {"halo": -1},
        {"batch_size": 0},
        {"blend": "unknown"},
        {"tile_size": 1.5},
    ],
)
def test_invalid_configuration_fails(kwargs):
    with pytest.raises(ValueError):
        TileConfig(**kwargs)


def test_training_mode_and_nonfinite_inputs_fail():
    model = local_model()
    image = torch.zeros((3, 17, 17))
    with pytest.raises(ValueError, match="eval"):
        tiled_predict(model.train(), image)
    model.eval()
    image[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        tiled_predict(model, image)


@pytest.mark.parametrize(
    "model,error",
    [
        (nn.Sequential(nn.AvgPool2d(2)), "preserve"),
        (nn.Sequential(nn.Flatten()), "preserve"),
    ],
)
def test_changed_output_geometry_fails(model, error):
    with pytest.raises(ValueError, match=error):
        tiled_predict(model.eval(), torch.rand((3, 32, 32)), TileConfig(16, 4, 2))


def test_nonfinite_model_output_fails():
    class Bad(nn.Module):
        def forward(self, image):
            return image * float("inf")

    with pytest.raises(FloatingPointError, match="non-finite"):
        tiled_predict(Bad().eval(), torch.ones((3, 12, 12)))


def test_global_context_is_deliberately_not_claimed_to_match():
    class GlobalMean(nn.Module):
        def forward(self, image):
            return image - image.mean((-2, -1), keepdim=True)

    image = torch.arange(3 * 32 * 32).reshape(3, 32, 32).float()
    model = GlobalMean().eval()
    actual = tiled_predict(model, image, TileConfig(16, 0, 2))
    assert not torch.allclose(actual, model(image.unsqueeze(0))[0])
