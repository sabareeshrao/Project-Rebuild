import pytest

from rebuild_gpt.schedule import cosine_warmup_lr


def lr(step):
    return cosine_warmup_lr(step, max_lr=1.0, min_lr=0.1, warmup_steps=2, decay_steps=6)


def test_schedule_matches_upstream_warmup_and_cosine_endpoints():
    assert lr(0) == pytest.approx(1 / 3)
    assert lr(1) == pytest.approx(2 / 3)
    assert lr(2) == pytest.approx(1)
    assert lr(4) == pytest.approx(0.55)
    assert lr(6) == pytest.approx(0.1)
    assert lr(30) == pytest.approx(0.1)


def test_no_warmup_starts_at_maximum():
    assert cosine_warmup_lr(0, max_lr=0.01, min_lr=0, warmup_steps=0, decay_steps=10) == pytest.approx(0.01)


def test_decay_never_increases_after_warmup():
    rates = [lr(t) for t in range(2, 12)]
    assert rates == sorted(rates, reverse=True)


@pytest.mark.parametrize("options,match", [
    ({"iteration": -1}, "iteration"),
    ({"warmup_steps": -1}, "warmup_steps"),
    ({"warmup_steps": 6}, "decay_steps"),
    ({"decay_steps": 2}, "decay_steps"),
    ({"max_lr": 0}, "max_lr"),
    ({"max_lr": float("nan")}, "max_lr"),
    ({"min_lr": -1}, "min_lr"),
    ({"min_lr": 2}, "min_lr"),
])
def test_schedule_rejects_invalid_configuration(options, match):
    config = dict(iteration=0, max_lr=1.0, min_lr=0.1, warmup_steps=2, decay_steps=6)
    config.update(options)
    with pytest.raises(ValueError, match=match):
        cosine_warmup_lr(**config)
