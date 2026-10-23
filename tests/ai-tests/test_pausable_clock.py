from src.experiment.pausable_clock import PausableClock


def test_now_advances_with_the_underlying_clock(fake_time):
    clock = PausableClock()
    assert clock.now() == 0.0
    fake_time(1.0)
    assert clock.now() == 1.0


def test_pause_freezes_now(fake_time):
    clock = PausableClock()
    fake_time(1.0)
    clock.pause()
    assert clock.is_paused
    fake_time(5.0)  # time passes while paused
    assert clock.now() == 1.0


def test_resume_continues_from_where_it_left_off(fake_time):
    clock = PausableClock()
    fake_time(1.0)
    clock.pause()
    fake_time(5.0)  # 5s of "dead" time while paused
    clock.resume()
    assert not clock.is_paused
    assert clock.now() == 1.0  # still 1.0 right at resume
    fake_time(0.5)
    assert clock.now() == 1.5  # ticks normally afterward


def test_resume_without_pause_is_a_noop(fake_time):
    clock = PausableClock()
    fake_time(2.0)
    clock.resume()
    assert clock.now() == 2.0


def test_double_pause_does_not_reset_the_pause_point(fake_time):
    clock = PausableClock()
    fake_time(1.0)
    clock.pause()
    fake_time(1.0)
    clock.pause()  # already paused - should not move the freeze point
    fake_time(1.0)
    clock.resume()
    assert clock.now() == 1.0
