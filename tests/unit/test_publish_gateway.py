from autoscorer.gamelogic.publish import PublishGateway, PublishMode


def test_manual_mode_never_auto_publishes():
    gateway = PublishGateway(mode=PublishMode.MANUAL)
    assert not gateway.should_auto_publish(confidence=1.0)


def test_autonomous_mode_always_auto_publishes():
    gateway = PublishGateway(mode=PublishMode.AUTONOMOUS)
    assert gateway.should_auto_publish(confidence=0.0)


def test_confidence_fallback_publishes_above_threshold_only():
    gateway = PublishGateway(mode=PublishMode.AUTONOMOUS_WITH_CONFIDENCE_FALLBACK, confidence_threshold=0.9)
    assert gateway.should_auto_publish(confidence=0.95)
    assert not gateway.should_auto_publish(confidence=0.5)


def test_mode_can_be_changed_at_runtime():
    gateway = PublishGateway(mode=PublishMode.MANUAL)
    assert not gateway.should_auto_publish()
    gateway.mode = PublishMode.AUTONOMOUS
    assert gateway.should_auto_publish()
