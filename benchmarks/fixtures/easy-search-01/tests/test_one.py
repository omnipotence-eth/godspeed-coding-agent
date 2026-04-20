from pkg.main import hello


def test_hello():
    assert hello() == "hi"
