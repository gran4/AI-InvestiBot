

manager = ResourceManager(100, max_percent_used=70, max_in_one_stock=100)

def test_max_percent_used():
    assert manager.check(13, "None") == 13