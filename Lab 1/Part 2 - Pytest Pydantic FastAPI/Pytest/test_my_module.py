from my_module import square

def test_square_gives_correct_value(input_value):
    subject = square(input_value)
    assert subject == 16