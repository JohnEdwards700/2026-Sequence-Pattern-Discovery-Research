import pytest
from src.data.tokenizer import Tokenizer

def test_tokenization():
    tokenizer = Tokenizer()
    
    # Test with a simple protein sequence
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    tokens = tokenizer.tokenize(sequence)
    
    assert len(tokens) == len(sequence)
    assert all(isinstance(token, int) for token in tokens)

def test_special_tokens():
    tokenizer = Tokenizer()
    
    # Test special tokens
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    tokens = tokenizer.tokenize(sequence, add_special_tokens=True)
    
    assert tokens[0] == tokenizer.cls_token_id
    assert tokens[-1] == tokenizer.sep_token_id

def test_decoding():
    tokenizer = Tokenizer()
    
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    tokens = tokenizer.tokenize(sequence)
    decoded_sequence = tokenizer.decode(tokens)
    
    assert decoded_sequence == sequence

def test_tokenizer_edge_cases():
    tokenizer = Tokenizer()
    
    # Test empty sequence
    tokens = tokenizer.tokenize("")
    assert tokens == []

    # Test sequence with invalid characters
    sequence = "ACDXYZ"
    tokens = tokenizer.tokenize(sequence)
    assert len(tokens) < len(sequence)  # Some characters should be ignored

@pytest.mark.parametrize("sequence, expected_length", [
    ("ACDEFGHIKLMNPQRSTVWY", 20),
    ("", 0),
    ("ACDXYZ", 3),  # Assuming XYZ are invalid
])
def test_tokenization_parametrized(sequence, expected_length):
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize(sequence)
    assert len(tokens) == expected_length