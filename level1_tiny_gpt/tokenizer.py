# tokenizer.py
# Converts text <--> numbers so our model can process it

class CharTokenizer:
    """
    A character-level tokenizer.
    It builds a vocabulary from whatever text you give it,
    then can encode (text -> numbers) and decode (numbers -> text).
    """

    def __init__(self, text):
        # Find every unique character in the text
        # sorted() makes sure the order is consistent every run
        self.chars = sorted(set(text))

        # How many unique characters do we have?
        self.vocab_size = len(self.chars)

        # Build two lookup tables:
        # char_to_int: given a character, what number is it?
        self.char_to_int = { ch: i for i, ch in enumerate(self.chars) }

        # int_to_char: given a number, what character is it?
        self.int_to_char = { i: ch for i, ch in enumerate(self.chars) }

    def encode(self, text):
        """Convert a string of text into a list of integers."""
        return [self.char_to_int[ch] for ch in text]

    def decode(self, integers):
        """Convert a list of integers back into a string."""
        return ''.join([self.int_to_char[i] for i in integers])


# ---- Quick test (only runs when you run this file directly) ----
if __name__ == "__main__":
    # Read our training text
    with open("data/input.txt", "r") as f:
        text = f.read()

    # Build the tokenizer from that text
    tokenizer = CharTokenizer(text)

    print(f"Total characters in text : {len(text)}")
    print(f"Unique characters (vocab): {tokenizer.vocab_size}")
    print(f"All characters: {''.join(tokenizer.chars)}")
    print()

    # Test encode and decode
    sample = "Hello"
    # Note: only works for characters that exist in input.txt
    sample = "First Citizen"
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)

    print(f"Original : {sample}")
    print(f"Encoded  : {encoded}")
    print(f"Decoded  : {decoded}")