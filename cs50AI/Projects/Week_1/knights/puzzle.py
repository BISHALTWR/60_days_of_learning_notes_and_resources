from logic import *

AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

# Puzzle 0
# A says "I am both a knight and a knave."
knowledge0 = And(
    # A says "I am both a knight and a knave."
    Biconditional(AKnight, And(AKnight, AKnave)),
    Biconditional(Not(AKnight), AKnave), # Cant be other if one
    Biconditional(Not(AKnave), AKnight), # Cant be other if one
    # A cannot be both a knight and a knave
    Not(And(AKnight, AKnave))
)


# Puzzle 1
# A says "We are both knaves."
# B says nothing.
knowledge1 = And(
    Biconditional(AKnight,And(AKnave, BKnave)), # What A said
    # ,True # Is B trivially true?

    # Game Rules
    Not(And(AKnight, AKnave)),
    Not(And(BKnight, BKnave)),
    Biconditional(Not(AKnight), AKnave), # Cant be other if one
    Biconditional(Not(AKnave), AKnight), # Cant be other if one
    Biconditional(Not(BKnight), BKnave), # Cant be other if one
    Biconditional(Not(BKnave), BKnight), # Cant be other if one
)

# Puzzle 2
# A says "We are the same kind."
# B says "We are of different kinds."
knowledge2 = And(
    Biconditional(AKnight, Biconditional(AKnight, BKnight)),
    Biconditional(BKnight, Not(Biconditional(AKnight, BKnight))),
    Not(And(AKnight, AKnave)),
    Not(And(BKnight, BKnave)),

    Biconditional(Not(AKnight), AKnave), # Cant be other if one
    Biconditional(Not(AKnave), AKnight), # Cant be other if one
)

# Puzzle 3
# A says either "I am a knight." or "I am a knave.", but you don't know which.
# B says "A said 'I am a knave'."
# B says "C is a knave."
# C says "A is a knight."
knowledge3 = And(
    Or(AKnight, AKnave),
    Biconditional(BKnight, Biconditional(AKnight, AKnave)),
    Biconditional(BKnight, CKnave),
    Biconditional(CKnight, AKnight),

    # These things below are game rules (No one can be both)
    Not(And(AKnight, AKnave)),
    Not(And(BKnight, BKnave)),
    Not(And(CKnight, CKnave)),

    Biconditional(Not(AKnight), AKnave), # Cant be other if one
    Biconditional(Not(AKnave), AKnight), # Cant be other if one
    Biconditional(Not(BKnight), BKnave), # Cant be other if one
    Biconditional(Not(BKnave), BKnight), # Cant be other if one
    Biconditional(Not(CKnight), CKnave), # Cant be other if one
    Biconditional(Not(CKnave), CKnight), # Cant be other if one
)


def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3)
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")


if __name__ == "__main__":
    main()
