import os

path = "logs/partition/Amazon2014Toys_550_LOU"

for id_ in sorted(os.listdir(path)):
    print(os.path.join(path, id_, "sid_vocab.json,"))