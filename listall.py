import os

# # path = "logs/Iters/Amazon2014Beauty_550_LOU/0304092437"
# # path = "logs/Iters/Amazon2014Toys_550_LOU/0304092426"
# # path = "logs/Iters/Amazon2014Tools_550_LOU/0304092444"
# path = "logs/Iters/Amazon2014Sports_550_LOU/0304092451"

# path = "logs/partition/Amazon2014Toys_550_LOU"
path = "logs/partition/Amazon2014Tools_550_LOU"

for id_ in sorted(os.listdir(path)):
    print(os.path.join(path, id_, "sem_id.pkl,"))

# for i in [1, 5, 10, 100, 200, 500, 1000]:
#     print(os.path.join(path, f"sem_id_{i}.pkl,"))
