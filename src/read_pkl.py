import pickle

with open("../data/predicted_ingr.pkl", "rb") as fp:   # Unpickling
    b = pickle.load(fp)

print(b)

