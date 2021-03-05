import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd


def load_ibtracs():
    return pd.read_csv("data/ibtracs_1980-2020_simplified.csv")


if __name__ == "__main__":
    ibtracs = load_ibtracs()
