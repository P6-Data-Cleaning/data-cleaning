import pandas as pd
import os
import time
from cleaning import cleaning
import dask.dataframe as dd

def main ():
    start_time = time.time()

    print(f"Execution time: {time.time() - start_time} seconds")


if __name__ == '__main__':
    main()


