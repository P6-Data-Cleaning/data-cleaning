import pandas as pd
import os
import time
from duplicate import remove_duplicate


def main ():
    start_time = time.time()

    print("Reading the CSV file...")
    cleaned = remove_duplicate('aisdk-2025-02-14.csv')

    print(f"Execution time: {time.time() - start_time} seconds")


if __name__ == '__main__':
    main()


