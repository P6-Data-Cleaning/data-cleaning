import dask.dataframe as dd
import time

def measure_performance(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} execution time: {time.time() - start_time:.2f} seconds")
        return result
    return wrapper

@measure_performance
def missing_time(df):
    df['# Timestamp'] = dd.to_datetime(
        df['# Timestamp'], format='%d/%m/%Y %H:%M:%S', errors='coerce'
    )

    df = df.shuffle(on="MMSI")

    def process_partition(pdf):
        if pdf.empty:
            return pdf

        pdf = pdf.sort_values(['MMSI', '# Timestamp'])

        # Calculate time differences (seconds)
        pdf['time_diff'] = pdf.groupby('MMSI')['# Timestamp'].diff().dt.total_seconds()

        pdf['max_diff'] = pdf.groupby('MMSI')['time_diff'].transform('max')

        # Filter out any MMSI whose max gap is over one hour
        pdf = pdf.loc[pdf['max_diff'] <= 3600]
        pdf = pdf.drop(columns=['max_diff', 'time_diff'])

        return pdf

    # Apply the function to each partition
    return df.map_partitions(process_partition)