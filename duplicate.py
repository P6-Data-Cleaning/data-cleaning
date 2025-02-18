
def remove_duplicate(file):
    cleaned = file.drop_duplicates()
    cleaned.to_csv('cleaned.csv', index=False)
    print("Duplicate rows removed and saved to 'cleaned.csv'")
    return cleaned