import pandas as pd


def run_eda(data_path: str = "data/census.csv"):
    df = pd.read_csv(data_path)

    # Strip whitespace from column names and string columns
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    print("=== Dataset Info ===")
    print(df.info())
    print("\n=== Missing Values ('?') ===")
    print((df == "?").sum()[(df == "?").sum() > 0])

    print("\n=== Target Class Distribution ===")
    print(df["salary"].value_counts(normalize=True))

    print("\n=== First 5 Rows ===")
    print(df.head())


if __name__ == "__main__":
    run_eda()