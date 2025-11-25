from loader import load_price_data

def main():
    df = load_price_data("SPY", start="2015-01-01")
    print(df.head())

if __name__ == "__main__":
    main()
