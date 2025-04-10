import sys
from agents.sec_agent.get_SEC_data import SECDataRetrieval


def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    sec_data = SECDataRetrieval(ticker)
    print(sec_data.extract_risk_factors())
    # print("\n")
    balance_sheets = sec_data.extract_balance_sheet_as_json()
    print(balance_sheets["tenk"])
    print("\n")
    print(balance_sheets["tenq"])
    # print("\n")
    print(sec_data.extract_management_discussion())


if __name__ == "__main__":
    main()
