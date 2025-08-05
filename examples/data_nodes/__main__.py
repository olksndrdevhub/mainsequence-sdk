import argparse

from examples.data_nodes.simple_simulated_prices import test_simulated_prices
from examples.data_nodes.simple_data_nodes import build_test_time_series


def main():
    parser = argparse.ArgumentParser(
        description="Run data node functions: simulated prices or test time series."
    )
    parser.add_argument(
        "command",
        choices=["simulated_prices", "test_time_series"],
        help="Function to run: choose 'simulated_prices' or 'test_time_series'"
    )
    args = parser.parse_args()

    if args.command == "simulated_prices":
        test_simulated_prices()
    elif args.command == "test_time_series":
        build_test_time_series()


if __name__ == "__main__":
    main()
