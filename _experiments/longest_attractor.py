import argparse
import json


def main():
    parser = argparse.ArgumentParser(
        description="Print controls with max min_attr_len_viol and max min_attr_len per instance."
    )
    parser.add_argument("json_file", help="Path to the checker results JSON file")
    parser.add_argument("--print-control", action="store_true", help="Print the list of controls for each maximum")
    args = parser.parse_args()

    with open(args.json_file) as f:
        data = json.load(f)

    for instance, controls in data.items():
        max_viol_val = None
        max_viol_controls = []
        max_len_val = None
        max_len_controls = []

        for control_str, metrics in controls.items():
            viol = metrics["min_attr_len_viol"]
            length = metrics["min_attr_len"]

            if max_viol_val is None or viol > max_viol_val:
                max_viol_val = viol
                max_viol_controls = [control_str]
            elif viol == max_viol_val:
                max_viol_controls.append(control_str)

            if max_len_val is None or length > max_len_val:
                max_len_val = length
                max_len_controls = [control_str]
            elif length == max_len_val:
                max_len_controls.append(control_str)

        print(f"Instance: {instance}")
        print(f"  Max min_attr_len_viol = {max_viol_val}")
        if args.print_control:
            for c in max_viol_controls:
                print(f"    {c}")
        print(f"  Max min_attr_len = {max_len_val}")
        if args.print_control:
            for c in max_len_controls:
                print(f"    {c}")
        print()


if __name__ == "__main__":
    main()
