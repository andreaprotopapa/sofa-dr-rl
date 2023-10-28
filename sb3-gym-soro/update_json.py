import json
import sys

def update_json(json_file, updates):
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)

        for key, new_value in updates.items():
            data[key] = new_value

        with open(json_file, 'w') as f:
            json.dump(data, f, indent=4)

        print(f"Updated JSON file: {json_file}")

    except FileNotFoundError:
        print(f"JSON file not found: {json_file}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {json_file}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python update_json.py <json_file> <key1> <new_value1> [<key2> <new_value2> ...]")
        sys.exit(1)

    json_file = sys.argv[1]
    updates = {}
    
    # Parse the key-value pairs from command-line arguments
    for i in range(2, len(sys.argv), 2):
        key = sys.argv[i]
        new_value = sys.argv[i + 1]
        # argument is a string: convert in int or float
        try:
            updates[key] = [float(new_value) if "." in new_value else int(new_value)]
        except ValueError:
            # If conversion fails, keep the item as is
            updates[key] = [new_value]

    update_json(json_file, updates)