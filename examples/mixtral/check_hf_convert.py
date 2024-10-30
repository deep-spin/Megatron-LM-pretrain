import sys
import torch
import tqdm
from transformers import AutoModelForCausalLM

def compare_state_dicts(state_dict1, state_dict2, rtol=1e-5, atol=1e-8):
    # Compare parameter keys
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())

    keys_only_in_1 = keys1 - keys2
    keys_only_in_2 = keys2 - keys1

    if keys_only_in_1 or keys_only_in_2:
        print("State dicts have different keys.")
        if keys_only_in_1:
            print("Keys only in model 1:", keys_only_in_1)
        if keys_only_in_2:
            print("Keys only in model 2:", keys_only_in_2)
        return False
    else:
        print("State dicts have the same keys.")

    # Compare parameter tensors
    tensors_differ = False
    for key in tqdm.tqdm(keys1, desc="Comparing tensors"):
        tensor1 = state_dict1[key]
        tensor2 = state_dict2[key]
        if not torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol):
            print(f"Tensors differ at key: {key}")
            tensors_differ = True

    if not tensors_differ:
        print("All tensors are equal within the specified tolerance.")
    return not tensors_differ

def compare_configs(config1, config2):
    # Convert configs to dictionaries
    config_dict1 = config1.to_dict()
    config_dict2 = config2.to_dict()

    # Find differences
    keys1 = set(config_dict1.keys())
    keys2 = set(config_dict2.keys())

    all_keys = keys1.union(keys2)
    differences = {}

    for key in all_keys:
        val1 = config_dict1.get(key, "Key not in model 1")
        val2 = config_dict2.get(key, "Key not in model 2")
        if val1 != val2:
            differences[key] = (val1, val2)

    if differences:
        print("Configurations differ in the following keys:")
        for key, (val1, val2) in differences.items():
            print(f" - {key}:")
            print(f"    Model 1: {val1}")
            print(f"    Model 2: {val2}")
    else:
        print("Configurations are identical.")

def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_models.py <model_name_or_path1> <model_name_or_path2>")
        sys.exit(1)

    model_name1 = sys.argv[1]
    model_name2 = sys.argv[2]

    print(f"Loading model 1 from '{model_name1}'...")
    model1 = AutoModelForCausalLM.from_pretrained(model_name1)

    print(f"Loading model 2 from '{model_name2}'...")
    model2 = AutoModelForCausalLM.from_pretrained(model_name2)

    print("\nComparing configurations...")
    compare_configs(model1.config, model2.config)


    print("\nComparing state dictionaries...")
    state_dicts_equal = compare_state_dicts(model1.state_dict(), model2.state_dict())

    if state_dicts_equal:
        print("\nThe models have identical state dictionaries.")
    else:
        print("\nThe models have different state dictionaries.")

if __name__ == "__main__":
    main()