import torch
import sys
import os

from concurrent.futures import ProcessPoolExecutor, as_completed

def truncate_key(key, depth=3):
    """
    Truncates the key to the specified depth by joining the first `depth` components.
    
    Args:
        key (str): Original key.
        depth (int): Depth to truncate the key (default is 3).
        
    Returns:
        str: Truncated key up to the specified depth.
    """
    return '.'.join(key.split('.')[:depth])

def compare_state_dicts(state_dict1, state_dict2):
    """
    Compares two PyTorch state dicts to check if the tensors inside are equal within a given tolerance.

    Args:
        state_dict1 (dict): First state dictionary.
        state_dict2 (dict): Second state dictionary.
        tol (float): Tolerance for floating point comparison (default is 1e-6).

    Returns:
        bool: True if all tensors are equal within the tolerance, False otherwise.
    """
    # Default dicts have some unused keys that we need to remove
    state_dict2 = {k: v for k, v in state_dict2.items() if "core_attention" not in k}

    # Get unique truncated keys up to depth of 3
    truncated_keys1 = {truncate_key(key) for key in state_dict1.keys()}
    truncated_keys2 = {truncate_key(key) for key in state_dict2.keys()}

    # Check for keys that are only in one of the dictionaries
    only_in_dict1 = truncated_keys1 - truncated_keys2
    only_in_dict2 = truncated_keys2 - truncated_keys1

    if only_in_dict1 or only_in_dict2:
        print("Unique keys up to depth 3 that differ between the state dicts:")
        if only_in_dict1:
            print("Keys only in the first state dict:", only_in_dict1)
        if only_in_dict2:
            print("Keys only in the second state dict:", only_in_dict2)
        return False

    # Check if both state dicts have the same keys
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())

    # Check if both state dicts have the same keys
    if keys1 != keys2:
        print("State dicts have different keys.")
        # Print keys that are in one dict but not the other
        only_in_dict1 = keys1 - keys2
        only_in_dict2 = keys2 - keys1
        if only_in_dict1:
            print("Keys only in the first state dict:", only_in_dict1)
        if only_in_dict2:
            print("Keys only in the second state dict:", only_in_dict2)
        return False

    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = []
        for key in state_dict1.keys():
            tensor1 = state_dict1[key]
            tensor2 = state_dict2[key]
            futures.append(executor.submit(compare_values, key, tensor1, tensor2))

        # Collect and check results
        any_fail = False
        for future in as_completed(futures):
            key, success, reason = future.result()
            print(f"Key '{key}': {reason}")
            if not success:
                any_fail = True

    # All tensors are equal
    return not any_fail

def compare_values(key, a, b, atol=1e-4, rtol=1e-4):
    if type(a) != type(b):
        if key.endswith("_extra_state"):
            return key, True, "Extra state keys are allowed to have different types"
        return key, False, f"Values have different types: {type(a)} and {type(b)}."
    
    if a is None:
        if b is not None:
            return key, False, f"Value 1 is None and the other is not: {b}."
        return key, True, "Both values are None."

    if isinstance(a, torch.Tensor):
        if a.shape != b.shape:
            return key, False, "Tensors have different shapes."
        
        if a.dtype != b.dtype:
            return key, False, "Tensors have different data types."
        
        if not torch.allclose(a, b, atol=atol, rtol=rtol):
            diff = torch.abs(a - b)
            return (
                key,
                False,
                f"Tensors are not equal within tolerance (Max diff: {diff.max().item()})."
            )
        
        return key, True, "Tensors are equal within tolerance."
    
    return key, False, f"Value 1 has unsupported type: {type(a)}."
    

def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_state_dicts.py <path_to_state_dict1> <path_to_state_dict2>")
        sys.exit(1)

    path1 = sys.argv[1]
    path2 = sys.argv[2]

    # Check if the paths exist
    if not os.path.exists(path1) or not os.path.exists(path2):
        print("One or both of the provided paths do not exist.")
        sys.exit(1)

    # Load the state dicts
    state_dict1 = torch.load(path1, map_location="cpu")
    state_dict2 = torch.load(path2, map_location="cpu")

    # Compare the state dicts
    are_equal = compare_state_dicts(state_dict1["model"], state_dict2["model"])

    if are_equal:
        print("The state dictionaries are equal.")
    else:
        print("The state dictionaries are not equal.")

if __name__ == "__main__":
    main()