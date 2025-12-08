"""
Validation script for OpenAI fine-tuning JSONL format.

Checks that the output file meets OpenAI's requirements:
- Each line is valid JSON
- Each object has exactly one key: "messages"
- Messages array has exactly 3 objects (system, user, assistant)
- Each message has "role" and "content" fields
- All content is string type
"""

import json
import sys
from pathlib import Path


def validate_openai_jsonl(filepath: str) -> bool:
    """Validate the JSONL file meets OpenAI's requirements."""

    if not Path(filepath).exists():
        print(f"ERROR: File not found: {filepath}")
        return False

    errors = []
    warnings = []
    line_count = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line_count = line_num

            # Skip empty lines
            if not line.strip():
                warnings.append(f"Line {line_num}: Empty line")
                continue

            try:
                obj = json.loads(line.strip())

                # Check top-level structure
                if not isinstance(obj, dict):
                    errors.append(f"Line {line_num}: Root must be an object, got {type(obj).__name__}")
                    continue

                if "messages" not in obj:
                    errors.append(f"Line {line_num}: Missing 'messages' key")
                    continue

                # Check for extra keys (optional warning)
                if len(obj.keys()) > 1:
                    extra_keys = [k for k in obj.keys() if k != "messages"]
                    warnings.append(f"Line {line_num}: Extra keys present: {extra_keys}")

                messages = obj["messages"]

                # Validate messages array
                if not isinstance(messages, list):
                    errors.append(f"Line {line_num}: 'messages' must be an array")
                    continue

                if len(messages) != 3:
                    errors.append(f"Line {line_num}: Expected 3 messages, got {len(messages)}")
                    continue

                # Validate each message
                expected_roles = ["system", "user", "assistant"]
                for i, (msg, expected_role) in enumerate(zip(messages, expected_roles)):
                    if not isinstance(msg, dict):
                        errors.append(f"Line {line_num}: Message {i} must be an object")
                        continue

                    if msg.get("role") != expected_role:
                        errors.append(
                            f"Line {line_num}: Message {i} should have role='{expected_role}', "
                            f"got '{msg.get('role')}'"
                        )

                    if "content" not in msg:
                        errors.append(f"Line {line_num}: Message {i} missing 'content' field")
                    elif not isinstance(msg.get("content"), str):
                        errors.append(
                            f"Line {line_num}: Message {i} content must be string, "
                            f"got {type(msg.get('content')).__name__}"
                        )
                    elif len(msg.get("content", "")) == 0:
                        warnings.append(f"Line {line_num}: Message {i} has empty content")

                    # Check for extra keys in message
                    expected_keys = {"role", "content"}
                    extra_msg_keys = set(msg.keys()) - expected_keys
                    if extra_msg_keys:
                        warnings.append(
                            f"Line {line_num}: Message {i} has extra keys: {extra_msg_keys}"
                        )

            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON - {e}")

    # Print results
    print(f"\n{'='*60}")
    print(f"VALIDATION RESULTS: {filepath}")
    print(f"{'='*60}")
    print(f"Total lines: {line_count}")

    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for warn in warnings[:10]:  # Show first 10
            print(f"  ⚠️  {warn}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more warnings")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for err in errors[:20]:  # Show first 20
            print(f"  ❌ {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors")
        print(f"\n{'='*60}")
        print("VALIDATION FAILED ❌")
        print(f"{'='*60}")
        return False
    else:
        print(f"\n{'='*60}")
        print("VALIDATION PASSED ✓")
        print(f"{'='*60}")
        print(f"\nThe file is valid for OpenAI fine-tuning!")
        print(f"You can upload it to: https://platform.openai.com/finetune")
        return True


def print_sample(filepath: str, num_samples: int = 2):
    """Print a few sample lines from the file."""

    print(f"\n{'='*60}")
    print(f"SAMPLE ENTRIES (first {num_samples})")
    print(f"{'='*60}\n")

    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break

            obj = json.loads(line)
            print(f"Sample {i+1}:")
            print(json.dumps(obj, indent=2, ensure_ascii=False))
            print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_openai_format.py <filepath.jsonl>")
        print("\nExample:")
        print("  python validate_openai_format.py synthetic/tutoring_finetune_openai.jsonl")
        sys.exit(1)

    filepath = sys.argv[1]

    # Validate
    is_valid = validate_openai_jsonl(filepath)

    # Print samples if valid
    if is_valid:
        print_sample(filepath)

    sys.exit(0 if is_valid else 1)
