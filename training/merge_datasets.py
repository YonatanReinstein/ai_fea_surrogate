import torch, json, os, argparse


def merge(input_dirs, output_dir):
    dataset, metadata = [], []
    for d in input_dirs:
        pt_path = f"{d}/dataset.pt"
        meta_path = f"{d}/metadata.json"
        if not os.path.exists(pt_path):
            print(f"WARNING: {pt_path} not found, skipping.")
            continue
        part = torch.load(pt_path, weights_only=False)
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"  {d}: {len(part)} samples")
        dataset.extend(part)
        metadata.extend(meta)

    os.makedirs(output_dir, exist_ok=True)
    torch.save(dataset, f"{output_dir}/dataset.pt")
    with open(f"{output_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Merged {len(dataset)} samples -> {output_dir}/dataset.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dirs", nargs="+", required=True, help="Partial dataset directories to merge.")
    parser.add_argument("--output_dir", required=True, help="Destination directory for merged dataset.")
    args = parser.parse_args()
    merge(args.input_dirs, args.output_dir)
