source ../../../env/bin/activate
hf download yoyo496/fea-bistable-dataset dataset_a.pt dataset_b.pt --repo-type dataset --local-dir ""
python -c "
import torch
a = torch.load('dataset_a.pt',  weights_only=False)
b = torch.load('dataset_b.pt',  weights_only=False)
torch.save(a + b, 'dataset.pt')
print(f'Merged {len(a)} + {len(b)} = {len(a+b)} samples -> dataset.pt')
"

rm dataset_a.pt dataset_b.pt