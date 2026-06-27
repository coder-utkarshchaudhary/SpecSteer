import torch

model_data = torch.load('model/checkpoints/best_model.pt', map_location=torch.device('cpu'))

total_params = sum(tensor.numel() for tensor in model_data.values())
print(f"Total parameters: {total_params:,}")