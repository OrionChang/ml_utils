import torch


def use_cuda():
    """
    Set the default device to CUDA if available, otherwise set it to CPU.
    """
    # Check if CUDA is available
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    torch.set_default_device(device)
    print(f"Using device = {torch.get_default_device()}")
