import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard.writer import SummaryWriter

def get_tensorboard_writer(log_dir = None) -> SummaryWriter:
    """
    Create and return a TensorBoard SummaryWriter.
    
    Args:
        log_dir (str): Directory where TensorBoard logs will be written.
            Default is None, which uses TensorBoard's default 'runs/CURRENT_DATETIME_HOSTNAME'.
    
    Returns:
        SummaryWriter: TensorBoard writer object. If log_dir is provided, writes logs to 'runs/{log_dir}'.
    """
    full_log_dir = None

    if log_dir is not None:
        # Parent directory for all TensorBoard logs
        parent_dir = "runs/"  

        log_dir = log_dir.replace('\\', '/')
        # Strip leading slash if present
        if log_dir.startswith('/'):
            log_dir = log_dir[1:]
        
        # Combine parent directory with log directory
        full_log_dir = parent_dir + log_dir
        
    return SummaryWriter(full_log_dir)

def add_images_to_tensorboard(writer: SummaryWriter, images: torch.Tensor, title: str = 'Images'):
    """
    Add a batch of images to TensorBoard.
    
    Args:
        writer (SummaryWriter): TensorBoard writer object.
        images (torch.Tensor): Batch of images to add.
        title (str): Title for the images in TensorBoard.
        image_format (str): Format of the input images ('NCHW' or 'NHWC').
            Default is 'NCHW' (PyTorch default).
    """
    img_grid = torchvision.utils.make_grid(images)
    writer.add_image(title, img_grid)
    writer.flush()

def matplotlib_imshow(img: torch.Tensor, one_channel: bool = False):
    """
    Display an image using matplotlib.
    
    Args:
        img (torch.Tensor): Image to display.
        one_channel (bool): If True, image is expected to be grayscale.
            Default is False.
    """
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def log_model_graph(writer: SummaryWriter, model: torch.nn.Module, sample_input: torch.Tensor):
    """
    Add model graph to TensorBoard.
    
    Args:
        writer (SummaryWriter): TensorBoard writer object.
        model (torch.nn.Module): PyTorch model to visualize.
        sample_input (torch.Tensor): Sample input to trace through the model.
    """
    writer.add_graph(model, sample_input)
    writer.flush()

def log_scalars(writer: SummaryWriter, tag: str, scalar_dict: dict, global_step = None):
    """
    Log multiple scalars to TensorBoard under the same tag.
    
    Args:
        writer (SummaryWriter): TensorBoard writer object.
        tag (str): Data identifier.
        scalar_dict (dict): Dictionary of scalar names and values.
        global_step (int, optional): Global step value to record.
    """
    writer.add_scalars(tag, scalar_dict, global_step)
    writer.flush()

def log_scalar(writer: SummaryWriter, tag: str, scalar_value: float, global_step = None):
    """
    Log a single scalar to TensorBoard.
    
    Args:
        writer (SummaryWriter): TensorBoard writer object.
        tag (str): Data identifier.
        scalar_value (float): Value to log.
        global_step (int, optional): Global step value to record.
    """
    writer.add_scalar(tag, scalar_value, global_step)
    writer.flush()

def log_embedding(writer: SummaryWriter, features: torch.Tensor, metadata = None, label_img = None, global_step = None, tag = 'default'):
    """
    Add embedding projections to TensorBoard.
    
    Args:
        writer (SummaryWriter): TensorBoard writer object.
        features (torch.Tensor): Features matrix to be embedded and visualized.
        metadata (list, optional): Labels for each datapoint.
        label_img (torch.Tensor, optional): Images to display with the embedding.
        global_step (int, optional): Global step value to record.
        tag (str): Name for the embedding.
    """
    writer.add_embedding(features, metadata=metadata, label_img=label_img, 
                            global_step=global_step, tag=tag)
    writer.flush()

def log_histograms(writer: SummaryWriter, model: torch.nn.Module, step: int):
    """
    Log histograms of model parameters to TensorBoard.
    
    Args:
        writer (SummaryWriter): TensorBoard writer object.
        model (torch.nn.Module): Model containing parameters to log.
        step (int): Global step value to record.
    """
    for name, param in model.named_parameters():
        if param.requires_grad:
            writer.add_histogram(f"params/{name}", param.data, step)
            if param.grad is not None:
                writer.add_histogram(f"grads/{name}", param.grad, step)
    writer.flush()

def close_writer(writer: SummaryWriter):
    """
    Close the TensorBoard writer properly.
    
    Args:
        writer (SummaryWriter): TensorBoard writer object to close.
    """
    writer.flush()
    writer.close()
