"""
    Various methods are kept here to keep the other code files simple
"""
import torch
from models import models_vM6 as models

def pick_model(args, dicts):
    """
        Use args to initialize the appropriate model
    """
    
    # Preprocessing
    kernel_sizes = [size for size in args.kernel_sizes if size.isnumeric()] # Removing commas if multiple filter sizes passed
    print(kernel_sizes)
    
    if args.loss_weights:
        loss_weights = str(args.loss_weights).split(",")
    else:
        loss_weights = None
        
    print(loss_weights)
    
    if args.model == "conv_encoder":
                
        model = models.ConvEncoder(args.embed_file, kernel_sizes, args.num_filter_maps, args.gpu, dicts, args.embed_size, args.dropout, args.conv_activation, loss_weights)
    
#    elif args.model == "rnn":
#        model = models.VanillaRNN(args.embed_file, dicts, args.rnn_dim, args.cell_type, args.rnn_layers, args.gpu, args.embed_size,
#                                  args.bidirectional)
#    elif args.model == "conv_attn":
#        model = models.ConvAttnPool(Y, args.embed_file, filter_size, args.num_filter_maps, args.lmbda, args.gpu, dicts,
#                                    embed_size=args.embed_size, dropout=args.dropout)
#    elif args.model == "saved":
#        model = torch.load(args.test_model)
    
    print(args.gpu)

    if args.gpu:
        model.cuda()
        
    return model

def make_param_dict(args):
    """
        Make a list of parameters to save for future reference
    """
    param_vals = [args.kernel_sizes, args.dropout, args.num_filter_maps,
        args.command, args.weight_decay, args.data_path,
        args.embed_file, args.lr]
    
    param_names = ["kernel_sizes", "dropout", "num_filter_maps", "command",
        "weight_decay", "data_path", "embed_file", "lr"]
    params = {name:val for name, val in zip(param_names, param_vals) if val is not None}
    return params
