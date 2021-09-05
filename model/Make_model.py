from vector_fields import *
from GCDE import *

def make_model(args):
    if args.model_type == 'type1':
        vector_field_f = FinalTanh_f(input_channels=args.input_dim, hidden_channels=args.hid_dim,
                                        hidden_hidden_channels=args.hid_hid_dim,
                                        num_hidden_layers=args.num_layers)
        vector_field_g = VectorField_g(input_channels=args.input_dim, hidden_channels=args.hid_dim,
                                        hidden_hidden_channels=args.hid_hid_dim,
                                        num_hidden_layers=args.num_layers, num_nodes=args.num_nodes, cheb_k=args.cheb_k, embed_dim=args.embed_dim,
                                        g_type=args.g_type)
        model = NeuralGCDE(args, func_f=vector_field_f, func_g=vector_field_g, input_channels=args.input_dim, hidden_channels=args.hid_dim,
                                        output_channels=args.output_dim, initial=True,
                                        device=args.device, atol=1e-9, rtol=1e-7, solver=args.solver)
        return model, vector_field_f, vector_field_g