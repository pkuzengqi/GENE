import argparse

OPTIMIZERS = ["Adam", "SGD"]
MODES = ['train', 'eval', 'infer','eval_ratio']
FILE_PREFIXS = ['train', 'dev', 'test']

def parse_args():
    """
    read command line argument
    :return: args
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--proj_path', type=str, default='./')
    parser.add_argument('--data_path', type=str, default='./data/ace/ace_graph/')
    parser.add_argument('--model_path', type=str, default='./checkpoint/', help="model saving directory")
    parser.add_argument('--emb_path', type=str, default='./emb/', help="")
    parser.add_argument('--log_path', type=str, default='./log/')

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--print_every', type=int, default=100, help='print the loss for every 1000 batches')
    parser.add_argument('--save_every', type=int, default=10, help='save the model for every 1 epoch(s)')
    parser.add_argument('--max_batch_for_train', type=int, default=10000)
    parser.add_argument('--max_batch_for_eval', type=int, default=100)
    parser.add_argument('--random_seed',type=int, default=42)

    parser.add_argument('--mode', choices=MODES, default='train')
    parser.add_argument('--version', type=str, default='1000.0')
    parser.add_argument('--model_base', type=str, default='SEM_ARC')
    parser.add_argument('--load_model', type=str, default='none.model', help='format model_base.0.0.0.model')
    parser.add_argument('--load_emb', type=str, default='', help='format DGI.0.0, if not found use default')

    parser.add_argument('--data_input', type=str, default='narr', help='narr coref eer temp')
    parser.add_argument('--views',type=str, default=None, help='event2entity entity2entity event2event all')
    parser.add_argument('--pooling',type=str, default='avg', help='cat avg weighted no')


    # param in trainer
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--optimizer', choices=OPTIMIZERS, default='Adam')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--clip', type=float, default=0.5, help='clip gradient')

    # param in GAT
    parser.add_argument('--input_dim', type=int, default=768, help='input bert word emb dim')
    parser.add_argument('--emb_dim', type=int, default=256, help='output event/entity node emb dim')
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden size in models')
    parser.add_argument('--num_heads', type=int, default=2, help='number of attention heads')
    parser.add_argument('--num_layers', type=int, default=2, help='number of propagation layers')

    # param in Prober
    parser.add_argument('--task', type=str, default=None)

    # param in sampler
    parser.add_argument('--num_seeds', type=int, default=1)
    parser.add_argument('--num_hops', type=int, default=2)

    # lambda for four losses
    parser.add_argument('--lambda_sem',type=float, default=1.0, help='lambda of semantic reconstruction loss')
    parser.add_argument('--lambda_dgi', type=float, default=1.0, help='lambda of graph infomax loss')
    parser.add_argument('--lambda_skg',type=float, default=1.0, help='lambda of skipgram loss')
    parser.add_argument('--lambda_arc',type=float, default=1.0, help='lambda of adversarial relation classifier loss')


    args = parser.parse_args()

    assert 0 <= args.dropout <= 1

    return args

