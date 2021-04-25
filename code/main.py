import torch.optim as optim
from utils import *
from args import parse_args
from evaluator import Evaluator
from HeteroEventNet import HeteroEventNet
import time
import warnings
warnings.filterwarnings("ignore")


class Model():

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if args.mode != 'tuple':
            self.create_or_load_model() # get self.model

            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=self.args.learning_rate, betas=(0.7, 0.99))

        self.file_prefix = 'train'


    def create_or_load_model(self):
        g = construct_hetero_dglgraph(data_path=self.args.data_path,
                                         file_prefix='train',
                                         emb_path=self.args.emb_path,
                                         load_version=self.args.load_emb,
                                         homo=self.args.homogenous,
                                         datainput=self.args.data_input)
        valg = construct_hetero_dglgraph(data_path=self.args.data_path,
                                         file_prefix='dev',
                                         emb_path=self.args.emb_path,
                                         load_version=self.args.load_emb,
                                         homo=self.args.homogenous,
                                         datainput=self.args.data_input)

        self.model = HeteroEventNet(g,valg,
                             nfeat=self.args.input_dim,
                             nemb=self.args.emb_dim,
                             nhid=self.args.hidden_dim,
                             pooling=self.args.pooling,
                             max_batch=self.args.max_batch_for_train,
                             dropout=self.args.dropout,
                             views=self.args.views,
                             batch_size=self.args.batch_size,
                             model_base=self.args.model_base
                             ).to(self.device)


        model_file = os.path.join(self.args.model_path, self.args.load_model)
        if os.access(model_file, os.F_OK):

            # print("Model's state_dict:")
            # for param_tensor in self.model.state_dict():
            #   print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())


            if torch.cuda.is_available():
                self.model.load_state_dict(torch.load(model_file))
                self.model.cuda()
            else:
                self.model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))

            print('Model %s loaded\n' % model_file)

        else:

            print('Model %s created\n' % os.path.join(self.args.model_path,
                                                       '{}.{}.model'.format(self.args.model_base,self.args.version)))

            print_param_num = True
            if print_param_num:
                # print('Parameters:', end='\n\t')
                # for p in self.model.parameters():
                #     if p.requires_grad:
                #         print(p.name, p.numel())
                print(self.get_parameter_number(self.model))
                print()

    def get_parameter_number(self, net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def save_model(self):
        model_file = os.path.join(self.args.model_path, '{}.{}.model'.format(self.args.model_base,self.args.version))
        torch.save(self.model.state_dict(), model_file)

        # print("Model's state_dict:")
        # for param_tensor in self.model.state_dict():
        #     print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())

        # model_dict = self.model.state_dict()
        # for no_save in ['encoder.wordid2idf.weight', 'encoder.graph_pos.weight', 'encoder.graph_wgt.weight']:
        #     del model_dict[no_save]
        # torch.save(model_dict, model_file)
        print('Model %s saved' % model_file)



    def train(self):
        self.model.train()
        best_val_loss = float('inf')
        patience = 0  # early stop: if not beat best val loss for a while (5 epoch), stop training

        g = construct_hetero_dglgraph(data_path=self.args.data_path,
                                         file_prefix='train',
                                         emb_path=self.args.emb_path,
                                         load_version=self.args.load_emb,
                                         datainput=self.args.data_input)

        homo_g = construct_homo_from_hetero_dglgraph(g)

        val_g = construct_hetero_dglgraph(data_path=self.args.data_path,
                                         file_prefix='dev',
                                         emb_path=self.args.emb_path,
                                         load_version=self.args.load_emb,
                                         datainput=self.args.data_input)
        homo_val_g = construct_homo_from_hetero_dglgraph(val_g)
        val_metapath_dict = get_metapath_dict(val_g.etypes)

        for epoch in range(self.args.epochs):

            start = time.time()
            epoch_loss = 0 # train loss


            for batch_num in range(self.args.max_batch_for_train):

                self.model.train()
                self.optimizer.zero_grad()

                sem,dgi,skg,arc = self.model(g,homo_g, batch_num=batch_num)
                loss = self.get_loss(sem,dgi,skg,arc)


                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                self.optimizer.step()
                epoch_loss += loss.data




                if batch_num % self.args.print_every == 0:

                    end = time.time()
                    print('Epoch %d: cur_loss=%4.3f, [sem,dgi,skg,arc]=[%4.3f,%4.3f,%4.3f,%4.3f], time=%4.3f per %d batches' \
                          % (epoch, loss.data,
                             sem.data,
                             dgi.data,
                             skg.data,
                             arc.data,
                             (end - start), self.args.print_every))
                    start = time.time()


            train_loss = epoch_loss
            val_loss = self.eval(val_g, homo_val_g, val_metapath_dict)
            print("Epoch %d done\ttraining loss=%4.3f\tvalidation loss=%4.3f"%(epoch,train_loss,val_loss))

            if epoch < 5:
                self.save_model()
            else:
                if val_loss < best_val_loss:
                    self.save_model()
                    patience = 0
                    best_val_loss = val_loss

                else:
                    patience += 1

            if patience > 3: # patience value for early stopping
                print('Training Early Stop')
                break

        print('Training done')


    def get_loss(self, sem, dgi, skg, arc):
        # loss  for different lambdas

        loss = self.args.lambda_sem * sem + self.args.lambda_dgi * dgi + self.args.lambda_skg * skg + self.args.lambda_arc * arc
        return loss


    def eval(self, g, homo_g, metapath_dict):
        self.model.eval()
        eval_loss = 0
        for batch_num in range(self.args.max_batch_for_eval):
            sem, dgi, skg, arc = self.model(g, homo_g, metapath_dict=metapath_dict, batch_num=batch_num)
            loss = self.get_loss(sem, dgi, skg, arc)
            eval_loss += loss.data
        self.model.train()
        return eval_loss

    def infer(self):

        self.model.eval()

        for prefix in ['train','dev','test']:
            st = time.time()

            g = construct_hetero_dglgraph(data_path=self.args.data_path,
                                   file_prefix=prefix,
                                   emb_path=self.args.emb_path,
                                   load_version=self.args.load_emb,
                                   homo=self.args.homogenous,
                                   datainput=self.args.data_input)
            metapath_dict = get_metapath_dict(g.etypes)
            emb = self.model.encode(g,metapath_dict=metapath_dict)  # {ntype: [N,768]}

            # assert emb['event'].shape[0] == vocab_size_dict[prefix]['event']
            # assert emb['entity'].shape[0] == vocab_size_dict[prefix]['entity']
            print('Inference on %s set done: %4.3f s'%(prefix, time.time()-st))

            event_file = os.path.join(self.args.emb_path, '{}.{}.{}.eventemb.npy'.format(self.args.model_base,self.args.version,prefix))
            entity_file = os.path.join(self.args.emb_path,'{}.{}.{}.entityemb.npy'.format(self.args.model_base, self.args.version,prefix))
            np.save(event_file, emb['event'].detach().cpu().numpy())
            np.save(entity_file, emb['entity'].detach().cpu().numpy())
            print('\tSaved emb files to %s and %s'%(event_file,entity_file))

    def infer_tuple_baseline(self):

        for file_prefix in ['dev','test','train']:

            labels = json.load(open(os.path.join(self.args.data_path, file_prefix + '.label.json'), 'r'))
            event_file = os.path.join(self.args.data_path, file_prefix + '.eventemb.npy')
            entity_file = os.path.join(self.args.data_path, file_prefix + '.entityemb.npy')
            event_emb = np.load(event_file)
            entity_emb = np.load(entity_file)

            eve2ent = dict()
            for t in labels['event2entity']:
                eve2ent.setdefault(t[0],[])
                eve2ent[t[0]].append(t[1])


            cnt_event = event_emb.shape[0]

            for evidx in range(cnt_event):
                if evidx in eve2ent:
                    X = np.expand_dims(event_emb[evidx], axis=0)
                    attr = entity_emb[np.array(eve2ent[evidx])]
                    X = np.concatenate((X, attr), axis=0)
                    event_emb[evidx] = np.mean(X, axis=0)

            event_file = os.path.join(self.args.emb_path, 'Tuple.0.{}.eventemb.npy'.format(file_prefix))
            np.save(event_file,event_emb)
            entity_file = os.path.join(self.args.emb_path, 'Tuple.0.{}.entityemb.npy'.format(file_prefix))
            np.save(entity_file,entity_emb)






if __name__ == '__main__':

    args = parse_args()

    # log - redirect print
    log_file = os.path.join(args.log_path, '{}.{}.{}.log'.format(args.mode,args.model_base, args.version))
    redirect_stdout(open(log_file, 'w'))
    argsDict = args.__dict__
    for arg in argsDict.keys():
      print('\t'+str(arg)+ '\t=\t'+ str(argsDict[arg]))

    if args.mode == 'train':
        m = Model(args=args)
        m.train()
        m.infer()

    elif args.mode == 'infer':
        m = Model(args=args)
        m.infer()

    elif args.mode == 'eval':
        res = dict()
        if args.task == None:
            tasks = ['node', 'event2entity']
        else:
            tasks = args.task.split()
        for task in tasks:
            args.version += '.'+task
            e = Evaluator(task=task,
                          data_path=args.data_path,
                          emb_path=args.emb_path,
                          load_emb=args.load_emb,
                          emb_dim=args.emb_dim,
                          batch_size=args.batch_size,
                          epochs=args.epochs)
            res[task] = e.eval()

        # for std_split evaluator
        print('========FINAL REPORT (std)========')
        for task in tasks:
            print()
            for name, val in res[task].items():
                if type(val) == str:
                    print('%s=%s' % (name, val))
                else:
                    print('%s=%.4f' % (name, val))
        print('========END========')

