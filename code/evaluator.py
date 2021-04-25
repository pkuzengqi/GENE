
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn import preprocessing

import torch.optim as optim
from allennlp.training.metrics import Average


from Prober import *
from utils import *
from torch.utils.data import Dataset, DataLoader

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)




class Evaluator():

    def __init__(self, task='node', data_path='./data/ace/ace_graph/', emb_path='./emb/', load_emb='', emb_dim=256, batch_size=64, epochs = 30, input_dim=256):


        if task != 'node':
            emb_dim *= 2
            input_dim *= 2

        self.task = task
        self.data_path = data_path
        self.emb_path = emb_path
        self.emb_dim = emb_dim
        self.input_dim = input_dim
        self.load_emb = load_emb
        self.batch_size = batch_size
        self.epochs = epochs

    def eval(self):
        '''
        task: 'node' -- node typing
        task: 'event2entity'  -- argument role labeling
        '''

        X_train, Y_train = self.get_X_Y_from_dgl_graph(prefixes='train')
        X_val, Y_val = self.get_X_Y_from_dgl_graph(prefixes='dev')
        X_test, Y_test = self.get_X_Y_from_dgl_graph(prefixes='test')

        micro, macro = self.get_f1_scores(X_train, Y_train, X_val, Y_val, X_test, Y_test)
        results={'task':self.task, 'micro':micro,'macro':macro, 'train_shape':X_train.shape[0], 'test_shape':X_test.shape[0]}
        # print(results)

        return results

    def split(self, X,Y, train_ratio):
        X = preprocessing.normalize(X, norm='l2')
        X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=train_ratio, random_state=42)
        return X_train, X_test, y_train, y_test

    def split_docid(self, docid2event, train_ratio):
        all_docid = docid2event.keys()
        train_docid = np.random.choice(np.array(list(all_docid)), int(train_ratio * len(all_docid)), replace=False)
        test_docid = [i for i in all_docid if i not in train_docid]
        return set(train_docid), set(test_docid)

    def get_X_Y_with_docid(self, event2emb, docids, event2feat=None):

        # step 2: read coref_pred.csv, find those need train or test, put into
        X, Y, linenum = [], [], []
        lines = open(os.path.join(self.data_path, 'coref_candidate.csv'),'r').readlines()

        if event2feat == None:
            for i,line in enumerate(lines):
                l = line.split('\t')
                event1, event2, train_required, label = l[0], l[1], int(l[2]), int(l[3])
                if train_required and event1 in event2emb and event2 in event2emb:
                    emb = np.concatenate((event2emb[event1],event2emb[event2]))
                    if event1.split('-EV')[0] in docids:
                        X.append(emb)
                        Y.append(label)
                        linenum.append(i) #Y[i] corresponds to line[i], which is the linenum in coref_candidate
        else:
            for i,line in enumerate(lines):
                l = line.split('\t')
                event1, event2, train_required, label = l[0], l[1], int(l[2]), int(l[3])
                if train_required and event1 in event2emb and event2 in event2emb and event1 in event2feat and event2 in event2feat:
                    emb = np.concatenate((event2emb[event1], event2feat[event1], event2emb[event2], event2feat[event2]))
                    if event1.split('-EV')[0] in docids:
                        X.append(emb)
                        Y.append(label)
                        linenum.append(i) #Y[i] corresponds to line[i], which is the linenum in coref_candidate
        # test_linenum means which lines in coref.candidate which be replaced by y_pred from y_test
        return np.array(X), np.array(Y), linenum

    def get_event2emb(self):
        '''
        :return: {eventid:[emb]}
        '''
        event2emb = dict()
        for file_prefix in ['train', 'dev', 'test']:
            # get emb list
            event_file = os.path.join(self.emb_path, self.load_emb + '.' + file_prefix + '.eventemb.npy')
            if not os.access(event_file, os.F_OK):
                event_file = os.path.join(self.data_path, file_prefix + '.eventemb.npy')
            event_emb = np.load(event_file)

            # get nid2eventid
            eventid2nid = json.load(open(os.path.join(self.data_path, file_prefix + '.event2nid.json'), 'r'))
            for eventid, nid in eventid2nid.items():
                event2emb[eventid] = event_emb[int(nid)]

        return event2emb

    def get_event2feat(self):
        '''
        :return: {eventid:[emb]}
        another set of trained embeddings
        '''
        e = json.load(open(os.path.join(self.data_path, 'coref_feature.json'),'r'))
        event2feat = dict()
        for eventid, emblist in e.items():
            event2feat[eventid] = np.array(emblist)
        return event2feat


    def get_docid2event(self, prefixes = 'train dev'):
        '''
        :return: {docid:[eventid]}
        '''
        docid2event = dict()
        event2docid = dict()
        for file_prefix in prefixes.split():
            # get nid2eventid
            nid2event = dict()
            eventid2nid = json.load(open(os.path.join(self.data_path, file_prefix + '.event2nid.json'), 'r'))
            for eventid, nid in eventid2nid.items():
                nid2event[nid] = eventid

            labels = json.load(open(os.path.join(self.data_path, file_prefix + '.label.json'), 'r'))
            for nid,e in enumerate(labels['event']):
                # event/entity: [typeid, doc_id, token_start, token_end]
                docid = e[1]
                eventid = nid2event[nid]
                docid2event.setdefault(docid,[])
                docid2event[docid].append(eventid)
                event2docid[eventid] = docid
        return docid2event

    def get_f1_scores(self, X_train, y_train, X_val, y_val, X_test, y_test):


        ### MLP classifier
        clf = Prober_Trainer(task=self.task, emb_dim=self.emb_dim, batch_size=self.batch_size, epochs=self.epochs, input_dim=self.input_dim)
        clf.train(X_train, y_train, x_val=X_val, y_val=y_val)
        _, y_pred = clf.infer(X_test, y_test)


        micro = f1_score(y_test, y_pred, average="micro")
        macro = f1_score(y_test, y_pred, average="macro")
        print('========EVALUATION========')
        print("task:%s\nmicro_f1: %.4f\nmacro_f1: %.4f\n"%(self.task,micro,macro))
        print('========END========')
        return micro, macro


    def get_X_Y_from_dgl_graph(self, prefixes='train dev'):

        X = np.empty((0, self.input_dim))
        Y = np.empty((0))

        for file_prefix in prefixes.split():
            g = construct_dglgraph(self.data_path, file_prefix, emb_path=self.emb_path, load_version=self.load_emb, homo=False)

            if self.task == 'node':
                ##### read data #####
                datalen = g.number_of_nodes('event') + g.number_of_nodes('entity')
                attr = np.concatenate((g.nodes['event'].data['x'], g.nodes['entity'].data['x']), axis=0)
                label = np.concatenate((g.nodes['event'].data['y'], g.nodes['entity'].data['y'] + type_class_dict['event']),
                                            axis=0)
                ##### get pairs #####
                X = np.concatenate((X,attr),axis=0)
                Y = np.concatenate((Y,label),axis=0)

            elif self.task == 'event2entity':
                ##### read data #####
                entitycnt = g.number_of_nodes('entity')
                eventcnt = g.number_of_nodes('event')
                edgecnt = g.number_of_edges('argrole')
                attrfrom = g.nodes['event'].data['x']
                attrto = g.nodes['entity'].data['x']
                edgefrom, edgeto = g.all_edges(form='uv', etype='argrole', order='eid')
                label = g.edges['argrole'].data['y']
                adj = g.adjacency_matrix(etype='argrole', transpose=True).to_dense()  # use the 01 martix to check whether relation exists
                ##### get pairs #####
                # true examples from graph
                attr = np.empty((edgecnt,self.input_dim))
                for i in range(edgecnt):
                    attr[i] = np.concatenate((attrfrom[edgefrom[i]],attrto[edgeto[i]]),axis=0)
                X = np.concatenate((X,attr),axis=0)
                Y = np.concatenate((Y,label),axis=0)

                # add negative examples
                add_negative = False
                if add_negative:
                    attr = np.empty((edgecnt, self.emb_dim))
                    for i in range(edgecnt):
                        idx1 = np.random.randint(eventcnt)
                        idx2 = np.random.randint(entitycnt)
                        while adj[idx1][idx2]:
                            idx1 = np.random.randint(eventcnt)
                            idx2 = np.random.randint(entitycnt)
                        attr[i] = np.concatenate((attrfrom[idx1], attrto[idx2]), axis=0)
                    X = np.concatenate((X,attr),axis=0)
                    Y = np.concatenate((Y,np.zeros_like(label)),axis=0)

            elif self.task == 'temp':
                # event ordering
                ##### read data #####
                eventcnt = g.number_of_nodes('event')
                edgecnt = g.number_of_edges('temp')
                attrfrom = g.nodes['event'].data['x']
                edgefrom, edgeto = g.all_edges(form='uv', etype='temp', order='eid')
                label = g.edges['temp'].data['y']
                adj = g.adjacency_matrix(etype='temp', transpose=True).to_dense()  # use the 01 martix to check whether relation exists
                ##### get pairs #####
                # true examples from graph
                attr = np.empty((edgecnt,self.emb_dim))
                for i in range(edgecnt):
                    attr[i] = np.concatenate((attrfrom[edgefrom[i]],attrfrom[edgeto[i]]),axis=0)
                X = np.concatenate((X,attr),axis=0)
                Y = np.concatenate((Y,label),axis=0)

                # add negative examples
                add_negative = False
                if add_negative:
                    attr = np.empty((edgecnt, self.emb_dim))
                    for i in range(edgecnt):
                        idx1 = np.random.randint(eventcnt)
                        idx2 = np.random.randint(eventcnt)
                        while adj[idx1][idx2]:
                            idx1 = np.random.randint(eventcnt)
                            idx2 = np.random.randint(eventcnt)
                        attr[i] = np.concatenate((attrfrom[idx1], attrfrom[idx2]), axis=0)
                    X = np.concatenate((X,attr),axis=0)
                    Y = np.concatenate((Y,np.zeros_like(label)),axis=0)

        return X, Y



class Prober_Trainer():

    def __init__(self, task='node', emb_dim=256, input_dim=768, batch_size=64, epochs = 30):
        super().__init__()
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nclass = type_class_dict[task]

        #mark
        self.model = ProjectedProber(nfeat=emb_dim,
                            nclass=self.nclass,
                            input_emb=input_dim,
                            dropout=0.1
                            ).to(self.device)

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=1e-4, betas=(0.7, 0.99))

        self.criterion = nn.CrossEntropyLoss()
        self.scorer = Average()


        self.mask_emb = None
        if task == 'event2entity':
            self.mask_emb = nn.Embedding.from_pretrained(torch.from_numpy(get_argroletyping_masks())).to(self.device)
            self.mask_emb.weight.requires_grad = False

    def train(self, x_train, y_train, x_val, y_val):

        train_iter = self.get_iterators_XY(x_train, y_train)
        val_iter = self.get_iterators_XY(x_val, y_val)
        best_val_loss = float('inf')
        patience = 0 # early stop: if not beat best val loss for a while (5 epoch), stop training

        for epoch in range(self.epochs):
            st = time.time()
            epoch_loss = 0 # train_loss
            batch_num = 0
            for attr, label in train_iter:

                batch_num += 1
                self.model.train()
                self.optimizer.zero_grad()

                loss= self.train_step(attr,label)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                epoch_loss += loss.data


            # eval for each epoch


            print('Epoch %d used in total %.4f s' % (epoch, time.time() - st))
            st = time.time()

            train_loss = epoch_loss / batch_num
            print("\tTraining loss=%4.8f, acc=%4.3f" % (train_loss, self.scorer.get_metric(reset=True)))
            val_loss = self.get_val_loss(val_iter)
            print("\tValidation loss=%4.8f, acc=%4.3f" % (train_loss, self.scorer.get_metric(reset=True)))


            if val_loss < best_val_loss:
                patience = 0
                best_val_loss = val_loss
            else:
                patience += 1
            if patience >= 5: # patience value for early stopping
                break # stop training
        print('Training done')


    def train_step(self, attr, label):

        # attr = [batch_size=64, emb_dim=768]
        # label = [batch_size=64]

        attr = attr.float()
        label = label.long() # the target of cross entropy loss should be long
        if torch.cuda.is_available():
            attr = attr.cuda()
            label = label.cuda()

        logits = self.model(attr)
        # logits = [batch_size=32, nclass=]

        if self.mask_emb is not None:
            mask = self.mask_emb(label)
            logits = logits.mul(mask)

        # cross entropy loss
        loss = self.criterion(logits, label)

        # get accuracy
        pred = torch.argmax(logits, 1)
        acc = (pred == label).sum().float() / self.batch_size
        self.scorer(acc)
        return loss

    def get_val_loss(self, val_iter):

        self.model.eval()

        epoch_loss = 0
        batch_num = 0

        for attr, label in val_iter:

            batch_num += 1

            loss = self.train_step(attr, label)
            epoch_loss += loss.data


        val_loss = epoch_loss / batch_num

        self.model.train()
        return val_loss


    def infer(self, x, y):
        # x = [N=470, emb_dim=768*2]
        # mask = [event_type=51, arg_role=238]
        infer_logits = []
        infer_pred = []
        self.model.eval()

        test_iter = self.get_iterators_XY(x, y, shuffle=False)

        for attr, label in test_iter:
            attr = attr.float()
            label = label.long()
            if torch.cuda.is_available():
                attr = attr.cuda()
                label = label.cuda()

            # logits = [batch_size=64, nclass=2]
            logits = self.model(attr)

            if self.mask_emb is not None:
                mask = self.mask_emb(label)
                logits = logits.mul(mask)

            logits = F.softmax(logits, dim=1)
            infer_logits.append(logits)
            pred = torch.argmax(logits, 1)
            infer_pred.append(pred)

        return torch.cat(infer_logits,0).detach().float().cpu().numpy(), torch.cat(infer_pred,0).detach().float().cpu().numpy()




    def get_iterators_XY(self, X,Y, shuffle=True):
        # used in evaluator
        st = time.time()
        data = self.XYDataset(X,Y)
        iter = DataLoader(data, batch_size=self.batch_size, shuffle=shuffle, num_workers=0, drop_last=False)
        print('get_iterators uses in total %.4f s' % (time.time() - st))

        return iter

    def get_iterators_X(self, X, shuffle=True):
        # used in evaluator
        st = time.time()
        data = self.XDataset(X)
        iter = DataLoader(data, batch_size=self.batch_size, shuffle=shuffle, num_workers=0, drop_last=False)
        print('get_iterators uses in total %.4f s' % (time.time() - st))

        return iter

    class XYDataset(Dataset):
        def __init__(self, X, Y):
            self.X = X
            self.Y = Y
        def __getitem__(self, i):
            return self.X[i], self.Y[i]
        def __len__(self):
            return self.X.shape[0]

    class XDataset(Dataset):
        def __init__(self, X):
            self.X = X
        def __getitem__(self, i):
            return self.X[i]
        def __len__(self):
            return self.X.shape[0]