import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets
import torch.nn as nn
from scipy.spatial.distance import cdist
import numpy as np
from data_loader import mnist, svhn, usps, office31
from torch.autograd import grad
from itertools import chain
import random


def digit_load(args): 
    train_bs = args.batch_size
    if args.trans == 's2m':
        train_source = svhn.SVHN(args.dataset_root + '/svhn/', split='train', download=True,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]))
        test_source = svhn.SVHN(args.dataset_root + '/svhn/', split='test', download=True,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]))  
        train_target = mnist.MNIST_idx(args.dataset_root + '/mnist/', train=True, download=True,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]))      
        test_target = mnist.MNIST(args.dataset_root + '/mnist/', train=False, download=True,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]))
    elif args.trans == 'u2m':
        train_source = usps.USPS(args.dataset_root + '/usps/', train=True, download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(28, padding=4),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))
        test_source = usps.USPS(args.dataset_root + '/usps/', train=False, download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(28, padding=4),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))    
        train_target = mnist.MNIST_idx(args.dataset_root + '/mnist/', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))    
        test_target = mnist.MNIST(args.dataset_root + '/mnist/', train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))
    elif args.trans == 'm2u':
        train_source = mnist.MNIST(args.dataset_root + '/mnist/', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))
        test_source = mnist.MNIST(args.dataset_root + '/mnist/', train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))

        train_target = usps.USPS_idx(args.dataset_root + '/usps/', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))
        test_target = usps.USPS(args.dataset_root + '/usps/', train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))
    dset_loaders = {}
    dset_loaders["source_train"] = DataLoader(train_source, batch_size=train_bs, shuffle=True, 
        num_workers=args.num_workers, drop_last=False)
    dset_loaders["source_test"] = DataLoader(test_source, batch_size=train_bs*2, shuffle=True, 
        num_workers=args.num_workers, drop_last=False)
    dset_loaders["target_train"] = DataLoader(train_target, batch_size=train_bs, shuffle=True, 
        num_workers=args.num_workers, drop_last=False)
    dset_loaders["target_train_no_shuff"] = DataLoader(train_target, batch_size=train_bs, shuffle=False, 
        num_workers=args.num_workers, drop_last=False)
    dset_loaders["target_test"] = DataLoader(test_target, batch_size=train_bs*2, shuffle=False, 
        num_workers=args.num_workers, drop_last=False)
        
    return dset_loaders

def office31_load(args):
    train_bs = args.batch_size
    source = args.trans.split("2")[0]
    target = args.trans.split("2")[1]

    dset_loaders = {}
    dset_loaders["source_train"] = office31.get_office_dataloader(source,train_bs,True)
    dset_loaders["source_test"] = office31.get_office_dataloader(source,train_bs,True)
    dset_loaders["target_train"] = office31.get_office_dataloader(target,train_bs,True)
    dset_loaders["target_train_no_shuff"] = office31.get_office_dataloader(target,train_bs,False)
    dset_loaders["target_test"] = office31.get_office_dataloader(target,train_bs,True)
    
    return dset_loaders

def init_weights_orthogonal(m):
    if type(m) == nn.Conv2d:
        nn.init.orthogonal_(m.weight)
    if type(m) == nn.Linear:
        nn.init.orthogonal_(m.weight)

def init_weights_xavier_normal(m):
    if type(m) == nn.Conv2d:
        nn.init.xavier_normal(m.weight)
    if type(m) == nn.Linear:
        nn.init.xavier_normal(m.weight)
        
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)



# def discrepancy(out1, out2):
#     p = F.softmax(out1, dim=-1)
#     _kl = torch.sum(p * (F.log_softmax(out1, dim=-1)
#                                   - F.log_softmax(out2, dim=-1)), 1)
#     return torch.mean(_kl)

def discrepancy_matrix(out1, out2):
    out1 = F.softmax(out1,dim=1)
    out2 = F.softmax(out2,dim=1)
    mul = out1.transpose(0, 1).mm(out2)
    loss_dis = torch.sum(mul) - torch.trace(mul)
    return loss_dis


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, size_average=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.size_average = size_average
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        if self.size_average:
            loss = (- targets * log_probs).mean(0).sum()
        else:
            loss = (- targets * log_probs).sum(1)
        return loss
#     单分类器的伪标签
def obtain_label_one(loader, netE, netC1, args, c=None):
    start_test = True
    netE.eval()
    netC1.eval()
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0] #(32,3,224,224)
            labels = data[1] #(32)
            indexs = data[2]
            inputs = inputs.cuda()
            feas = netE(inputs)
            outputs1 = netC1(feas)
            outputs = outputs1
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)  #(600)
    all_output = nn.Softmax(dim=1)(all_output) #(600,12)
    _, predict = torch.max(all_output, 1) #(600)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    all_fea = all_fea.float().cpu().numpy()

    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    dd = cdist(all_fea, initc, 'cosine')
    pred_label = dd.argmin(axis=1) #(600)
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)


    log_str = 'Only source accuracy = {:.2f}% -> After the clustering = {:.2f}%'.format(accuracy*100, acc*100)
    print(log_str+'\n')
    return pred_label.astype('int')
# pseudo labels                                      
def obtain_label(loader, netE, netC1, netC2, args, c=None):
    start_test = True
    netE.eval()
    netC1.eval()
    netC2.eval()
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0] #(32,3,224,224)
            labels = data[1] #(32)
            indexs = data[2]
            inputs = inputs.cuda()
            feas = netE(inputs)
            # outputs1, _, _ = netC1(feas)
            # outputs2, _, _ = netC2(feas)
            outputs1 = netC1(feas)
            outputs2 = netC2(feas)
            outputs = outputs1 + outputs2 
            #torch.stack([outputs1,outputs2]).mean(dim=0)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)  #(600)
    ###沿列序softmax，
    all_output = nn.Softmax(dim=1)(all_output) #(600,12)
    ###索引每行的最大值，predic是索引号
    _, predict = torch.max(all_output, 1) #(600)
    #print("all_label:",all_label.size()[0],"right:",torch.squeeze(predict).float().eq(all_label.data).sum().item())
    ###squeeze是对矩阵维度进行压缩
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    ###按列序进行拼接
    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    ###.t()是对张量进行转置，.norm是求范数的值
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    ###将float转化为numpy
    all_fea = all_fea.float().cpu().numpy()

    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    ###这里即（9）式
    initc = aff.transpose().dot(all_fea) #对应位置相乘再相加
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    dd = cdist(all_fea, initc, 'cosine')
    ###argmin即返回每一行最小值的索引号
    pred_label = dd.argmin(axis=1) #(600)
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    for round in range(1):
        ###转化为one-hot矩阵
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)


    log_str = 'Only source accuracy = {:.2f}% -> After the clustering = {:.2f}%'.format(accuracy*100, acc*100)
    print(log_str+'\n')
    return pred_label.astype('int')

def obtain_label1(loader, netE, netC1, netC2, args, c=None):
    start_test = True
    netE.eval()
    netC1.eval()
    netC2.eval()
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0] #(32,3,224,224)
            labels = data[1] #(32)
            indexs = data[2]
            inputs = inputs.cuda()
            feas = netE(inputs)
            outputs1, _, _ = netC1(feas)
            outputs2, _, _ = netC2(feas)
            outputs = outputs1 + outputs2 
            #torch.stack([outputs1,outputs2]).mean(dim=0)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)  #(600)
    ###沿列序softmax，
    all_output = nn.Softmax(dim=1)(all_output) #(600,12)
    ###索引每行的最大值，predic是索引号
    _, predict = torch.max(all_output, 1) #(600)
    #print("all_label:",all_label.size()[0],"right:",torch.squeeze(predict).float().eq(all_label.data).sum().item())
    ###squeeze是对矩阵维度进行压缩
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    ###按列序进行拼接
    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    ###.t()是对张量进行转置，.norm是求范数的值
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    ###将float转化为numpy
    all_fea = all_fea.float().cpu().numpy()

    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    ###这里即（9）式
    initc = aff.transpose().dot(all_fea) #对应位置相乘再相加
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    dd = cdist(all_fea, initc, 'cosine')
    ###argmin即返回每一行最小值的索引号
    pred_label = dd.argmin(axis=1) #(600)
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    for round in range(1):
        ###转化为one-hot矩阵
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)


    log_str = 'Only source accuracy = {:.2f}% -> After the clustering = {:.2f}%'.format(accuracy*100, acc*100)
    print(log_str+'\n')
    return pred_label.astype('int')


def gradient_discrepancy_loss(args, preds_s1,preds_s2, src_y, preds_t1, preds_t2, tgt_y, netE, netC1, netC2):
    loss_w = Weighted_CrossEntropy
    loss = nn.CrossEntropyLoss()
    #CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=0.1)
    total_loss = 0
    c_candidate = list(range(args.class_num))
    random.shuffle(c_candidate)
    # gmn iterations
    for c in c_candidate[0:args.gmn_N]:
        # gm loss
        gm_loss = 0

        src_ind = (src_y == c).nonzero().squeeze()
        #print("src_y,",src_y,"src_ind:",src_ind)
        tgt_ind = (tgt_y == c).nonzero().squeeze()
        if src_ind.shape == torch.Size([]) or tgt_ind.shape == torch.Size([]) or src_ind.shape == torch.Size([0]) or tgt_ind.shape == torch.Size([0]):
            continue

        p_s1 = preds_s1[src_ind]
        p_s2 = preds_s2[src_ind]
        p_t1 = preds_t1[tgt_ind]
        p_t2 = preds_t2[tgt_ind]
        s_y = src_y[src_ind]
        t_y = tgt_y[tgt_ind]
        
        #print("src_ind:",s_y,"tgt_ind:",t_y)

        src_loss1 = loss(p_s1 , s_y)
        
        tgt_loss1 = loss_w(p_t1 , t_y)

        src_loss2 = loss(p_s2 , s_y)
        tgt_loss2 = loss_w(p_t2 , t_y)

        grad_cossim11 = []
        #grad_mse11 = []
        grad_cossim22 = []
        #grad_mse22 = []

        #netE+C1
        for n, p in netC1.named_parameters():
            # if len(p.shape) == 1: continue

            real_grad = grad([src_loss1],
                                [p],
                                create_graph=True,
                                only_inputs=True,
                                allow_unused=False)[0]
            fake_grad = grad([tgt_loss1],
                                [p],
                                create_graph=True,
                                only_inputs=True,
                                allow_unused=False)[0]

            if len(p.shape) > 1:
                _cossim = F.cosine_similarity(fake_grad, real_grad, dim=1).mean()
            else:
                _cossim = F.cosine_similarity(fake_grad, real_grad, dim=0)
            #_mse = F.mse_loss(fake_grad, real_grad)
            grad_cossim11.append(_cossim)
            #grad_mse.append(_mse)

        grad_cossim1 = torch.stack(grad_cossim11)
        gm_loss1 = (1.0 - grad_cossim1).sum()
        #grad_mse1 = torch.stack(grad_mse)
        #gm_loss1 = (1.0 - grad_cossim1).sum() * args.Q + grad_mse1.sum() * args.Z
        #netE+C2
        for n, p in netC2.named_parameters():
            # if len(p.shape) == 1: continue

            real_grad = grad([src_loss2],
                                [p],
                                create_graph=True,
                                only_inputs=True)[0]
            fake_grad = grad([tgt_loss2],
                                [p],
                                create_graph=True,
                                only_inputs=True)[0]

            if len(p.shape) > 1:
                _cossim = F.cosine_similarity(fake_grad, real_grad, dim=1).mean()
            else:
                _cossim = F.cosine_similarity(fake_grad, real_grad, dim=0)
            #_mse = F.mse_loss(fake_grad, real_grad)
            grad_cossim22.append(_cossim)
            #grad_mse.append(_mse)

        grad_cossim2 = torch.stack(grad_cossim22)
        #grad_mse2 = torch.stack(grad_mse)
        #gm_loss2 = (1.0 - grad_cossim2).sum() * args.Q + grad_mse2.sum() * args.Z
        gm_loss2 = (1.0 - grad_cossim2).sum()
        gm_loss = (gm_loss1 + gm_loss2)/2.0
        total_loss += gm_loss
        
    return total_loss/args.gmn_N

def gradient_discrepancy_loss_margin(args, p_s1,p_s2, s_y, p_t1, p_t2, t_y, netE, netC1, netC2):
    loss_w = Weighted_CrossEntropy
    loss = nn.CrossEntropyLoss()
    #CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=0.1)
    # gm loss
    gm_loss = 0

    #print("src_ind:",s_y,"tgt_ind:",t_y)

    src_loss1 = loss(p_s1 , s_y)
    
    tgt_loss1 = loss_w(p_t1 , t_y)

    src_loss2 = loss(p_s2 , s_y)
    tgt_loss2 = loss_w(p_t2 , t_y)

    grad_cossim11 = []
    #grad_mse11 = []
    grad_cossim22 = []
    #grad_mse22 = []

    #netE+C1
    for n, p in netC1.named_parameters():
        # if len(p.shape) == 1: continue

        real_grad = grad([src_loss1],
                            [p],
                            create_graph=True,
                            only_inputs=True,
                            allow_unused=False)[0]
        fake_grad = grad([tgt_loss1],
                            [p],
                            create_graph=True,
                            only_inputs=True,
                            allow_unused=False)[0]

        if len(p.shape) > 1:
            _cossim = F.cosine_similarity(fake_grad, real_grad, dim=1).mean()
        else:
            _cossim = F.cosine_similarity(fake_grad, real_grad, dim=0)
        #_mse = F.mse_loss(fake_grad, real_grad)
        grad_cossim11.append(_cossim)
        #grad_mse.append(_mse)

    grad_cossim1 = torch.stack(grad_cossim11)
    gm_loss1 = (1.0 - grad_cossim1).mean()
    #grad_mse1 = torch.stack(grad_mse)
    #gm_loss1 = (1.0 - grad_cossim1).sum() * args.Q + grad_mse1.sum() * args.Z
    #netE+C2
    for n, p in netC2.named_parameters():
        # if len(p.shape) == 1: continue

        real_grad = grad([src_loss2],
                            [p],
                            create_graph=True,
                            only_inputs=True)[0]
        fake_grad = grad([tgt_loss2],
                            [p],
                            create_graph=True,
                            only_inputs=True)[0]

        if len(p.shape) > 1:
            _cossim = F.cosine_similarity(fake_grad, real_grad, dim=1).mean()
        else:
            _cossim = F.cosine_similarity(fake_grad, real_grad, dim=0)
        #_mse = F.mse_loss(fake_grad, real_grad)
        grad_cossim22.append(_cossim)
        #grad_mse.append(_mse)

    grad_cossim2 = torch.stack(grad_cossim22)
    #grad_mse2 = torch.stack(grad_mse)
    #gm_loss2 = (1.0 - grad_cossim2).sum() * args.Q + grad_mse2.sum() * args.Z
    gm_loss2 = (1.0 - grad_cossim2).mean()
    gm_loss = (gm_loss1 + gm_loss2)/2.0
        
    return gm_loss



def Entropy_div(input_):
    epsilon = 1e-5
    input_ = torch.mean(input_, 0) + epsilon
    entropy = input_ * torch.log(input_)
    entropy = torch.sum(entropy)
    return entropy 

def Entropy_condition(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1).mean()
    return entropy 

def Entropy_inf(input_):
    return Entropy_condition(input_) + Entropy_div(input_)

def Weighted_CrossEntropy(input_,labels):
    input_s = F.softmax(input_)
    entropy = -input_s * torch.log(input_s + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    weight = 1.0 + torch.exp(-entropy)
    weight = weight / torch.sum(weight).detach().item()
    #print("cross:",nn.CrossEntropyLoss(reduction='none')(input_, labels))
    return torch.mean(weight * nn.CrossEntropyLoss(reduction='none')(input_, labels))

###softmax cross entropy
def ent(output):
    return - torch.mean(output * torch.log(output + 1e-5))
# def ent(input_):
#     # return - torch.mean(output * torch.log(output + 1e-5))
#     entropy = -input_ * torch.log(input_ + 1e-5)
#     entropy = torch.sum(entropy, dim=1) #按列序进行求和，（128，1）
#     return entropy
def discrepancy(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1)- F.softmax(out2)))
#减法
def cdd1(output_t1,output_t2):
    mul = output_t1.transpose(0, 1).mm(output_t2)
    mul_t1 =  mul - mul.transpose(0,1)
    mul_t2 = mul_t1 - mul_t1.transpose(0,1)
    mul_t3 = torch.clamp(mul_t2,min=0)
    cdd_loss = torch.sum( mul_t3)
    cdd_diag = torch.trace(mul)
    return cdd_loss#-cdd_diag  #+ c2#c1# + c2# + c

def cdd(output_t1,output_t2):
    mul = output_t1.transpose(0, 1).mm(output_t2)
    cdd_loss = torch.sum(mul) - torch.trace(mul)
    return cdd_loss

# 有效果
def cdds(output_t1,output_t2):
    mul = output_t1.mm(output_t2.transpose(0, 1))
    # 对角线元素越大越好
    Tr = torch.trace(mul)
    # 非对角元素对称位置尽量相等
    NotTr = torch.sum(torch.abs(mul - mul.transpose(0, 1)))/2
    cdd_loss =   torch.sum(mul) #- 1*NotTr#-(torch.sum(mul) - torch.trace(mul))#负值
    return cdd_loss

def cdds2(output_t1,output_t2):
    mul = output_t1.mm(output_t2.transpose(0, 1))
    cdd_loss =  torch.sum(mul)- torch.trace(mul)#正值
    return cdd_loss
# 加法
def cdd2(output_t1,output_t2):
    mul = output_t1.mm(output_t2.transpose(0, 1))
    mul_t1 =  mul + mul.transpose(0,1) - torch.diag(mul)
    # 取矩阵的上三角部分
    mul_t2 = torch.triu(mul_t1)
    cdd_loss = 2 * torch.trace(mul_t2)  -torch.sum(mul_t2)
    return cdd_loss
#乘法
def cdd3(output_t1,output_t2):
    add = (output_t1 + output_t2)/2
    add_loss = add * add
    mul = output_t1 * output_t1 * output_t1
    mul2 = output_t2 * output_t2 * output_t2
    # mul_F = F.softmax(mul)
    # loss_F = - torch.mean(mul_F * torch.log(mul_F + 1e-5))
    cdd_loss =  torch.sum(mul) + torch.sum(mul2)
    return  cdd_loss
#用一个分类器
def cdd_one(output_t1):
    mul = output_t1 * output_t1 * output_t1
    # mul_row = torch.sum(mul,dim=1) #(16,1)
    # mul_col = torch.sum(mul,dim=0)
    # mul_div = torch.sum(mul_col*mul_col)
    # mul_ent = torch.sum(mul_col)
    cdd_loss = torch.sum(mul)
    return  cdd_loss


def cdd9(output_t1):
    mul = output_t1.transpose(0, 1).mm(output_t1)
    cdd_loss = torch.sum(mul) - torch.trace(mul)
    return cdd_loss

def loss_11(output_t1):
    mul = output_t1 * output_t1
    cdd_loss = torch.sqrt(torch.sum(mul))
    return cdd_loss

def loss_l2(output_t1):
    return torch.norm(output_t1)

def loss_nuc(output_t1):
    return torch.norm(output_t1,p="nuc")

def loss_inf(output_t1):
    return torch.norm(output_t1,p="inf",dim=1,keepdim=1)

# def cdd3_(output_t1,output_t2):
#     add = F.softmax(output_t1 + output_t2)
#     add_loss = add * add * add
#     mul = output_t1 * output_t1 * output_t1
#     mul2 = output_t2 * output_t2 * output_t2
#     # mul_F = F.softmax(mul)
#     # loss_F = - torch.mean(mul_F * torch.log(mul_F + 1e-5))
#     cdd_loss = 10 * torch.sum(add_loss) - torch.sum(mul) - torch.sum(mul2)
#     return  cdd_loss
# def hsic(x, y):
#     Kx = np.expand_dims(x, 0) - np.expand_dims(x, 1)
#     Kx = np.exp(- Kx**2) # 计算核矩阵
 
#     Ky = np.expand_dims(y, 0) - np.expand_dims(y, 1)
#     Ky = np.exp(- Ky**2) # 计算核矩阵

#     Kxy = np.dot(Kx, Ky)
#     n = Kxy.shape[0]
#     h = np.trace(Kxy) / n**2 + np.mean(Kx) * np.mean(Ky) - 2 * np.mean(Kxy) / n
#     return h * n**2 / (n - 1)**2

# def hsic(x, y):
#     Kx = torch.unsqueeze(x, 0) - torch.unsqueeze(x, 1)
#     Kx = torch.exp(- Kx**2) # 计算核矩阵
 
#     Ky = torch.unsqueeze(y, 0) - torch.unsqueeze(y, 1)
#     Ky = torch.exp(- Ky**2) # 计算核矩阵

#     Kxy = Kx.mm(Ky)
#     n = Kxy.shape[0]
#     h = torch.trace(Kxy) / n**2 + torch.mean(Kx) * torch.mean(Ky) - 2 * torch.mean(Kxy) / n
#     return h * n**2 / (n - 1)**2

def kernel_matrix(x, sigma):
    ndim = x.dim()
    x1 = x.unsqueeze(0)
    x2 = x.unsqueeze(1)
    axis = tuple(range(2, ndim+1))
    return torch.exp(-0.5*torch.sum(torch.pow(x1-x2, 2), dim=axis) / sigma ** 2)

def hsic(Kx, Ky, m):
    Kxy = torch.mul(Kx, Ky)
    h = torch.sum(torch.diagonal(Kxy)) / m ** 2 + torch.mean(Kx) * torch.mean(Ky) - 2 * torch.mean(Kxy) / m
    return h * (m / (m-1)) ** 2

