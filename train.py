from GTS2_st import *
import torch
from torch import nn
import sys
from utils import *
from DataLoader import *
from torch.cuda import amp
from transformers import get_cosine_schedule_with_warmup
import ast
import time
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import roc_curve, auc

def test_fun(net, test_set, criterion ,batch_size, divide = True, read = False,param_name = "divide.params", param_path = "parameters/"):
    if read:
        dic = torch.load(param_path + param_name)
        net.load_state_dict(dic["model_state_dict"])
    net.eval()
    device_ids      = [torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")]
    res_ls = []
    total_loss = 0.0
    labels_all = []
    cnt = 0
    enable_amp  = True if "cuda" in device_ids[0].type else False
    scaler      = amp.GradScaler(enabled= enable_amp)
    # net         = nn.DataParallel(net, device_ids = device_ids)
    net.to(device_ids[0])
    print("\n begin test on %s\n"%str(device_ids))
    test_iter  = torch.utils.data.DataLoader(test_set, batch_size = batch_size)
    for idx, value in enumerate(test_iter):
        ini_time    = time.time()
        input_ids, seg_ids, att_mask, labels, index, head_ids, rel_ids, tail_ids= value
        input_ids   = input_ids.to(device_ids[0])
        att_mask    = att_mask.to(device_ids[0])  
        labels      = labels.to(device_ids[0])
        seg_ids     = seg_ids.to(device_ids[0])
        head_values = index[0].clone().detach().to(device_ids[0])
        tail_values = index[2].clone().detach().to(device_ids[0])
        rel_values  = index[1].clone().detach().to(device_ids[0])
        labels_all.append(labels.flatten())
        if divide:
            head_ids    = head_ids.to(device_ids[0])
            rel_ids     = rel_ids.to(device_ids[0])
            tail_ids    = tail_ids.to(device_ids[0])
            with torch.no_grad():
                with amp.autocast(enabled = enable_amp):
                    output  = net(input_ids, seg_ids, att_mask,head_values,tail_values, rel_values, edge_idx, edge_type,
                                         head_ids, rel_ids, tail_ids)
                    res_ls.append(output.flatten())
        else:
            with torch.no_grad():
                with amp.autocast(enabled= enable_amp):
                    output  = net(input_ids, seg_ids, att_mask,head_values,tail_values, rel_values, edge_idx, edge_type)
                    res_ls.append(output.flatten())
        # calculate loss
        with torch.no_grad():
            loss            = criterion(output, labels.view(-1,1).float())
            total_loss     += float(loss)
            cnt            += 1
    res = get_result(torch.sigmoid(torch.cat(res_ls)).cpu())
    res_before = torch.sigmoid(torch.cat(res_ls)).cpu()
    labels = torch.cat(labels_all).cpu()
    f1 = f1_score(labels, res)
    macro_f1 = f1_score(labels, res, average='macro')
    p = precision_score(labels, res)
    r = recall_score(labels, res)
    acc = accuracy_score(labels, res)
    print("\nF1-SCORE be: ", f1)
    print("\nMacro-F1 be: ", macro_f1)
    print("\nloss be: ", total_loss/cnt)
    print("\nAccuracy be: ", acc)
    if read:
        with open("test_loss", "w") as f:
            f.write("%s"%(total_loss/cnt))
        with open("test_result", "w") as f:
            f.write("%s"%res_ls)
        with open("test_labels", "w") as f:
            f.write("%s"%labels_all)
        with open("F1_MacroF1_summary_compbert_co_attfusion", "a") as f:
            f.write("For model "+param_name+" F1 %s"%f1)
            f.write(",Macro F1 %s"%macro_f1)
            f.write(",precision %s"%p)
            f.write(",recall %s"%r)
            f.write(",accuracy %s\n"%acc)
    return f1, macro_f1, torch.sigmoid(torch.cat(res_ls)).cpu(), torch.cat(labels_all).cpu(), round(total_loss/cnt,4)
    

def run_with_amp(net, train_set, test_set ,criterion, batch_size, gradient_accumulate_step, max_grad_norm, epochs=1, optimizer=None, scheduler=None, divide = True,read = True, param_name = "divide.params", param_path = "parameters/", log_path = "log_divide_info/"):
    # num_gpu     = torch.cuda.device_count()
    if not cpu:
        # device_ids  = [try_gpu(i) for i in range(num_gpu)]
        device_ids      = [torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")]
    else:
        device_ids      = [torch.device("cpu")]
    if read:
        print("Read parameters to finetune:")
        dic = torch.load(param_path + param_name)
        net.load_state_dict(dic["model_state_dict"])

    print("\ntrain on %s\n"%str(device_ids))
    enable_amp  = True if "cuda" in device_ids[0].type else False
    scaler      = amp.GradScaler(enabled= enable_amp)
    net.to(device_ids[0])
    train_iter  = torch.utils.data.DataLoader(train_set, batch_size = batch_size)
    current_F1 = 0.0
    current_valid_loss = 1000.0
    valid_loss_ls = []
    total_F1 = []
    freeze = False
    max_F1 = 0
    tt_loss = 0
    for epoch in range(epochs):
        net.train()
        for idx, value in enumerate(train_iter):
            ini_time    = time.time()
            input_ids, seg_ids, att_mask, labels, index, head_ids, rel_ids, tail_ids= value
            input_ids   = input_ids.to(device_ids[0])
            att_mask    = att_mask.to(device_ids[0])
            labels      = labels.to(device_ids[0])
            seg_ids     = seg_ids.to(device_ids[0])
            head_values = index[0].clone().detach().to(device_ids[0])
            tail_values = index[2].clone().detach().to(device_ids[0])
            rel_values  = index[1].clone().detach().to(device_ids[0])
            if divide:
                head_ids    = head_ids.to(device_ids[0])
                rel_ids     = rel_ids.to(device_ids[0])
                tail_ids    = tail_ids.to(device_ids[0])
                # when forward process, use amp
                with amp.autocast(enabled = enable_amp):
                    output      = net(input_ids, seg_ids, att_mask,head_values,tail_values, rel_values, edge_idx, edge_type,
                                      head_ids, rel_ids, tail_ids)
            else:
                with amp.autocast(enabled= enable_amp):
                    output      = net(input_ids, seg_ids, att_mask,head_values,tail_values, rel_values, edge_idx, edge_type)
            
            
            loss                = criterion(output, labels.view(-1,1).float())
            if gradient_accumulate_step > 1:
                # 如果显存不足，通过 gradient_accumulate 来解决
                loss    = loss/gradient_accumulate_step
            
            # 放大梯度，避免其消失
            scaler.scale(loss).mean().backward()
            # do the gradient clip
            gradient_norm = nn.utils.clip_grad_norm_(net.parameters(),max_grad_norm)
            
            if (idx + 1) % gradient_accumulate_step == 0:
                # 多少 step 更新一次梯度
                # 通过 scaler.step 来unscale 回梯度值， 如果气结果不是infs 和Nans， 调用optimizer.step()来更新权重
                # 否则忽略step调用， 保证权重不更新
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                #print("train 1 times")
            tt_loss += loss.item()
            # 每1000次计算 print 出一次loss
            if idx % 50  == 0 or idx == len(train_iter) -1:
                with torch.no_grad():
                    if idx !=0:
                        bal = tt_loss /50
                        write_log("For model "+param_name+":",log_path + "train_log")
                        write_log("batch average loss: " + str(bal) + "; grad_norm: " + str(gradient_norm),log_path + "train_log")
                        print("For model "+param_name+":")
                        print("==============Epochs "+ str(epoch) + " ======================")
                        print("batch average loss: " + str(bal) + "; grad_norm: " + str(gradient_norm))
                        tt_loss = 0
            
            with open(log_path + "divide_log_train", "a") as f:
                f.write("Epoch %s, Batch %s: %.4f sec\n"%(epoch, idx, time.time() - ini_time))
        print("\n In epoch ", epoch)
        F1, macro_F1 ,res_ls, labels_all,avg_loss = test_fun(net, test_set, criterion ,batch_size, divide, read = False, param_path = param_path)
        # use macro as evaluation 
        F1 = macro_F1
        total_F1.append(F1)
        # not freeze at all when training
        # early stopping condition
        if F1 > current_F1 or avg_loss < current_valid_loss:
            if F1 > current_F1:
                max_F1 = F1
            print("Save model "+ param_name)
            current_F1 = F1
            current_valid_loss = avg_loss
            torch.save({
              'epoch': epoch,
              'model_state_dict': net.state_dict(),
              },param_path + param_name)
            write_log("\nFor model "+param_name,log_path + "train_log")
            write_log("\nCurrent Macro F1: " + str(current_F1),log_path + "train_log")
            with open("test_loss", "w") as f:
                f.write("%s"%(avg_loss))
            with open("test_result", "w") as f:
                f.write("%s"%res_ls)
            with open("test_labels", "w") as f:
                f.write("%s"%labels_all)

        elif avg_loss >= current_valid_loss:
            valid_loss_ls.append(avg_loss)
            write_log("\nFor model "+param_name,log_path + "train_log")
            write_log("\nLoss greater than previous loss in epoch "+str(epoch),log_path + "train_log")
            # if more than seven times average loss greater than previous loss, stop training
            if len(valid_loss_ls) >= 7:
                write_log("\nFor model "+param_name,log_path + "train_log")
                write_log("\nEarly stop in epoch "+str(epoch),log_path + "train_log")
                write_log("\nMax F1 is : "+str(max_F1), log_path + "max_F1")
                break
        else:
            write_log("\nFor model "+param_name,log_path + "train_log")
            write_log("\nEarly stop in epoch "+str(epoch),log_path + "train_log")
            break

    # record F1
    with open("F1_SCORE", "w") as f:
        f.write("%s"%total_F1)
        
        
if __name__ == "__main__":
    from sklearn.model_selection import StratifiedShuffleSplit

    cpu         = not True
    torch.cuda.empty_cache()
    seed = 7
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    wiki = False
    edge_idx, edge_type, head2idx, rel2idx, all_df, num_relation, num_nodes = read_dataset("", "", final_year=2000, wiki = wiki)
    if wiki:
        df = all_df
        df["head_description"][pd.isna(df['head_description'])] = ""
        df["tail_description"][pd.isna(df['tail_description'])] = ""
        df["head"]     = df["head"]     + "," + df["head_description"]
        df["relation"] = df["relation"] + "," + df["relation_description"]
        df["tail"]     = df["tail"]     + "," + df["tail_description"]
        all_df         = df.drop(["head_description", "relation_description", "tail_description", "ids"], axis=1).loc[:, \
                                                                                    ["head", "relation", "tail", "index_where", "labels"]]                        
    s = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    for train_idx, test_idx in s.split(all_df.iloc[:, :-1], all_df.iloc[:, -1]):
        train, test = all_df.iloc[train_idx, :], all_df.iloc[test_idx, :]
    s = StratifiedShuffleSplit(n_splits=1, test_size = 0.5, random_state = seed)
    for valid_idx, test_idx in s.split(test.iloc[:,:-1], test.iloc[:,-1]):
        valid_s, test_s = test.iloc[valid_idx, :], test.iloc[test_idx,:]
    print("Done reading Dataset: ")

    # grid search result
    test_set    = language_Dataset(test_s, edge_idx, edge_type)
    train_set   = language_Dataset(train, edge_idx, edge_type)
    valid_set   = language_Dataset(valid_s, edge_idx, edge_type)
    print("Successfully create Data Loader: ")
    param_path = "./param/"
    time_stamp  = 6
    count = 1
    comp_types = ["sub", "mul", "corr"]
    num_hiddenes = [64]
    nd_dims = [32]
    num_heads = [4]
    num_basis = 32
    num_layers = [3]
    lrs = [2e-5]
#    For model corr_lr_2e-05_divide_dimgroup_20_50_num_layers_3_fu_dim_768_node.params
    for t in range(5):
        for num_layer in num_layers:
            for nd_dim, num_hiddens, num_head in zip(nd_dims, num_hiddenes, num_heads):
       # for param in param_ls:
                for comp_type in comp_types:
                   # lr,nd_dim,num_hiddens,num_layer,fusion_dim, comp_type, module, num_head = param
                    for lr in lrs:
                        module = "node"
                        divide = True
                        epoch       = 15
                        conv_dim    = [[nd_dim] + [num_hiddens]*num_layer] + [[num_hiddens] * (num_layer + 1)] * (time_stamp - 1)
                        #node_dim    = nd_dim
                        #print(module)
                        print("Start initialize: ")
                        tmp         = CompGcn_with_temporal_bert(conv_dim, num_relation, num_nodes+1, nd_dim, num_hiddens, num_basis, 1,divide = divide, 
                                time_stamp=time_stamp,comp_type = comp_type, num_layers = num_layer, num_heads = num_head, module = module, norm_shape = num_hiddens)
                        print("Initial Model parameter success: ")
                        batch_size  = 32
                        loss        = nn.BCEWithLogitsLoss()
                        optimizer   = torch.optim.AdamW(tmp.parameters(), lr = lr, eps = 1e-4)
                        scheduler   = get_cosine_schedule_with_warmup(optimizer= optimizer, num_warmup_steps = 0, num_training_steps= len(torch.utils.data.DataLoader(train_set, batch_size = batch_size)), num_cycles = 0.5)
                        if divide:
                            param_name  = comp_type +"_co_attfusion_1_times_"+str(t)+"_lr_"+str(lr)+ "_divide_" + "dimgroup_"+str(nd_dim)+"_"+str(num_hiddens)+"_num_layers_"+str(num_layer)+"_"+module+".params"
                        else: 
                            param_name  = comp_type +"_times_"+str(t)+"_lr_"+str(lr)+ "_nodivide_" + "dimgroup_"+str(nd_dim)+"_"+str(num_hiddens)+"_num_layers_"+str(num_layer)+"_fu_dim_"+str(fusion_dim)+".params"
                        print("\nStart training models:", param_name)
                        run_with_amp(tmp, train_set, valid_set, loss, batch_size, 1, 8000,epochs = epoch, optimizer=optimizer, scheduler=scheduler, divide = divide, read = False, param_name = param_name, param_path = param_path)
                        test_fun(tmp, test_set, loss, batch_size, divide, read = True, param_name = param_name, param_path = param_path)
   