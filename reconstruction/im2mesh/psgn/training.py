import os
from tqdm import trange
import torch
import torch.nn.functional as F
from im2mesh.common import chamfer_distance
from im2mesh.training import BaseTrainer
from im2mesh.utils import visualize as vis
from torch import nn
from pathlib import Path
from im2mesh.eval import MeshEvaluator
from statistics import mean

class Trainer(BaseTrainer):
    r''' Trainer object for the Point Set Generation Network.

    The PSGN network is trained on Chamfer distance. The Trainer object
    obtains methods to perform a train and eval step as well as to visualize
    the current training state by plotting the respective point clouds.

    Args:
        model (nn.Module): PSGN model
        optiimzer (PyTorch optimizer): The optimizer that should be used
        device (PyTorch device): the PyTorch device
        input_type (string): The input type (e.g. 'img')
        vis_dir (string): the visualisation directory
    '''
    def __init__(self, model, optimizer, device=None, input_type='img',
                 vis_dir=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data, epoch, margin, reg):
        r''' Performs a train step.

        The chamfer loss is calculated and an appropriate backward pass is
        performed.

        Args:
            data (tensor): training data
        '''
        self.model.train()
        anc_points = data.get('pointcloud').to(self.device)
        anc_inputs = data.get('inputs').to(self.device)

        # pos_inputs = data.get('inputs.Bias').to(self.device)

        loss, rec, reg_loss = self.compute_loss(epoch, anc_points, anc_inputs, margin, reg)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), rec.item(), reg_loss.item()
    
    def eval_step(self, data):
        r''' Performs an evaluation step.

        The chamfer loss is calculated and returned in a dictionary.

        Args:
            data (tensor): input data
        '''
        self.model.eval()

        device = self.device

        evaluator = MeshEvaluator(n_points=100000)

        points = data.get('pointcloud_chamfer').to(device)
        inputs = data.get('inputs').to(device)

        with torch.no_grad():
            points_out, _ = self.model(inputs)

        batch_size = points.shape[0]
        points_np = points.cpu()
        points_out_np = points_out.cpu().numpy()

        loss = chamfer_distance(points, points_out).mean()
        loss = loss.item()
        eval_dict = {
            'loss': loss,
        }

        return eval_dict

    def eval_step_full(self, data):
        r''' Performs an evaluation step.

        The chamfer loss is calculated and returned in a dictionary.

        Args:
            data (tensor): input data
        '''
        self.model.eval()

        device = self.device

        evaluator = MeshEvaluator(n_points=100000)

        points = data.get('pointcloud_chamfer').to(device)
        normals = data.get('pointcloud_chamfer.normals')
        inputs = data.get('inputs').to(device)

        with torch.no_grad():
            points_out, _ = self.model(inputs)

        batch_size = points.shape[0]
        points_np = points.cpu()
        points_out_np = points_out.cpu().numpy()

        completeness_list = []
        accuracy_list = []
        completeness2_list = []
        accuracy2_list = []
        chamfer_L1_list = []
        chamfer_L2_list = []
        edge_chamferL1_list = []
        edge_chamferL2_list = []
        emd_list = []
        fscore_list = []
        
        for idx in range(batch_size):
            eval_dict_pcl = evaluator.eval_pointcloud(points_out_np[idx], points_np[idx], normals_tgt=normals[idx])
            completeness_list.append(eval_dict_pcl['completeness'])
            accuracy_list.append(eval_dict_pcl['accuracy'])
            completeness2_list.append(eval_dict_pcl['completeness2'])
            accuracy2_list.append(eval_dict_pcl['accuracy2'])
            chamfer_L1_list.append(eval_dict_pcl['chamfer-L1'])
            chamfer_L2_list.append(eval_dict_pcl['chamfer-L2'])
            edge_chamferL1_list.append(eval_dict_pcl['edge-chamfer-L1'])
            edge_chamferL2_list.append(eval_dict_pcl['edge-chamfer-L2'])
            emd_list.append(eval_dict_pcl['emd'])
            fscore_list.append(eval_dict_pcl['fscore'])


        completeness = mean(completeness_list).item()
        accuracy = mean(accuracy_list).item()
        completeness2 = mean(completeness2_list).item()
        accuracy2 = mean(accuracy2_list).item()
        chamfer_L1 = mean(chamfer_L1_list).item()
        chamfer_L2 = mean(chamfer_L2_list).item()
        edge_chamferL1 = mean(edge_chamferL1_list).item()
        edge_chamferL2 = mean(edge_chamferL2_list).item()
        emd = mean(emd_list).item()
        fscore = mean(fscore_list)

        loss = chamfer_distance(points, points_out).mean()
        loss = loss.item()
        eval_dict = {
            'loss': loss,
            'chamfer': loss,
            'completeness': completeness,
            'completeness2': completeness2,
            'accuracy': accuracy,
            'accuracy2': accuracy2,
            'chamfer-L1': chamfer_L1,
            'chamfer-L2': chamfer_L2,
            'edge-chamfer-L2': edge_chamferL2,
            'edge-chamfer-L1': edge_chamferL1,
            'emd': emd,
            'fscore': fscore
        }

        return eval_dict

    def visualize(self, data, epoch):
        r''' Visualizes the current output data of the model.

        The point clouds for respective input data is plotted.

        Args:
            data (tensor): input data
        '''
        device = self.device
        points_gt = data.get('pointcloud').to(device)
        inputs = data.get('inputs').to(device)

        with torch.no_grad():
            points_out, _ = self.model(inputs)

        points_out = points_out.cpu().numpy()
        points_gt = points_gt.cpu().numpy()

        batch_size = inputs.size(0)
        for i in trange(batch_size):
            out_folder = os.path.join(self.vis_dir, str(epoch))
            if not os.path.exists(out_folder):
                os.mkdir(out_folder)
            input_img_path = os.path.join(self.vis_dir, str(epoch), '%03d_in.png' % i)
            vis.visualize_data(
                inputs[i].cpu(), self.input_type, input_img_path)
            out_file = os.path.join(self.vis_dir, str(epoch), '%03d.png' % i)
            out_file_gt = os.path.join(self.vis_dir, str(epoch), '%03d_gt.png' % i)
            vis.visualize_pointcloud(points_out[i], out_file=out_file)
            vis.visualize_pointcloud(points_gt[i], out_file=out_file_gt)

    def compute_loss(self, epoch, anc_points, anc_inputs, m, reg):
        r''' Computes the loss.

        The Point Set Generation Network is trained on the Chamfer distance.

        Args:
            points (tensor): GT point cloud data
            inputs (tensor): input data for the model
        '''
        anc_points_out, anc_feature = self.model(anc_inputs)
        anc_loss = chamfer_distance(anc_points, anc_points_out).mean()

        # _, pos_feature = self.model(pos_inputs)

        # pdist = nn.PairwiseDistance(p=2)
        # anc_pos_dist = pdist(anc_feature, pos_feature)

        batch_size = anc_feature.shape[0]

        trip_list = []

        
        reg_list = []
        # vals = 0.11
        sigma = m * 0.997/3
        # l1 = nn.L1Loss()

        # for n in range(batch_size):
        #     anc_list = torch.stack([anc_feature[n]]*(2*batch_size-2), dim=0) 
        #     neg_list = torch.cat([anc_feature[0:n], anc_feature[n+1:], pos_feature[0:n], pos_feature[n+1:]], dim=0)
        #     anc_neg_dist = pdist(anc_list, neg_list)
        #     anc_neg_list = anc_neg_dist[(anc_pos_dist[n] < anc_neg_dist) * (anc_neg_dist < anc_pos_dist[n] + m)]
        #     neg_num = anc_neg_list.shape[0]
        #     if neg_num !=0:
        #         triploss_list = [torch.max(torch.tensor([anc_pos_dist[n] - anc_neg_list[n_neg] + m, 0])) for n_neg in range(neg_num)]
        #         triplet_loss = torch.mean(torch.tensor(triploss_list))
        #         trip_list.append(triplet_loss)
        
        if reg == 'gau':
            for n in range(batch_size): 
                t_feature = anc_feature[n]
                t_feature_list = torch.stack([anc_feature[n]]*(batch_size-1), dim=0)
                s_feature_list = torch.cat([anc_feature[0:n], anc_feature[n+1:]], dim=0)
                t_points_list = torch.stack([anc_points[n]]*(batch_size-1), dim=0)
                t_points = anc_points[n]
                s_points_list = torch.cat([anc_points[0:n], anc_points[n+1:]], dim=0)
                s_num = batch_size - 1
                # t_points_list = t_points[None, :, :]
                # t_points_list = t_points_list.repeat(s_points_list.shape[0],1,1)
                p_deno = torch.sum(torch.exp(-1 * chamfer_distance(t_points_list, s_points_list) / (2 * sigma)))
                p_hat_deno = torch.sum(torch.einsum('bl, bl -> b' , t_feature_list, s_feature_list))
                # p_hat_deno = sum([torch.dot(t_feature, t_feature_list[s_idx]) for s_idx in range(s_num)])
                # p_hat_nume =
                # t_reg_list = []
                # reg_t = 0.
                p = torch.exp(-1 * chamfer_distance(t_points_list, s_points_list) / (2 * sigma))/ p_deno
                p_hat = torch.einsum('bl, bl -> b' , t_feature_list, s_feature_list)/ p_hat_deno
                reg_t = torch.mean(torch.abs(p_hat-p))
                # torch.sum(F.l1_loss(p_hat, p))

                # for s_idx in range(s_num):
                #     reg_t += F.l1_loss((torch.dot(t_feature, s_feature_list[s_idx]) / p_hat_deno),
                #         (torch.exp(-1 * chamfer_distance(t_points[None,...], s_points_list[s_idx][None,...])/ (2 * sigma)) / p_deno))
                #     # t_reg_list.append(reg)
                # reg_t = reg_t / s_num
                # t_reg = torch.mean(torch.stack(t_reg_list), dim=0)
                reg_list.append(reg_t)
        else:
            for n in range(batch_size):
                t_feature = anc_feature[n]
                t_feature_list = torch.stack([anc_feature[n]]*(batch_size-1), dim=0)
                s_feature_list = torch.cat([anc_feature[0:n], anc_feature[n+1:]], dim=0)
                t_points_list = torch.stack([anc_points[n]]*(batch_size-1), dim=0)
                # t_points = anc_points[n]
                s_points_list = torch.cat([anc_points[0:n], anc_points[n+1:]], dim=0)
                s_num = batch_size - 1
                # p_deno = sum([1 - (chamfer_distance(t_points[None,...], s_points_list[s_idx][None,...]) / vals) for s_idx in range(s_num)])
                p_deno = torch.sum(1 - chamfer_distance(t_points_list, s_points_list) / vals)
                p_hat_deno = torch.sum(torch.einsum('bl, bl -> b' , t_feature_list, s_feature_list))
                # p_hat_nume =
                # t_reg_list = []
                p = (1 - chamfer_distance(t_points_list, s_points_list) / vals)/ p_deno
                p_hat = torch.einsum('bl, bl -> b' , t_feature_list, s_feature_list)/ p_hat_deno
                reg_t = torch.mean(torch.abs(p_hat-p))
                # reg_t = 0.
                # for s_idx in range(s_num):
                #     reg_t += F.l1_loss((torch.dot(t_feature, s_feature_list[s_idx]) / p_hat_deno),
                #         (1 - (chamfer_distance(t_points[None,...], s_points_list[s_idx][None,...]) / vals) / p_deno))
                #     # t_reg_list.append(reg)
                # reg_t = reg_t / s_num
                # t_reg = torch.mean(torch.stack(t_reg_list), dim=0)
                reg_list.append(reg_t)


        reg_loss = torch.mean(torch.stack(reg_list), dim=0)
            
            


        # triplet_output = torch.mean(torch.tensor(trip_list))

        # triplet_loss = nn.TripletMarginLoss(margin=m, p=2)
        # triplet_output = triplet_loss(anc_feature, pos_feature, neg_feature)
        w_a = 1
        # w_t = 1
        w_r = 1
        # loss = w_a * anc_loss + w_t * triplet_output + w_r * reg_loss
        loss = w_a * anc_loss  + w_r * reg_loss

        return loss, anc_loss, reg_loss



    def plot_dot_step(self, data, epoch, m):

        self.model.train()
        anc_points = data.get('pointcloud').to(self.device)
        anc_inputs = data.get('inputs').to(self.device)

        pos_inputs = data.get('inputs.Bias').to(self.device)

        anc_points_out, anc_feature = self.model(anc_inputs)
        anc_loss = chamfer_distance(anc_points, anc_points_out).mean()

        _, pos_feature = self.model(pos_inputs)
        batch_size = anc_feature.shape[0]

        pdist = nn.PairwiseDistance(p=2)
        anc_pos_dist = pdist(anc_feature, pos_feature)
        anc_pos_dot = []
        for m in range(batch_size):
            anc_pos = torch.dot(anc_feature[m], pos_feature[m])
            anc_pos_dot.append(anc_pos)
        # import pudb; pu.db
        anc_pos_cham = chamfer_distance(anc_points, anc_points)

        

        anc_neg_dist_list_pos = []
        mix_dist_list_pos = []

        anc_neg_cham_list_pos = []
        mix_cham_list_pos = []

        # import pudb; pu.db

        for n in range(batch_size):
            anc_list = torch.stack([anc_feature[n]]*(2*batch_size-2), dim=0) 
            neg_list = torch.cat([anc_feature[0:n], anc_feature[n+1:], pos_feature[0:n], pos_feature[n+1:]], dim=0)
            anc_neg_dist = pdist(anc_list, neg_list)
            neg_indexes = [idx for idx, dist in enumerate(anc_neg_dist) if (anc_pos_dist[n] < dist) and (dist < anc_pos_dist[n] + m)]
            # neg_num = len(neg_indexes)
            
            anc_neg_dot = []
            for m in range(2*batch_size-2):
                anc_neg = torch.dot(anc_list[m], neg_list[m])
                anc_neg_dot.append(anc_neg)

            anc_neg_list = torch.Tensor(anc_neg_dot)[neg_indexes]

            anc_neg_dist_list_pos.extend(anc_neg_list)
            mix_dist_list_pos.extend(torch.Tensor(anc_neg_dot))
        
            anc_points_list = torch.stack([anc_points[n]]*(2*batch_size-2), dim=0) 
            neg_points = torch.cat([anc_points[0:n], anc_points[n+1:], anc_points[0:n], anc_points[n+1:]], dim=0)
            # neg_points_list = neg_points[neg_indexes]
            anc_mix_cham = chamfer_distance(anc_points_list, neg_points)
            anc_neg_cham = anc_mix_cham[neg_indexes]

            anc_neg_cham_list_pos.extend(anc_neg_cham)
            mix_cham_list_pos.extend(anc_mix_cham)

        anc_neg_dist_list = []
        mix_dist_list = []

        anc_neg_cham_list = []
        mix_cham_list = []

        for n in range(batch_size):
            anc_list = torch.stack([anc_feature[n]]*(batch_size-1), dim=0) 
            neg_list = torch.cat([anc_feature[0:n], anc_feature[n+1:]], dim=0)
            anc_neg_dist = pdist(anc_list, neg_list)
            neg_indexes = [idx for idx, dist in enumerate(anc_neg_dist) if (anc_pos_dist[n] < dist) and (dist < anc_pos_dist[n] + m)]
            # neg_num = len(neg_indexes)
            # anc_neg_list = anc_neg_dist[neg_indexes]

            anc_neg_dot = []
            for m in range(batch_size-1):
                anc_neg = torch.dot(anc_list[m], neg_list[m])
                anc_neg_dot.append(anc_neg)

            anc_neg_list = torch.Tensor(anc_neg_dot)[neg_indexes]

            anc_neg_dist_list.extend(anc_neg_list)
            mix_dist_list.extend(torch.Tensor(anc_neg_dot))
        
            anc_points_list = torch.stack([anc_points[n]]*(batch_size-1), dim=0) 
            neg_points = torch.cat([anc_points[0:n], anc_points[n+1:]], dim=0)
            # neg_points_list = neg_points[neg_indexes]
            anc_mix_cham = chamfer_distance(anc_points_list, neg_points)
            anc_neg_cham = anc_mix_cham[neg_indexes]
            

            anc_neg_cham_list.extend(anc_neg_cham)
            mix_cham_list.extend(anc_mix_cham)

        anc_neg_dist_list_pos = torch.Tensor(anc_neg_dist_list_pos).cpu().detach().numpy().tolist()
        mix_dist_list_pos = torch.Tensor(mix_dist_list_pos).cpu().detach().numpy().tolist()

        anc_neg_cham_list_pos = torch.Tensor(anc_neg_cham_list_pos).cpu().detach().numpy().tolist()
        mix_cham_list_pos = torch.Tensor(mix_cham_list_pos).cpu().detach().numpy().tolist()

        anc_neg_dist_list = torch.Tensor(anc_neg_dist_list).cpu().detach().numpy().tolist()
        mix_dist_list = torch.Tensor(mix_dist_list).cpu().detach().numpy().tolist()

        anc_neg_cham_list = torch.Tensor(anc_neg_cham_list).cpu().detach().numpy().tolist()
        mix_cham_list = torch.Tensor(mix_cham_list).cpu().detach().numpy().tolist()

        anc_pos_dot = torch.Tensor(anc_pos_dot).cpu().detach().numpy().tolist()
        anc_pos_cham = anc_pos_cham.cpu().detach().numpy().tolist()
        

        return anc_loss.item(), anc_neg_dist_list_pos, mix_dist_list_pos, anc_neg_cham_list_pos, mix_cham_list_pos, anc_neg_dist_list, mix_dist_list, anc_neg_cham_list, mix_cham_list, anc_pos_dot, anc_pos_cham

