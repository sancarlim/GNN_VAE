import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import List, Tuple
import math
import random

from torch.nn import functional as f


class MTPLoss:
    """ Computes the loss for the MTP model. """

    def __init__(self,
                 num_modes: int,
                 regression_loss_weight: float = 1.,
                 angle_threshold_degrees: float = 5.):
        """
        Inits MTP loss.
        :param num_modes: How many modes are being predicted for each agent.
        :param regression_loss_weight: Coefficient applied to the regression loss to
            balance classification and regression performance.
        :param angle_threshold_degrees: Minimum angle needed between a predicted trajectory
            and the ground to consider it a match.
        """
        self.num_modes = num_modes
        self.num_location_coordinates_predicted = 2  # We predict x, y coordinates at each timestep.
        self.regression_loss_weight = regression_loss_weight
        self.angle_threshold = angle_threshold_degrees

    def _get_trajectory_and_modes(self,
                                  model_prediction: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Splits the predictions from the model into mode probabilities and trajectory.
        :param model_prediction: Tensor of shape [batch_size, n_timesteps * n_modes * 2 + n_modes].
        :return: Tuple of tensors. First item is the trajectories of shape [batch_size, n_modes, n_timesteps, 2].
            Second item are the mode probabilities of shape [batch_size, num_modes].
        """
        mode_probabilities = model_prediction[:, -self.num_modes:].clone()

        desired_shape = (model_prediction.shape[0], self.num_modes, -1, self.num_location_coordinates_predicted)
        trajectories_no_modes = model_prediction[:, :-self.num_modes].clone().reshape(desired_shape)

        return trajectories_no_modes, mode_probabilities

    @staticmethod
    def _angle_between(ref_traj: torch.Tensor,
                       traj_to_compare: torch.Tensor) -> float:
        """
        Computes the angle between the last points of the two trajectories.
        The resulting angle is in degrees and is an angle in the [0; 180) interval.
        :param ref_traj: Tensor of shape [n_timesteps, 2].
        :param traj_to_compare: Tensor of shape [n_timesteps, 2].
        :return: Angle between the trajectories.
        """

        EPSILON = 1e-5

        if (ref_traj.ndim != 2 or traj_to_compare.ndim != 2 or
                ref_traj.shape[1] != 2 or traj_to_compare.shape[1] != 2):
            raise ValueError('Both tensors should have shapes (-1, 2).')

        if torch.isnan(traj_to_compare[-1]).any() or torch.isnan(ref_traj[-1]).any():
            return 180. - EPSILON

        traj_norms_product = float(torch.norm(ref_traj[-1]) * torch.norm(traj_to_compare[-1]))

        # If either of the vectors described in the docstring has norm 0, return 0 as the angle.
        if math.isclose(traj_norms_product, 0):
            return 0.

        # We apply the max and min operations below to ensure there is no value
        # returned for cos_angle that is greater than 1 or less than -1.
        # This should never be the case, but the check is in place for cases where
        # we might encounter numerical instability.
        dot_product = float(ref_traj[-1].dot(traj_to_compare[-1]))
        angle = math.degrees(math.acos(max(min(dot_product / traj_norms_product, 1), -1)))

        if angle >= 180:
            return angle - EPSILON

        return angle

    @staticmethod
    def _compute_ave_l2_norms(tensor: torch.Tensor) -> float:
        """
        Compute the average of l2 norms of each row in the tensor.
        :param tensor: Shape [1, n_timesteps, 2].
        :return: Average l2 norm. Float.
        """
        l2_norms = torch.norm(tensor, p=2, dim=2)
        avg_distance = torch.mean(l2_norms)
        return avg_distance.item()

    def _compute_angles_from_ground_truth(self, target: torch.Tensor,
                                          trajectories: torch.Tensor) -> List[Tuple[float, int]]:
        """
        Compute angle between the target trajectory (ground truth) and the predicted trajectories.
        :param target: Shape [1, n_timesteps, 2].
        :param trajectories: Shape [n_modes, n_timesteps, 2].
        :return: List of angle, index tuples.
        """
        angles_from_ground_truth = []
        for mode, mode_trajectory in enumerate(trajectories):
            # For each mode, we compute the angle between the last point of the predicted trajectory for that
            # mode and the last point of the ground truth trajectory.
            angle = self._angle_between(target[0], mode_trajectory)

            angles_from_ground_truth.append((angle, mode))
        return angles_from_ground_truth

    def _compute_best_mode(self,
                           angles_from_ground_truth: List[Tuple[float, int]],
                           target: torch.Tensor, trajectories: torch.Tensor) -> int:
        """
        Finds the index of the best mode given the angles from the ground truth.
        :param angles_from_ground_truth: List of (angle, mode index) tuples.
        :param target: Shape [1, n_timesteps, 2]
        :param trajectories: Shape [n_modes, n_timesteps, 2]
        :return: Integer index of best mode.
        """

        # We first sort the modes based on the angle to the ground truth (ascending order), and keep track of
        # the index corresponding to the biggest angle that is still smaller than a threshold value.
        angles_from_ground_truth = sorted(angles_from_ground_truth)
        max_angle_below_thresh_idx = -1
        for angle_idx, (angle, mode) in enumerate(angles_from_ground_truth):
            if angle <= self.angle_threshold:
                max_angle_below_thresh_idx = angle_idx
            else:
                break

        # We choose the best mode at random IF there are no modes with an angle less than the threshold.
        if max_angle_below_thresh_idx == -1:
            best_mode = random.randint(0, self.num_modes - 1)

        # We choose the best mode to be the one that provides the lowest ave of l2 norms between the
        # predicted trajectory and the ground truth, taking into account only the modes with an angle
        # less than the threshold IF there is at least one mode with an angle less than the threshold.
        else:
            # Out of the selected modes above, we choose the final best mode as that which returns the
            # smallest ave of l2 norms between the predicted and ground truth trajectories.
            distances_from_ground_truth = []

            for angle, mode in angles_from_ground_truth[:max_angle_below_thresh_idx + 1]:
                norm = self._compute_ave_l2_norms(target - trajectories[mode, :, :])

                distances_from_ground_truth.append((norm, mode))

            distances_from_ground_truth = sorted(distances_from_ground_truth)
            best_mode = distances_from_ground_truth[0][1]

        return best_mode

    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the MTP loss on a batch.
        The predictions are of shape [batch_size, n_ouput_neurons of last linear layer]
        and the targets are of shape [batch_size, 1, n_timesteps, 2]
        :param predictions: Model predictions for batch.
        :param targets: Targets for batch.
        :param mask: mask to be applied to pred for non-existing frames. [batch_size, n_timesteps, 1]
        :return: zero-dim tensor representing the loss on the batch.
        """

        batch_losses = torch.Tensor().requires_grad_(True).to(predictions.device)
        trajectories, modes = self._get_trajectory_and_modes(predictions)

        trajectories = trajectories * mask
        targets = targets * mask

        for batch_idx in range(predictions.shape[0]):

            angles = self._compute_angles_from_ground_truth(target=targets[batch_idx],
                                                            trajectories=trajectories[batch_idx])

            best_mode = self._compute_best_mode(angles,
                                                target=targets[batch_idx],
                                                trajectories=trajectories[batch_idx])

            best_mode_trajectory = trajectories[batch_idx, best_mode, :].unsqueeze(0)

            regression_loss = f.smooth_l1_loss(best_mode_trajectory, targets[batch_idx])

            mode_probabilities = modes[batch_idx].unsqueeze(0)
            best_mode_target = torch.tensor([best_mode], device=predictions.device)
            classification_loss = f.cross_entropy(mode_probabilities, best_mode_target)

            loss = classification_loss + self.regression_loss_weight * regression_loss

            batch_losses = torch.cat((batch_losses, loss.unsqueeze(0)), 0)

        avg_loss = torch.mean(batch_losses)

        return avg_loss



def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def check_overlap(preds):
        intersect=[]
        #y_intersect=[np.argwhere(np.diff(np.sign(preds[i,:,1].detach().cpu().numpy()-preds[j,:,1].detach().cpu().numpy()))).size > 0  for i in range(len(preds)-1) for j in range(i+1,len(preds))]
        #x_intersect=[np.argwhere(np.diff(np.sign(preds[i,:,0].detach().cpu().numpy()-preds[j,:,0].detach().cpu().numpy()[::-1]))).size > 0 for i in range(len(preds)-1)  for j in range(i+1,len(preds))]
        #intersect = [True if y and x else False for y,x in zip(y_intersect,x_intersect)]
        '''
        for i in range(len(preds)-1):
            for j in range(i+1,len(preds)):
                y_intersect=(torch.sign(preds[i,:,1]-preds[j,:,1])-torch.sign(preds[i,:,1]-preds[j,:,1])[0]).bool().any()  #True if non all-zero
                x_intersect=(torch.sign(preds[i,:,0]-reversed(preds[j,:,0]))-torch.sign(preds[i,:,0]-reversed(preds[j,:,0]))[0]).bool().any()
                intersect.append(True if y_intersect and x_intersect else False)
        '''
        y_sub = torch.cat([torch.sign(preds[i:-1,:,1]-preds[i+1:,:,1]) for i in range(len(preds)-1)])  #N(all combinations),6
        y_intersect=( y_sub - y_sub[:,0].view(len(y_sub),-1)).bool().any(dim=1) #True if non all-zero (change sign)
        x_sub = torch.cat([torch.sign(preds[i:-1,:,0]-reversed(preds[i+1:,:,0])) for i in range(len(preds)-1)])
        x_intersect = (x_sub -x_sub[:,0].view(len(x_sub),-1)).bool().any(dim=1)
        #x_intersect=torch.cat([(torch.sign(preds[i:-1,:,0]-reversed(preds[i+1:,:,0]))-torch.sign(preds[i:-1,:,0]-reversed(preds[i+1:,:,0]))[0]).bool().any(dim=1) for i in range(len(preds)-1)])
        intersect = torch.logical_and(y_intersect,x_intersect) #[torch.count_nonzero(torch.logical_and(y,x))/len(x) for y,x in zip(y_intersect,x_intersect)] #to intersect, both True
        return torch.count_nonzero(intersect)/len(intersect) #percentage of intersections between all combinations
        #y_intersect=[np.argwhere(np.diff(np.sign(preds[i,:,1].cpu()-preds[j,:,1].cpu()))).size > 0 for j in range(i+1,len(preds)) for i in range(len(preds)-1)]

    
def compute_change_pos(feats,gt, scale_factor):
    gt_vel = gt.clone()  #.detach().clone()
    feats_vel = feats[:,:,:2].clone()
    new_mask_feats = (feats_vel[:, 1:]!=0) * (feats_vel[:, :-1]!=0) 
    new_mask_gt = (gt_vel[:, 1:]!=0) * (gt_vel[:, :-1]!=0) 
    
    gt_vel[:, 1:] = (gt_vel[:, 1:] - gt_vel[:, :-1]) * new_mask_gt

    #if scale_factor==1:
    #    gt_vel[:, :1] = (gt_vel[:, :1] - (feats_vel[:, -1:]*5.398+0.0013)) * new_mask_gt[:,0:1]
    #else:
    rescale_xy=torch.ones((1,1,2), device=feats.device)*scale_factor
    gt_vel[:, :1] = (gt_vel[:, :1] - feats_vel[:, -1:]*rescale_xy) * new_mask_gt[:,0:1]

    feats_vel[:, 1:] = (feats_vel[:, 1:] - feats_vel[:, :-1]) * new_mask_feats
    feats_vel[:, 0] = 0
    
    return feats_vel, gt_vel


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters():
        if(p.requires_grad) and ("bias" not in n) and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu().numpy())
            max_grads.append(p.grad.abs().max().cpu().numpy())
    if len(max_grads) != 0:
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        plt.savefig('foo.png')

def compute_long_lat_error(pred,gt,mask):
    pred = pred*mask #B*V,T,C  (B n grafos en el batch)
    gt = gt*mask  # outputmask BV,T,C
    lateral_error = pred[:,:,0]-gt[:,:,0]
    long_error = pred[:,:,1] - gt[:,:,1]  #BV,T
    overall_num = mask.sum(dim=-1).type(torch.int)  #torch.Tensor[(BV,T)] - num de agentes (Y CON DATOS) en cada frame
    return lateral_error, long_error, overall_num


'''
def convert_global_coords_to_local(coordinates: np.ndarray,
                                   translation: Tuple[float, float, float],
                                   rotation: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Converts global coordinates to coordinates in the frame given by the rotation quaternion and
    centered at the translation vector. The rotation is meant to be a z-axis rotation.
    :param coordinates: x,y locations. array of shape [n_steps, 2].
    :param translation: Tuple of (x, y, z) location that is the center of the new frame.
    :param rotation: Tuple representation of quaternion of new frame.
        Representation - cos(theta / 2) + (xi + yi + zi)sin(theta / 2).
    :return: x,y locations in frame stored in array of share [n_times, 2].
    """
    yaw = angle_of_rotation(quaternion_yaw(Quaternion(rotation)))

    transform = make_2d_rotation_matrix(angle_in_radians=yaw)

    coords = (coordinates - np.atleast_2d(np.array(translation)[:2])).T

    return np.dot(transform, coords).T[:, :2]

'''
