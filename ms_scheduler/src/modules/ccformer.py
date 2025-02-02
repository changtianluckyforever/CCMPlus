import copy

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat

from modules.cclayers import SeriesDec, SeasonalLayerNorm, TemporalEmbedding
from prediction.transformer import TransformerEncoder
import torch.nn.functional as F
from modules.StandardNorm import Normalize

def get_manifold_e_by_tau(taus, tau_w=100):
    Es = np.floor(tau_w / taus).astype(np.int32)
    for i in range(len(Es)):
        if Es[i] % 2 == 0:
            Es[i] -= 1
    return Es


class ManifoldEmbedding(nn.Module):
    def __init__(self, c_in, d_model, args, tau, E):
        """
        https://arxiv.org/pdf/comp-gas/9602002.pdf
        tau_w = (E-1)*tau    # 100
        upper bound of tau_w is 2 sqrt(3x_0/x_1), where x_0 and x_1 are the mean of the time series and its first derivative, respectively.
        :param c_in:
        :param d_model:
        :param tau:
        :param E:
        """
        super(ManifoldEmbedding, self).__init__()
        self.args = args
        self.tau = tau
        self.E = E
        self.eps = torch.tensor(1e-6)
        # kernel_size means the length of the convolutional kernel, dilation means the distance between the elements of the kernel
        self.conv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=E, bias=False, dilation=tau)
        n_manifold_embed = args.seq_len - tau * (E - 1)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
        # self.conv_out_proj = AttentionLayer(DSAttention(), d_model, n_heads=1)
        self.conv_out_proj = nn.Sequential(nn.Linear(n_manifold_embed, 1), nn.GELU())
        self.low_rank_corr = args.low_rank_corr
        self.low_rank_d = args.low_rank_d
        if self.low_rank_corr:
            self.W = nn.Parameter(nn.init.orthogonal_(torch.Tensor(n_manifold_embed, args.low_rank_d)))
            self.U = nn.Parameter(nn.init.orthogonal_(torch.Tensor(n_manifold_embed, args.low_rank_d)))
            self.W_0 = nn.Parameter(nn.init.orthogonal_(torch.Tensor(args.seq_len, args.low_rank_d)))
            self.U_0 = nn.Parameter(nn.init.orthogonal_(torch.Tensor(args.seq_len, args.low_rank_d)))
        self.sparse_corr = args.sparse_corr
        self.momentum_pearson_flag = args.momentum_pearson_flag
        self.momentum_pearson_value = args.momentum_pearson_value
        self.args = args

        # in the following, we will write the new version of the forward function in order to use the pearson_sparse_matrix

    def forward(self, x, target, pearson_sparse_matrix=None):
        # the shape of x is [B, N, Lx, 1 + args.c_in],
        # the target shape is [B, N, Lx].
        # the shape of pearson_sparse_matrix is [N, N]
        # here the target is the time series,
        # it means the corresponding ground truth of the time series for time seriesfeatures x.
        B, N, L, H = x.shape
        inp_x = rearrange(x, "b n l h -> (b n) h l")
        conv_out = self.conv(inp_x)
        # Output_Length=⌊Input_Length−τ×(E−1)⌋
        # the shape of conv_out is [B*N, D, Lx - tau * (E - 1)],
        # where D is the args.d_model for the manifold embedding
        pearson_cor = pearson_sparse_matrix
        # if in the test phase, we do not need to calculate the pearson_cor, thepearson_cor is unchanged.
        # now the shape of pearson_cor is [N, N]
        with torch.no_grad():
            if self.args.model_training_flag:
                # could add self.args.current_iter as more conditions,
                # if we want to limit the number of iterations with this condition
                if self.args.current_epoch == 0:
                    # Subtract the mean from each element in the last dimension
                    vx = target - target.mean(dim=-1, keepdim=True)
                    # the shape of target is [B, N, Lx]
                    # the shape of vx is [B, N, Lx]
                    vx_hat = vx  # Since vx and vx_hat are the same in your case, you can reuse vx
                    # the shape of vx_hat is [B, N, Lx]
                    if self.low_rank_corr:
                        # the shape of self.W_0 is [args.seq_len, args.low_rank_d]
                        vx = torch.matmul(vx, self.W_0)
                        # the shape of vx is [B, N, args.low_rank_d]
                        # the shape of self.U_0 is [args.seq_len, args.low_rank_d]
                        vx_hat = torch.matmul(vx_hat, self.U_0)
                        # the shape of vx_hat is [B, N, args.low_rank_d]

                    if self.momentum_pearson_flag:
                        # now the shape of pearson_cor is [N, N], we want to expand it as [B, N, N]
                        pearson_cor_past = pearson_cor.unsqueeze(0).expand(B, -1, -1)
                        # the shape of pearson_cor is [B, N, N]
                        pearson_cor_now = torch.matmul(vx, vx_hat.mT) / (
                            torch.matmul(
                                torch.sqrt(torch.sum(vx**2, dim=-1, keepdim=True) + self.eps),
                                torch.sqrt(torch.sum(vx_hat**2, dim=-1, keepdim=True).mT + self.eps),
                            )
                        )
                        # the shape of torch.matmul(vx, vx_hat.mT) is [B, N, N]
                        pearson_cor = (
                            self.momentum_pearson_value * pearson_cor_past
                            + (1 - self.momentum_pearson_value) * pearson_cor_now
                        )
                        # the shape of pearson_cor is [B, N, N]

                    # if do not use the momentum pearson matrix
                    else:
                        pearson_cor = torch.matmul(vx, vx_hat.mT) / (
                            torch.matmul(
                                torch.sqrt(torch.sum(vx**2, dim=-1, keepdim=True) + self.eps),
                                torch.sqrt(torch.sum(vx_hat**2, dim=-1, keepdim=True).mT + self.eps),
                            )
                        )
                        # the shape of torch.matmul(vx, vx_hat.mT) is [B, N, N]
                        # the shape of pearson_cor is [B, N, N]

                    # 挑出来和第一个服务最相关的那几个服务
                    if self.sparse_corr:
                        if self.args.use_epsilon_greedy_sparse:
                            # self.low_rank_d is the number of top-k values to keep
                            # Define epsilon for epsilon-greedy policy
                            epsilon = self.args.epsilon
                            # the shape of pearson_cor is [B, N, N]
                            B, N, _ = pearson_cor.shape
                            # Get the top K values and indices
                            top_k_values, top_k_indices = torch.topk(pearson_cor, k=self.low_rank_d, dim=-1)
                            # Create a mask for epsilon-greedy policy
                            mask = torch.rand(B, N, self.low_rank_d, device=pearson_cor.device) < epsilon
                            # Create a tensor of all indices
                            all_indices = torch.arange(N, device=pearson_cor.device).expand(B, N, N)
                            # Create a mask to exclude top K indices
                            top_k_mask = torch.zeros_like(all_indices, dtype=torch.bool)
                            top_k_mask.scatter_(2, top_k_indices, True)
                            # Mask the top K indices to get remaining indices
                            remaining_indices = all_indices[~top_k_mask].view(B, N, N - self.low_rank_d)
                            # Randomly select from the remaining indices
                            rand_indices = torch.randint(
                                0, N - self.low_rank_d, (B, N, self.low_rank_d), device=pearson_cor.device
                            )
                            random_selected_indices = remaining_indices.gather(2, rand_indices)
                            # Apply mask to select random indices where needed
                            selected_indices = torch.where(mask, random_selected_indices, top_k_indices)
                            # Get corresponding values for the selected indices
                            selected_values = torch.gather(pearson_cor, dim=-1, index=selected_indices)
                            # Create a sparse correlation matrix
                            pearson_cor_sparse = torch.zeros_like(pearson_cor)
                            pearson_sparse_multiplier = torch.scatter(
                                pearson_cor_sparse, -1, selected_indices, selected_values
                            )
                            # the shape of pearson_sparse_multiplier is [B, N, N]
                        elif self.args.use_topk_sparse:
                            # self.low_rank_d is the number of top-k values to keep
                            top_k_values, top_k_indices = torch.topk(pearson_cor, k=self.low_rank_d, dim=-1)
                            pearson_cor_sparse = torch.zeros_like(pearson_cor)
                            # the shape of pearson_cor_sparse is [B, N, N]
                            pearson_sparse_multiplier = torch.scatter(
                                pearson_cor_sparse, -1, top_k_indices, top_k_values
                            )
                            # the shape of pearson_sparse_multiplier is [B, N, N]
                        else:
                            # Randomly select k unique values in the pearson_cor matrix
                            B, N, _ = pearson_cor.shape
                            # Generate a random permutation for each batch and tile it to shape (B, N, N)
                            rand_indices = torch.rand(B, N, N, device=pearson_cor.device).argsort(dim=-1)
                            # Select the first self.low_rank_d indices from each row in the last dimension
                            selected_indices = rand_indices[:, :, : self.low_rank_d]
                            # the shape of selected_indices is [B, N, self.low_rank_d]

                            # Gather the random values using these unique indices
                            random_values = torch.gather(pearson_cor, dim=-1, index=selected_indices)
                            # Initialize a sparse tensor with zeros
                            pearson_cor_sparse = torch.zeros_like(pearson_cor)
                            # Scatter the random values into the sparse matrix
                            pearson_sparse_multiplier = torch.scatter(
                                pearson_cor_sparse, -1, selected_indices, random_values
                            )
                            # The shape of pearson_sparse_multiplier is [B, N, N]

                        if self.args.normalize_multiplier:
                            # pearson_multiplier = pearson_sparse_multiplier / torch.sum(
                            #     pearson_sparse_multiplier, dim=-1, keepdim=True
                            # )
                            pearson_multiplier = F.softmax(pearson_sparse_multiplier, dim=-1)
                            # the shape of pearson_multiplier is [B, N, N]
                        else:
                            pearson_multiplier = pearson_sparse_multiplier
                            # the shape of pearson_multiplier is [B, N, N]

                    else:
                        pearson_multiplier = pearson_cor
                        # the shape of pearson_multiplier is [B, N, N]
                        if self.args.normalize_multiplier:
                            # pearson_multiplier = pearson_multiplier / torch.sum(
                            #     pearson_multiplier, dim=-1, keepdim=True
                            # )
                            pearson_multiplier = F.softmax(pearson_multiplier, dim=-1)
                            # the shape of pearson_multiplier is [B, N, N]
                        else:
                            pearson_multiplier = pearson_multiplier
                            # the shape of pearson_multiplier is [B, N, N]

                    # mean the first dimension of pearson_cor
                    # the shape of pearson_cor is [B, N, N]
                    pearson_cor = torch.mean(pearson_cor, dim=0)
                    # the shape of pearson_cor is [N, N]

                elif self.args.current_epoch != 0:
                    # now the shape of pearson_cor is [N, N]
                    # next we will update the pearson_cor matrix with Convergent Cross Mapping method
                    # now the shape of conv_out is  [B*N, D, Lx - tau * (E - 1)], where D is the args.d_model for the manifold embedding
                    # next we want to reshape it as [B, N, Lx - tau * (E - 1), D], and then send to another variable conv_out_ccm
                    L_out = conv_out.size(-1)
                    # the shape of conv_out is [B*N, D, L_out]
                    conv_out_ccm = conv_out.permute(0, 2, 1).view(B, N, L_out, -1)
                    # the shape of conv_out_ccm is [B, N, Lx - tau * (E - 1), self.args.d_model]
                    # Lx - tau * (E - 1) = L_out
                    # the target shape is [B, N, Lx]
                    target_ccm = target[:, :, : conv_out_ccm.shape[-2]]
                    # the shape of target_ccm is [B, N, L_out]

                    conv_out_ccm_flat = conv_out_ccm
                    # the shape of conv_out_ccm_flat is [B, N, L_out, d_model]
                    d_model = conv_out_ccm_flat.size(-1)

                    # Compute pairwise distances across L_out within the same service series
                    dist_matrix = torch.cdist(conv_out_ccm_flat, conv_out_ccm_flat, p=2)  # Shape: [B, N, L_out, L_out]

                    # Sort the distances and select the nearest neighbors
                    _, nearest_indices = dist_matrix.sort(dim=-1)
                    nearest_indices = nearest_indices[:, :, :, 1 : d_model + 2]
                    # Select (d_model + 1) nearest neighbors, excluding the state itself
                    # the shape of nearest_indices is [B, N, L_out, d_model+1]

                    # Gather the nearest distances
                    nearest_distances = torch.gather(dist_matrix, -1, nearest_indices)
                    # the shape of nearest_distances is [B, N, L_out, d_model+1]

                    # the shape of target_ccm is [B, N, L_out]
                    # Expand target_ccm for gathering values across all services
                    Y_i_expanded = target_ccm.unsqueeze(1).expand(B, N, N, L_out)  # Shape: [B, N, N, L_out]
                    # the shape of Y_i_expanded is [B, N, N, L_out], 第一个N是对后面的 （N, L_out）复制扩展, 第二个N代表有N个service

                    # Expand and transpose Y_i_expanded to align with nearest_indices_expanded
                    Y_i_expanded = Y_i_expanded.unsqueeze(-1).expand(
                        -1, -1, -1, -1, L_out
                    )  # Sh ape: [B, N, N, L_out, L_out]
                    Y_i_transposed = Y_i_expanded.transpose(-1, -2)  # Shape: [B, N, N, L_out, L_out]
                    # in the dimension of Y_i_transposed, 第一个L_out代表对时间序列重复L_out次, 第一个N是对后面的 （N, L_out, L_out）复制扩展, 第二个N代表有N个service

                    # the shape of nearest_indices is [B, N, L_out, d_model+1]
                    # Use the nearest_indices to gather corresponding Y_i values across all services
                    nearest_indices_expanded = nearest_indices.unsqueeze(2).expand(
                        B, N, N, L_out, d_model + 1
                    )  # Shape: [B, N, N, L_out, d_model + 1]
                    # the shape of Y_i_transposed is [B, N, N, L_out, L_out]
                    # the shape of nearest_indices_expanded is [B, N, N, L_out, d_model+1],
                    # 第二个N是对后面的 （L_out, d_model+1）复制扩展N 次, 第一个N代表有N个service

                    # the shape of Y_i_transposed is [B, N, N, L_out, L_out]
                    # the shape of nearest_indices_expanded is [B, N, N, L_out, d_model+1]
                    Y_nearest = torch.gather(
                        Y_i_transposed, 3, nearest_indices_expanded
                    )  # Shape: [B, N, N, L_out, d_model + 1]
                    # the shape of Y_nearest is [B, N, N, L_out, d_model + 1]
                    # 第一个N代表有N个service, 第二个N代表N个service其中的一个service的nearest index， （L_out, d_model + 1）， 在N个service中抽取的values

                    # the shape of nearest_distances is [B, N, L_out, d_model+1]
                    # the shape of u_z is [B, N, L_out, d_model+1]
                    # the shape of w_z is [B, N, L_out, d_model+1]
                    # N代表有N个service
                    # Calculate weights w_z for the nearest neighbors
                    u_z = torch.exp(
                        -nearest_distances / (nearest_distances[:, :, :, 0:1] + 1e-6)
                    )  # Add small epsilon to avoid division by zero
                    w_z = u_z / (torch.sum(u_z, dim=-1, keepdim=True) + 1e-6)  # Shape: [B, N, L_out, d_model + 1]

                    # the shape of Y_nearest is [B, N, N, L_out, d_model + 1]
                    # the shape of w_z is [B, N, L_out, d_model+1], N代表有N个service, 每一个服务对应一套权重，（L_out, d_model+1）
                    # Calculate Y_i_hat by weighted sum of the nearest neighbors
                    Y_i_hat = torch.sum(w_z.unsqueeze(2) * Y_nearest, dim=-1)
                    # the shape of Y_i_hat is [B, N, N, L_out]
                    # 第一个N代表有N个service, 第二个N代表第一个N中的service，其中的一个service去计算weight，利用ccm去预测N个service的 L_out。
                    # the shape of Y_i_hat is [B, N, N, L_out]
                    # the shape of target_ccm is [B, N, L_out]

                    # Expand target_ccm to match the shape of Y_i_hat for mean and std calculations
                    target_ccm_expanded = target_ccm.unsqueeze(1).expand(B, N, N, L_out)  # Shape: [B, N, N, L_out]
                    # 第2个N代表有N个service, 第一个N代表对后面的 （N, L_out）复制扩展N次

                    if self.low_rank_corr:
                        Y_i_hat = torch.matmul(Y_i_hat, self.W)
                        # the shape of self.W is (n_manifold_embed, args.low_rank_d)
                        # the shape of Y_i_hat is [B, N, N, args.low_rank_d]
                        target_ccm_expanded = torch.matmul(target_ccm_expanded, self.U)
                        # the shape of self.U is (n_manifold_embed, args.low_rank_d)
                        # the shape of target_ccm_expanded is [B, N, N, args.low_rank_d]

                    # Calculate the mean and standard deviation of target_ccm and Y_i_hat
                    mean_Y = target_ccm_expanded.mean(dim=-1, keepdim=True)  # Shape: [B, N, N, 1]
                    mean_Y_hat = Y_i_hat.mean(dim=-1, keepdim=True)  # Shape: [B, N, N, 1]
                    std_Y = target_ccm_expanded.std(dim=-1, keepdim=True)  # Shape: [B, N, N, 1]
                    # 第二个N代表有N个service， 第一个N代表对后面的 （N, 1）复制扩展N次
                    std_Y_hat = Y_i_hat.std(dim=-1, keepdim=True)  # Shape: [B, N, N, 1]
                    # 第一个N代表有N个service (用其中一个去生成weight）, 第二个N代表第一个N中的service，其中的一个service去计算weight，利用ccm去预测N个service的 L_out 的std

                    if self.momentum_pearson_flag:
                        # now the shape of pearson_corr is [N, N]
                        pearson_cor_past = pearson_cor.unsqueeze(0).expand(B, -1, -1)

                        # Compute the covariance
                        covariance = torch.mean(
                            (target_ccm_expanded - mean_Y) * (Y_i_hat - mean_Y_hat), dim=-1
                        )  # Shape: [B, N, N]
                        # 第一个N代表扩展N次，第二个N代表有N个service

                        # Compute the Pearson correlation coefficient
                        corr = covariance / (std_Y * std_Y_hat + 1e-6)  # Shape: [B, N, N]
                        # the shape of std_Y * std_Y_hat is [B, N, N, 1]
                        # 第一个N代表扩展N次（每一次用一个service去生成weight）, 第二个N代表有N个service
                        # the shape of covariance is [B, N, N]
                        # the shape of corr is [B, N, N]
                        # 第一个N代表扩展N次（每一次用一个service去生成weight）, 第二个N代表有N个service

                        # transpose the corr matrix, to swap the last two dimensions
                        corr = corr.transpose(1, 2)  # Shape: [B, N, N]

                        pearson_cor_now = corr
                        # the shape of pearson_cor is [B, N, N]

                        pearson_cor = self.momentum_pearson_value * pearson_cor_past + (1 - self.momentum_pearson_value) * pearson_cor_now
                        # the shape of pearson_cor is [B, N, N]

                    else:
                        # Compute the covariance
                        covariance = torch.mean(
                            (target_ccm_expanded - mean_Y) * (Y_i_hat - mean_Y_hat), dim=-1
                        )  # Shape: [B, N, N]
                        # 第一个N代表扩展N次，第二个N代表有N个service

                        # Compute the Pearson correlation coefficient
                        corr = covariance / (std_Y.squeeze(-1) * std_Y_hat.squeeze(-1) + 1e-6)  # Shape: [B, N, N]
                        # the shape of std_Y * std_Y_hat is [B, N, N, 1]
                        # 第一个N代表扩展N次（每一次用一个service去生成weight）, 第二个N代表有N个service
                        # the shape of covariance is [B, N, N]
                        # the shape of corr is [B, N, N]
                        # 第一个N代表扩展N次（每一次用一个service去生成weight）, 第二个N代表有N个service

                        # transpose the corr matrix, to swap the last two dimensions
                        corr = corr.transpose(1, 2)  # Shape: [B, N, N]

                        pearson_cor = corr
                        # the shape of pearson_cor is [B, N, N]

                    # 挑出来和第一个服务最相关的那几个服务
                    if self.sparse_corr:
                        if self.args.use_epsilon_greedy_sparse:
                            # self.low_rank_d is the number of top-k values to keep
                            # Define epsilon for epsilon-greedy policy
                            epsilon = self.args.epsilon
                            # the shape of pearson_cor is [B, N, N]
                            B, N, _ = pearson_cor.shape
                            # Get the top K values and indices
                            top_k_values, top_k_indices = torch.topk(pearson_cor, k=self.low_rank_d, dim=-1)
                            # Create a mask for epsilon-greedy policy
                            mask = torch.rand(B, N, self.low_rank_d, device=pearson_cor.device) < epsilon
                            # Create a tensor of all indices
                            all_indices = torch.arange(N, device=pearson_cor.device).expand(B, N, N)
                            # Create a mask to exclude top K indices
                            top_k_mask = torch.zeros_like(all_indices, dtype=torch.bool)
                            top_k_mask.scatter_(2, top_k_indices, True)
                            # Mask the top K indices to get remaining indices
                            remaining_indices = all_indices[~top_k_mask].view(B, N, N - self.low_rank_d)
                            # Randomly select from the remaining indices
                            rand_indices = torch.randint(
                                0, N - self.low_rank_d, (B, N, self.low_rank_d), device=pearson_cor.device
                            )
                            random_selected_indices = remaining_indices.gather(2, rand_indices)
                            # Apply mask to select random indices where needed
                            selected_indices = torch.where(mask, random_selected_indices, top_k_indices)
                            # Get corresponding values for the selected indices
                            selected_values = torch.gather(pearson_cor, dim=-1, index=selected_indices)
                            # Create a sparse correlation matrix
                            pearson_cor_sparse = torch.zeros_like(pearson_cor)
                            pearson_sparse_multiplier = torch.scatter(
                                pearson_cor_sparse, -1, selected_indices, selected_values
                            )
                            # the shape of pearson_sparse_multiplier is [B, N, N]
                        elif self.args.use_topk_sparse:
                            # self.low_rank_d is the number of top-k values to keep
                            top_k_values, top_k_indices = torch.topk(pearson_cor, k=self.low_rank_d, dim=-1)
                            pearson_cor_sparse = torch.zeros_like(pearson_cor)
                            # the shape of pearson_cor_sparse is [B, N, N]
                            pearson_sparse_multiplier = torch.scatter(
                                pearson_cor_sparse, -1, top_k_indices, top_k_values
                            )
                            # the shape of pearson_sparse_multiplier is [B, N, N]
                        else:
                            # Randomly select k unique values in the pearson_cor
                            B, N, _ = pearson_cor.shape
                            # Generate a random permutation for each batch and tile it to shape (B, N, N)
                            rand_indices = torch.rand(B, N, N, device=pearson_cor.device).argsort(dim=-1)
                            # Select the first self.low_rank_d indices from each row in the last dimension
                            selected_indices = rand_indices[:, :, : self.low_rank_d]
                            # the shape of selected_indices is [B, N, self.low_rank_d]

                            # Gather the random values using these unique indices
                            random_values = torch.gather(pearson_cor, dim=-1, index=selected_indices)
                            # Initialize a sparse tensor with zeros
                            pearson_cor_sparse = torch.zeros_like(pearson_cor)
                            # Scatter the random values into the sparse matrix
                            pearson_sparse_multiplier = torch.scatter(
                                pearson_cor_sparse, -1, selected_indices, random_values
                            )
                            # The shape of pearson_sparse_multiplier is [B, N, N]

                        # if self.args.normalize_multiplier:
                        #     pearson_multiplier = pearson_sparse_multiplier / torch.sum(
                        #         pearson_sparse_multiplier, dim=-1, keepdim=True
                        #     )
                        #     # the shape of pearson_multiplier is [B, N, N]
                        # else:
                        #     pearson_multiplier = pearson_sparse_multiplier
                        #     # the shape of pearson_multiplier is [B, N, N]

                        if self.args.normalize_multiplier:
                            pearson_multiplier = F.softmax(pearson_sparse_multiplier, dim=-1)
                            # the shape of pearson_multiplier is [B, N, N]
                        else:
                            pearson_multiplier = pearson_sparse_multiplier
                            # the shape of pearson_multiplier is [B, N, N]


                    else:
                        pearson_multiplier = pearson_cor
                        # the shape of pearson_multiplier is [B, N, N]
                        if self.args.normalize_multiplier:
                            pearson_multiplier = F.softmax(pearson_multiplier, dim=-1)
                            # the shape of pearson_multiplier is [B, N, N]
                        else:
                            pearson_multiplier = pearson_multiplier
                            # the shape of pearson_multiplier is [B, N, N]


                        # if self.args.normalize_multiplier:
                        #     pearson_multiplier = pearson_multiplier / torch.sum(
                        #         pearson_multiplier, dim=-1, keepdim=True
                        #     )
                        #     # the shape of pearson_multiplier is [B, N, N]
                        # else:
                        #     pearson_multiplier = pearson_multiplier
                        #     # the shape of pearson_multiplier is [B, N, N]

                    # mean the first dimension of pearson_cor.
                    # the shape of pearson_cor is [B, N, N]
                    pearson_cor = torch.mean(pearson_cor, dim=0)
                    # the shape of pearson_cor is [N, N]

                else:
                    print("something is wrong with the pearson_cor update process!!!!!!")

            else:
                # print("test a batch of data with ready pearson_cor matrix")
                # expand the shape of pearson_cor as [B, N, N]
                pearson_cor_test = repeat(pearson_cor, "n1 n2 -> b n1 n2", b=B)
                # the shape of pearson_cor_test is [B, N, N]
                # self.low_rank_d is the number of top-k values to keep
                top_k_values, top_k_indices = torch.topk(pearson_cor_test, k=self.low_rank_d, dim=-1)
                # the shape of top_k_values is [B, N, self.low_rank_d],
                #  the shape of top_k_indices is [B, N, self.low_rank_d]
                pearson_cor_sparse = torch.zeros_like(pearson_cor_test)
                # the shape of pearson_cor_sparse is [B, N, N]
                pearson_sparse_multiplier = torch.scatter(pearson_cor_sparse, -1, top_k_indices, top_k_values)
                # the shape of pearson_sparse_multiplier is [B, N, N]

                if self.args.normalize_multiplier:
                    pearson_multiplier = F.softmax(pearson_sparse_multiplier, dim=-1)
                    # the shape of pearson_multiplier is [B, N, N]
                else:
                    pearson_multiplier = pearson_sparse_multiplier
                    # the shape of pearson_multiplier is [B, N, N]



                # if self.args.normalize_multiplier:
                #     pearson_multiplier = pearson_sparse_multiplier / torch.sum(
                #         pearson_sparse_multiplier, dim=-1, keepdim=True
                #     )
                #     # the shape of pearson_multiplier is [B, N, N]
                # else:
                #     pearson_multiplier = pearson_sparse_multiplier
                    # the shape of pearson_multiplier is [B, N, N]
                # in the test phase, we will use the pearson_corr is unchanged for each batch of data
                # the shape of pearson_cor is [N, N]

        # the shape of conv_out is [B*N, D, Lx - tau * (E - 1)], where D is the args.d_model for the manifold embedding
        conv_out = self.conv_out_proj(conv_out).squeeze(-1)
        # the shape of conv_out is [B*N, D]
        conv_x = rearrange(conv_out, "(b n) d -> b n d", b=B)
        # the shape of conv_x is [B, N, D]
        # 让其他服务的第一个维度的数值，来根据加权决定，第一个服务的第一个特征数值
        # the data type of pearson_cor is torch.float32
        conv_x = torch.matmul(pearson_multiplier, conv_x)
        # the shape of conv_x is [B, N, D]
        # the shape of pearson_multiplier is [B, N, N]
        # the shape of pearson_cor is [N, N]
        return conv_x, pearson_cor


class MultiManifoldEmbedding(nn.Module):
    def __init__(self, c_in, d_model, args, taus=np.array([1, 2, 3, 4, 5, 6, 7, 8, 11, 18, 24])):
        super(MultiManifoldEmbedding, self).__init__()
        self.taus = taus
        self.Es = get_manifold_e_by_tau(taus)
        self.multi_space = nn.ModuleList(
            [ManifoldEmbedding(c_in, d_model, args, tau, E) for tau, E in zip(taus, self.Es)]
        )
        self.out_proj = nn.Sequential(nn.Linear(d_model, c_in), nn.GELU())

    def forward(self, x, target, pearson_sparse_matrix=None):
        B, N, L, H = x.shape
        #######
        # x_fft = torch.fft.rfft(x.mT, dim=-1)
        # amps = torch.abs(x_fft).mean(-2)[..., self.taus]
        # # amps[..., 0] = 0
        # amps = torch.softmax(amps, dim=-1)
        #######
        multi_space_x, corrs = [], []
        for i, me in enumerate(self.multi_space):
            if (L - self.taus[i] * (self.Es[i] - 1)) < 0:
                continue
            con_x, corr = me(x, target, pearson_sparse_matrix=pearson_sparse_matrix)
            # the shape of con_x is [B, N, D]
            # the shape of corr is [N, N]
            multi_space_x.append(con_x)
            corrs.append(corr)
        multi_space_x = torch.mean(torch.stack(multi_space_x), dim=0)
        multi_space_x = self.out_proj(multi_space_x)
        # todo evolution Casuality graph，采样
        corrs = torch.mean(torch.stack(corrs), dim=0)
        # the shape of corrs is [N, N]
        return multi_space_x, corrs


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, seq_len, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.x_proj = nn.Sequential(nn.Linear(seq_len, 1), nn.ReLU() if activation == "relu" else nn.GELU())
        self.enc_layer = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU() if activation == "relu" else nn.GELU())
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, target, pearson_sparse_matrix=None):
        new_x, attn = self.attention(x, target, pearson_sparse_matrix=pearson_sparse_matrix)
        # the shape of attn is [N, N]
        new_x = self.x_proj(x.mT).squeeze(-1) + self.dropout(new_x)
        y = self.dropout(self.enc_layer(new_x))
        return new_x + y, attn


class Encoder(nn.Module):
    def __init__(self, embed, attn_layers, enc_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.embed = embed
        # todo self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        self.attn_layers = nn.ModuleList(attn_layers)
        self.enc_layers = nn.ModuleList(enc_layers) if enc_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, x_mark, pearson_sparse_matrix=None):
        x = torch.flip(x, dims=[2])
        x_mark = torch.flip(x_mark, dims=[2])
        target = x.squeeze(-1)
        embed_x = torch.concat(
            [x, repeat(self.embed(x_mark), "b n l h -> b (repeat n) l h", repeat=x.shape[1])], dim=-1
        )
        attns = []
        if self.enc_layers is not None:
            for attn_layer, enc_layer in zip(self.attn_layers, self.enc_layers):
                enc_x, attn = attn_layer(embed_x, target, pearson_sparse_matrix=pearson_sparse_matrix)
                enc_x = enc_layer(enc_x)
                attns.append(attn)
            enc_x, attn = self.attn_layers[-1](enc_x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                enc_x, attn = attn_layer(embed_x, target, pearson_sparse_matrix=pearson_sparse_matrix)
                # the shape of attn is [N, N]
                attns.append(attn)
        attns = torch.mean(torch.stack(attns), dim=0)
        # the shape of attns is [N, N]
        if self.norm is not None:
            enc_x = self.norm(enc_x)
        return enc_x, attns


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len

        self.decomp = SeriesDec(args.moving_avg)
        # the number of service number depends on the specific dataset
        # ALI         26118     number of service
        # APP_RPC     113       number of service
        # AZURE       17577     number of service
        if args.data == "ALI":
            N = 1000        # 1000
        elif args.data == "APP_RPC":
            N = 113           # 113
        elif args.data == "AZURE":
            N = 1000         # 1000
        else:
            raise ValueError("Unknown data type specified.")
        # Create an N x N matrix (tensor) initialized with zeros as a model parameter
        self.pearson_sparse_matrix = torch.zeros((N, N), dtype=torch.float32)

        # Encoder
        self.trend_encoder = Encoder(
            embed=TemporalEmbedding(args.c_in),
            attn_layers=[
                EncoderLayer(
                    MultiManifoldEmbedding(args.c_in + 1, args.d_model, args, taus=np.array([1, 2, 3, 4, 5])),
                    args.c_in + 1,
                    args.seq_len,
                    args.d_ff,
                    dropout=args.dropout,
                    activation=args.activation,
                )
                for _ in range(args.e_layers)
            ],
            norm_layer=SeasonalLayerNorm(args.c_in + 1),
        )
        self.deep_encoder = TransformerEncoder(args)
        self.seasonal_encoder = copy.deepcopy(self.trend_encoder)

        from prediction.timesnet import Model as TimesNet

        self.combine_type = "concat" # you can change here to switch concat([ccm, timesnet]) and seq([ccm, timesnet])
        config = copy.deepcopy(args)
        if self.combine_type == "seq":
            config.enc_in = 49
            config.seq_len = N
            self.timesnet = TimesNet(config)
        elif self.combine_type == "concat":
            self.timesnet = TimesNet(config)

            self.projection = nn.Sequential(
                nn.Linear(args.c_in + 1 + args.d_model, args.pred_len, bias=True), nn.ReLU()
            )


            self.W_prediction = nn.Parameter(
                nn.init.orthogonal_(torch.Tensor(args.c_in + 1 + args.d_model, args.low_rank_prediction_d)))
            self.W_prediction_back = nn.Parameter(
                nn.init.orthogonal_(torch.Tensor(args.low_rank_prediction_d, args.c_in + 1 + args.d_model)) )

            # self.lr_projection = nn.Sequential(nn.Linear(args.low_rank_prediction_d, args.pred_len, bias=True),
            #                                    nn.ReLU())

        else:
            self.projection = nn.Sequential(
                nn.Linear(args.c_in + 1 + args.d_model, args.pred_len, bias=True), nn.ReLU()
            )

    def forecast(self, x_enc, x_mark_enc, pearson_sparse_matrix=None):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(2, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=2, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev

        # in this project, H is 1
        # the shape of x_enc is [B, N, Lx, H], the shape of x_mark_enc is [B, 1, Lx, 4]
        # args.seq_len = Lx
        # the shape of trend_part is [B, N, args.c_in+1]
        trend_part, t_attns = self.trend_encoder(x_enc, x_mark_enc, pearson_sparse_matrix=pearson_sparse_matrix)
        # the shape of t_attns is [N, N]
        # the shape of x_enc is [B, N, Lx, H], the shape of x_mark_enc is [B, 1, Lx, 4]
        # the shape of deep_enc_out is [B, N, d_model]
        # print(x_enc.shape, x_mark_enc.shape)
        # deep_enc_out = self.deep_encoder(x_enc, x_mark_enc)
        # enc_out = torch.cat([trend_part, deep_enc_out], dim=-1)

        enc_out = trend_part

        # print(trend_part.shape, deep_enc_out.shape, enc_out.shape, x_mark_enc.shape)
        if self.combine_type == "seq":
            dec_out = self.timesnet(enc_out, None, None, None)
        elif self.combine_type == "concat":
            B, N, L, H = x_enc.shape
            timesnet_out = self.timesnet(x_enc.flatten(0, 1), None, None, None).reshape(B, N, -1)
            enc_out = torch.cat([enc_out, timesnet_out], dim=-1)
            # the shape of enc_out is [B, N, args.c_in + 1 + args.d_model]

        if self.args.low_rank_prediction:
            #########
            # low rank prediction
            enc_out = torch.matmul(enc_out, self.W_prediction)
            # the shape of enc_out is [B, N, low_rank_prediction_d]
            enc_out = torch.matmul(enc_out, self.W_prediction_back)
            # the shape oof enc_out is [B, N, args.c_in + 1 + args.d_model]
            dec_out = self.projection(enc_out)
            # the shape of dec_out is [B, N, pred_len]

        else:
            dec_out = self.projection(enc_out)
            # the shape of dec_out is [B, N, pred_len]

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * stdev.squeeze(2).repeat(1, 1, self.pred_len)
        dec_out = dec_out + means.squeeze(2).repeat(1, 1, self.pred_len)
        return dec_out, t_attns

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, pearson_sparse_matrix=None):
        # print("inp ", x_enc.shape, x_mark_enc.shape)
        dec_out, t_attns = self.forecast(
            x_enc,
            x_mark_enc,
            pearson_sparse_matrix=(
                pearson_sparse_matrix if pearson_sparse_matrix is not None else self.pearson_sparse_matrix
            ),
        )
        self.pearson_sparse_matrix = t_attns
        return dec_out
        # return self.forecast(x_enc, x_mark_enc, pearson_sparse_matrix=pearson_sparse_matrix)
        seasonal_init, trend_init = self.decomp(x_enc)
        trend_part, t_attns = self.trend_encoder(seasonal_init, x_mark_enc)
        seasonal_part, s_attns = self.seasonal_encoder(trend_init, x_mark_enc)
        dec_out = self.projection(trend_part + seasonal_part)
        return dec_out
