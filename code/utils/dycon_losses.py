
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def adaptive_beta(epoch, total_epochs, max_beta=1.0, min_beta=0.1):
    """
    Exponential annealing of beta from max_beta -> min_beta over total_epochs.
    Matches the paper's description when max_beta=1.0, min_beta=0.1.
    """
    ratio = min_beta / max_beta
    exponent = epoch / max_epochs if total_epochs > 0 else 1.0
    beta = max_beta * (ratio ** exponent)
    return beta


def gambling_softmax(probs, temperature=1.0):
    """
    Gambling softmax over the channel dimension, based on probabilities p_s.

    Args:
        probs: Tensor of shape (B, C, ...) with probabilities along dim=1.
        temperature: Softmax temperature on log-probs.

    Returns:
        Tensor of same shape as probs, reweighted as in a gambling-style softmax.
    """
    EPS = 1e-8
    # Max probability per voxel/pixel over classes
    C, _ = probs.max(dim=1, keepdim=True)          # (B, 1, ...)
    log_p = torch.log(probs + EPS)
    logits = log_p / temperature + (1.0 - C)       # (B, C, ...)
    exp_logits = torch.exp(logits)
    denom = exp_logits.sum(dim=1, keepdim=True)
    return exp_logits / (denom + EPS)


def sigmoid_rampup(current_epoch, total_rampup_epochs, min_threshold, max_threshold, steepness=5.0):
    """
    Compute a dynamic threshold using a sigmoid ramp-up schedule.
    """
    if total_rampup_epochs == 0:
        return max_threshold
    current_epoch = max(0.0, min(float(current_epoch), total_rampup_epochs))
    phase = 1.0 - (current_epoch / total_rampup_epochs)
    ramp = math.exp(-steepness * (phase ** 2))
    return min_threshold + (max_threshold - min_threshold) * ramp


class UnCLoss(nn.Module):
    """
    UnCLoss: Uncertainty-aware consistency loss (Equation 1 style).

    Takes student and teacher logits and a beta scalar, and computes:
      (p_s - p_t)^2 / (exp(beta H_s) + exp(beta H_t)) + beta (H_s + H_t),
    averaged over voxels, batch, and classes.
    """
    def __init__(self, *args, **kwargs):
        """
        *args, **kwargs are accepted for backward compatibility.
        This way UnCLoss() and UnCLoss(device='cuda') both work.
        """
        super(UnCLoss, self).__init__()

    def forward(self, s_logits, t_logits, beta):
        EPS = 1e-6

        # Student probabilities and entropy
        p_s = F.softmax(s_logits, dim=1)                    # (B, C, H, W, D)
        p_s_log = torch.log(p_s + EPS)
        H_s = -torch.sum(p_s * p_s_log, dim=1, keepdim=True)  # (B, 1, H, W, D)

        # Teacher probabilities and entropy
        p_t = F.softmax(t_logits, dim=1)
        p_t_log = torch.log(p_t + EPS)
        H_t = -torch.sum(p_t * p_t_log, dim=1, keepdim=True)

        # Entropy-scaled exponentials
        exp_H_s = torch.exp(beta * H_s)
        exp_H_t = torch.exp(beta * H_t)

        # Entropy-weighted squared difference
        loss_term = (p_s - p_t) ** 2 / (exp_H_s + exp_H_t)

        # Add entropy penalty
        loss = loss_term.sum(dim=1) + beta * (H_s + H_t)    # sum over classes
        loss = torch.mean(loss)                             # mean over all voxels + batch

        return loss


class FeCLoss(nn.Module):
    """
    FeCLoss implementing an Eq. 3–6 style formulation:

    - Student–student contrastive term with dual focal weights:
        F_k^+ = (1 - S_ik)^γ * exp(H_gs(i))
        F_q^- = (S_iq)^γ
    - Optional teacher top-k negatives in the denominator (Eq. 6).
    - Optional gambling_uncertainty to modulate positive weights (entropy from decoder).

    Args:
        device: computation device (string or torch.device). Used for building identity matrices.
        temperature: contrastive softmax temperature τ.
        gamma: focal exponent.
        use_focal: whether to enable focal weighting.
        rampup_epochs, lambda_cross, topk: kept for API consistency (topk used).
    """
    def __init__(self, device=None, temperature=0.6, gamma=2.0, use_focal=True,
                 rampup_epochs=2000, lambda_cross=0.0, topk=5):
        super(FeCLoss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.gamma = gamma
        self.use_focal = use_focal
        self.rampup_epochs = rampup_epochs
        self.lambda_cross = lambda_cross      # kept for compatibility; not used explicitly
        self.topk = topk

    def forward(self, feat, mask, teacher_feat=None, gambling_uncertainty=None, epoch=0):
        """
        feat:              (B, N, D) student embeddings (row = anchor i, col = patch)
        mask:              (B, 1, N) binary labels per patch (0/1)
        teacher_feat:      (B, N, D) teacher embeddings (optional)
        gambling_uncertainty: (B, N) entropy per patch (optional, from gambling_softmax)
        epoch:             current epoch (not used directly here, but kept for API)
        """
        EPS = 1e-8
        B, N, D = feat.shape

        dev = feat.device

        # Student–student similarity
        S = torch.matmul(feat, feat.transpose(1, 2)) / self.temperature  # (B, N, N)

        # label mask: positives if same label, else negatives
        mem_mask = torch.eq(mask, mask.transpose(1, 2)).float()          # (B, N, N)
        identity = torch.eye(N, device=dev).view(1, N, N).expand(B, -1, -1)
        pos_mask = mem_mask * (1.0 - identity)                           # no self-positives
        neg_mask = 1.0 - mem_mask                                        # negatives

        # remove self-sim from S
        S = S * (1.0 - identity)

        # exp(S) for pos/neg
        exp_S = torch.exp(S)
        Z_pos = exp_S * pos_mask                                         # (B, N, N)
        Z_neg = exp_S * neg_mask                                         # (B, N, N)

        # ---------- Teacher top-k negatives (Eq. 6) ----------
        if teacher_feat is not None:
            # student–teacher similarity
            S_cross = torch.matmul(feat, teacher_feat.transpose(1, 2)) / self.temperature  # (B, N, N)

            # negatives only (assume same label layout)
            mem_mask_cross = mem_mask
            cross_neg_mask = 1.0 - mem_mask_cross                                    # (B, N, N)

            # mask out non-negatives
            S_cross_masked = S_cross * cross_neg_mask + (-1e9) * (1.0 - cross_neg_mask)
            S_cross_flat = S_cross_masked.view(B * N, N)

            k = min(self.topk, N)
            topk_vals, _ = torch.topk(S_cross_flat, k=k, dim=-1)
            topk_vals = topk_vals.view(B, N, k)

            # mean exp(S_il) over top-k teacher negatives
            teacher_term = torch.exp(topk_vals).mean(dim=-1, keepdim=True)          # (B, N, 1)
        else:
            teacher_term = 0.0

        # ---------- Dual focal weights (Eq. 4) ----------
        if self.use_focal:
            # clamp similarities into [0,1] to match (1 - S) and S structure
            S_clamped = S.clamp(min=0.0, max=1.0)

            # negatives: F_q^- = (S_iq)^γ
            F_neg = (S_clamped ** self.gamma) * neg_mask

            # positives: F_k^+ = (1 - S_ik)^γ * exp(H_gs(i))
            one_minus_S = (1.0 - S_clamped).clamp(min=0.0)
            F_pos = (one_minus_S ** self.gamma) * pos_mask

            # entropy modulation of positives: exp(H_gs(i))
            if gambling_uncertainty is not None:
                # gambling_uncertainty: (B, N) entropy per anchor i
                H_exp = torch.exp(gambling_uncertainty).unsqueeze(-1)    # (B, N, 1)
                F_pos = F_pos * H_exp                                    # broadcast over j
        else:
            F_pos = pos_mask
            F_neg = neg_mask

        # ---------- Build Eq. 3-style denominator ----------
        # D(i) = sum_k Z_pos_ik + sum_q F_q^- [ Z_neg_iq + mean_teacher_exp ]
        # teacher_term is per-i, broadcast over j
        neg_contrib = (Z_neg + teacher_term) * F_neg                     # (B, N, N)
        sum_neg = neg_contrib.sum(dim=-1)                                # (B, N)
        sum_pos = Z_pos.sum(dim=-1)                                      # (B, N)

        D_i = sum_pos + sum_neg + EPS                                    # (B, N)
        D_i = D_i.unsqueeze(-1)                                          # (B, N, 1)

        # Probability assigned to each positive pair
        P_pos = Z_pos / (D_i + EPS)                                      # (B, N, N)

        # ---------- Final FeCL loss ----------
        # cross-entropy over positives with focal weights
        loss_matrix = -torch.log(P_pos + EPS) * F_pos                    # (B, N, N)

        # normalize per anchor by number of positives
        pos_count = pos_mask.sum(dim=-1) + EPS                           # (B, N)
        loss_per_anchor = loss_matrix.sum(dim=-1) / pos_count            # (B, N)

        loss_student = loss_per_anchor.mean()                            # scalar

        # Teacher already used in denominator; no separate lambda_cross term
        total_loss = loss_student
        return total_loss


if __name__ == "__main__":
    # Quick self-test

    # Test UnCLoss
    s_logits = torch.randn(2, 2, 8, 8, 8)
    t_logits = torch.randn(2, 2, 8, 8, 8)
    beta = 0.8
    uncl = UnCLoss()
    loss_u = uncl(s_logits, t_logits, beta)
    print("UnCLoss test:", loss_u.item())

    # Test FeCLoss
    feat = torch.randn(2, 64, 32).cuda()          # (B, N, D)
    mask = torch.randint(0, 2, (2, 1, 64)).cuda()
    teacher_feat = torch.randn(2, 64, 32).cuda()
    gambling_uncertainty = torch.rand(2, 64).cuda()

    fecl = FeCLoss(device="cuda:0", use_focal=True, topk=5)
    loss_f = fecl(feat=feat,
                  mask=mask,
                  teacher_feat=teacher_feat,
                  gambling_uncertainty=gambling_uncertainty,
                  epoch=10)
    print("FeCLoss test:", loss_f.item())





# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# def adaptive_beta(epoch, total_epochs, max_beta=5.0, min_beta=0.5):
#     ratio = min_beta / max_beta
#     exponent = epoch / total_epochs
#     beta = max_beta * (ratio ** exponent)
#     return beta
    
# def gambling_softmax(logits):
#     """
#     Compute gambling softmax probabilities over the channel dimension.
    
#     Args:
#         logits (Tensor): Input tensor of shape (B, C, ...).
    
#     Returns:
#         Tensor: Softmax probabilities of the same shape.
#     """
#     exp_logits = torch.exp(logits)
#     denom = torch.sum(exp_logits, dim=1, keepdim=True)
#     return exp_logits / (denom + 1e-18)

# def sigmoid_rampup(current_epoch, total_rampup_epochs, min_threshold, max_threshold, steepness=5.0):
#     """
#     Compute a dynamic threshold using a sigmoid ramp-up schedule.

#     Args:
#         current_epoch (int or float): The current training epoch.
#         total_rampup_epochs (int or float): The number of epochs over which to ramp up the threshold.
#         min_threshold (float): The initial threshold value, chosen based on the histogram's lower tail.
#         max_threshold (float): The target threshold value after ramp-up.
#         steepness (float, optional): Controls how quickly the threshold ramps up (default=5.0).

#     Returns:
#         float: The computed threshold for the current epoch.
#     """
#     if total_rampup_epochs == 0:
#         return max_threshold
#     current_epoch = max(0.0, min(float(current_epoch), total_rampup_epochs))
#     phase = 1.0 - (current_epoch / total_rampup_epochs)
#     ramp = math.exp(-steepness * (phase ** 2))
#     return min_threshold + (max_threshold - min_threshold) * ramp


# class UnCLoss(nn.Module):
#     """
#     UnCLoss implements an uncertainty-aware consistency loss that compares the prediction distributions 
#     from a student and a teacher network. It is designed for semi-supervised learning scenarios, where 
#     the teacher network provides guidance (e.g., via noise-added views) to the student network.
    
#     The loss is computed as follows:
    
#       1. Compute the softmax probability distributions for both student (p_s) and teacher (p_t) logits.
#          - s_logits: tensor of shape (B, C, H, W, D) from the student network.
#          - t_logits: tensor of shape (B, C, H, W, D) from the teacher network.
      
#       2. Compute the Shannon entropy for each distribution:
#          - H_s = -∑_c p_s * log(p_s + EPS) for the student (resulting in shape (B, 1, H, W, D)).
#          - H_t is computed similarly for the teacher.
#          These entropies represent the uncertainty in the respective predictions.
      
#       3. Scale and exponentiate the entropies using a parameter β:
#          - exp_H_s = exp(β * H_s)
#          - exp_H_t = exp(β * H_t)
#          Higher values of β increase the impact of the uncertainty in the subsequent weighting.
      
#       4. Compute an entropy-weighted squared difference between the student and teacher probability distributions:
#          - The basic difference (p_s - p_t)^2 is divided by (exp_H_s + exp_H_t), meaning that regions 
#            with lower uncertainty (i.e. lower entropy) receive a higher weight.
      
#       5. Add a regularization term proportional to the sum of the entropies, scaled by β:
#          - This term penalizes high uncertainty, encouraging the networks to make confident predictions.
      
#       6. The final loss is the mean over all elements:
#          - First summing the weighted differences over the class dimension, then averaging over spatial dimensions and batch.
    
#     Args:
#         s_logits (Tensor): Logits from the student network with shape (B, C, H, W, D).
#         t_logits (Tensor): Logits from the teacher network with shape (B, C, H, W, D).
#         beta (float): A scaling parameter that modulates the influence of the entropy terms. A higher beta
#                       increases the weighting effect of the entropy, emphasizing regions of high certainty.
    
#     Returns:
#         Tensor: A scalar tensor representing the mean uncertainty-aware consistency loss.
#     """
#     def __init__(self):
#         super(UnCLoss, self).__init__()

#     def forward(self, s_logits, t_logits, beta):
#         EPS = 1e-6

#         # Compute student softmax probabilities and their entropy.
#         p_s = F.softmax(s_logits, dim=1)  # (B, C, H, W, D)
#         p_s_log = torch.log(p_s + EPS)
#         H_s = -torch.sum(p_s * p_s_log, dim=1, keepdim=True)  # (B, 1, H, W, D)

#         # Compute teacher softmax probabilities and their entropy.
#         p_t = F.softmax(t_logits, dim=1)  # (B, C, H, W, D)
#         p_t_log = torch.log(p_t + EPS)
#         H_t = -torch.sum(p_t * p_t_log, dim=1, keepdim=True)  # (B, 1, H, W, D)

#         # Exponentiate the entropies scaled by beta.
#         exp_H_s = torch.exp(beta * H_s)
#         exp_H_t = torch.exp(beta * H_t)

#         # Compute the entropy-weighted squared difference between student and teacher distributions.
#         # The higher the certainty (lower entropy), the larger the weight on the difference.
#         loss = (p_s - p_t)**2 / (exp_H_s + exp_H_t)

#         # Sum the differences over the class dimension, add a penalty for high entropy, and average.
#         loss = torch.mean(loss.sum(dim=1) + beta * (H_s + H_t))

#         return loss.mean()

# # class FeCLoss(nn.Module):
#     """
#     FeCLoss with an auxiliary teacher-based hard negative branch and gambling softmax uncertainty mask for guiding positive samples.
    
#     The primary loss is an InfoNCE-style contrastive loss computed from student embeddings.
#     The focal weighting is applied to hard positives (same-class pairs with low similarity)
#     and hard negatives (different-class pairs with high similarity).
    
#     Additionally, an auxiliary loss is computed by comparing student embeddings with teacher embeddings.
    
#     A gambling softmax is computed to produce an uncertainty mask from decoded_logits. 
#     The entropy (uncertainty) is then used to modulate the loss contribution of positive pairs.
    
#     Args:
#         device: Computation device ('cpu' or 'cuda').
#         temperature: Scaling factor τ.
#         gamma: Exponent for focal weighting.
#         use_focal: Boolean flag to enable focal weighting on the primary loss.
#         rampup_epochs: Number of epochs over which the thresholds are ramped up.
#         lambda_cross: Weight for the auxiliary teacher-based negative loss.
#     """
#     def __init__(self, device, temperature=0.6, gamma=2.0, use_focal=False, rampup_epochs=2000, lambda_cross=1.0):
#         super(FeCLoss, self).__init__()
#         self.device = device
#         self.temperature = temperature
#         self.gamma = gamma
#         self.use_focal = use_focal
#         self.rampup_epochs = rampup_epochs
#         self.lambda_cross = lambda_cross

#     def forward(self, feat, mask, teacher_feat=None, gambling_uncertainty=None, epoch=0):
#         """
#         Compute the total loss as the sum of:
#          - The FeCLoss computed on student embeddings.
#          - An auxiliary cross-negative loss computed between student and teacher embeddings.
#          - Modulate the positive part of the student loss using an uncertainty mask
#            computed via gambling softmax from student decoder.
        
#         Args:
#             feat: Tensor of shape (B, N, D) - Student embeddings.
#             mask: Tensor of shape (B, 1, N) - Ground truth labels per patch.
#             teacher_feat: (Optional) Tensor of shape (B, N, D) - Teacher embeddings.
#             epoch: Current epoch for dynamic threshold computation.
#             gambling_uncertainty: (Optional) Tensor of shape (B, N) - entropy from student decoder.
        
#         Returns:
#             Total loss (scalar): student loss + lambda_cross * teacher auxiliary loss,
#             with positive samples optionally weighted by the uncertainty mask.
#         """
#         B, N, _ = feat.shape

#         # Primary FeCLoss (Student Only)
#         mem_mask = torch.eq(mask, mask.transpose(1, 2)).float()  # (B, N, N): 1 if same label.
#         mem_mask_neg = 1 - mem_mask  # (B, N, N): 1 if different labels.

#         feat_logits = torch.matmul(feat, feat.transpose(1, 2)) / self.temperature  # (B, N, N)
#         identity = torch.eye(N, device=self.device)
#         neg_identity = 1 - identity  # Zero out self-similarity.
#         feat_logits = feat_logits * neg_identity

#         feat_logits_max, _ = torch.max(feat_logits, dim=1, keepdim=True)
#         feat_logits = feat_logits - feat_logits_max.detach()

#         exp_logits = torch.exp(feat_logits)  # (B, N, N)
#         neg_sum = torch.sum(exp_logits * mem_mask_neg, dim=-1)  # (B, N)

#         denominator = exp_logits + neg_sum.unsqueeze(dim=-1)
#         division = exp_logits / (denominator + 1e-18)  # Softmax-like probability.

#         loss_matrix = -torch.log(division + 1e-18)
#         loss_matrix = loss_matrix * mem_mask * neg_identity

#         loss_student = torch.sum(loss_matrix, dim=-1) / (torch.sum(mem_mask, dim=-1) - 1 + 1e-18)
#         loss_student = loss_student.mean()

#         # Apply focal weighting to the student loss
#         if self.use_focal:
#             similarity = division  # Using normalized similarity as proxy.
#             focal_weights = torch.ones_like(similarity)
#             pos_thresh = sigmoid_rampup(epoch, self.rampup_epochs, min_threshold=1.3, max_threshold=1.5)
#             neg_thresh = sigmoid_rampup(epoch, self.rampup_epochs, min_threshold=0.3, max_threshold=0.5)
#             hard_pos_mask = mem_mask.bool() & (similarity < pos_thresh)
#             focal_weights[hard_pos_mask] = (1 - similarity[hard_pos_mask]).pow(self.gamma)
#             hard_neg_mask = mem_mask_neg.bool() & (similarity > neg_thresh)
#             focal_weights[hard_neg_mask] = similarity[hard_neg_mask].pow(self.gamma)
#             loss_student = torch.sum(loss_matrix * focal_weights, dim=-1) / (torch.sum(mem_mask, dim=-1) - 1 + 1e-18)
#             loss_student = loss_student.mean()

#         # Incorporate Gambling Softmax Uncertainty Mask for Positives
#         if gambling_uncertainty is not None:
#             loss_student_per_patch = torch.sum(loss_matrix, dim=-1) / (torch.sum(mem_mask, dim=-1) - 1 + 1e-18) 
#             loss_student = (loss_student_per_patch * gambling_uncertainty).mean()

#         # Auxiliary Cross-Negative Loss (Teacher-Student)
#         loss_cross = 0.0
#         if teacher_feat is not None:
#             # Compute cross-similarity between student and teacher embeddings.
#             cross_sim = torch.matmul(feat, teacher_feat.transpose(1, 2)) 
#             mem_mask_cross = torch.eq(mask, mask.transpose(1, 2)).float()
#             mem_mask_cross_neg = 1 - mem_mask_cross  # Different classes.
            
#             # Use a dynamic threshold for teacher negatives instead of selecting top-k negatives.
#             cross_neg_thresh = sigmoid_rampup(epoch, self.rampup_epochs, min_threshold=0.3, max_threshold=0.5)
#             cross_hard_neg_mask = mem_mask_cross_neg.bool() & (cross_sim > cross_neg_thresh)
            
#             # Compute auxiliary loss for these hard negatives: penalty increases as similarity increases.
#             if cross_hard_neg_mask.sum() > 0:
#                 loss_cross_term = -torch.log(1 - cross_sim + 1e-18)
#                 loss_cross_term = loss_cross_term * cross_hard_neg_mask.float()
#                 loss_cross = torch.sum(loss_cross_term) / (torch.sum(cross_hard_neg_mask.float()) + 1e-18)
#             else:
#                 loss_cross = 0.0

#         # Total Loss
#         total_loss = loss_student + self.lambda_cross * loss_cross
#         return total_loss

# class FeCLoss(nn.Module):
#     def __init__(self, device, temperature=0.6, gamma=2.0, use_focal=True, rampup_epochs=2000, lambda_cross=0.0, topk=5):
#         """
#         FeCLoss implementing an Eq. 3–6 style loss:
#          - Student–student contrastive with dual focal weights.
#          - Optional teacher top-k hard negatives in the denominator.
#          - Optional gambling_uncertainty for positive weighting.
#         lambda_cross is kept for API compatibility but unused when we
#         already inject teacher info via the denominator.
#         """
#         super(FeCLoss, self).__init__()
#         self.device = device
#         self.temperature = temperature
#         self.gamma = gamma
#         self.use_focal = use_focal
#         self.rampup_epochs = rampup_epochs
#         self.lambda_cross = lambda_cross
#         self.topk = topk

#     def forward(self, feat, mask, teacher_feat=None, gambling_uncertainty=None, epoch=0):
#         """
#         feat: (B, N, D) student embeddings
#         mask: (B, 1, N) labels per patch
#         teacher_feat: (B, N, D) teacher embeddings (optional)
#         gambling_uncertainty: (B, N) entropy per patch (optional)
#         """
#         EPS = 1e-8
#         B, N, D = feat.shape

#         # similarity between student patches
#         S = torch.matmul(feat, feat.transpose(1, 2)) / self.temperature  # (B, N, N)

#         # masks
#         mem_mask = torch.eq(mask, mask.transpose(1, 2)).float()  # positives
#         identity = torch.eye(N, device=self.device).view(1, N, N)
#         identity = identity.expand(B, -1, -1)
#         pos_mask = mem_mask * (1.0 - identity)                    # remove self
#         neg_mask = 1.0 - mem_mask                                 # negatives

#         # zero out self-sim so it doesn't interfere
#         S = S * (1.0 - identity)

#         # exp(S) for pos/neg
#         exp_S = torch.exp(S)
#         Z_pos = exp_S * pos_mask                                  # (B, N, N)
#         Z_neg = exp_S * neg_mask                                  # (B, N, N)

#         # ---------- Teacher top-k negatives (Eq. 6) ----------
#         if teacher_feat is not None:
#             S_cross = torch.matmul(feat, teacher_feat.transpose(1, 2)) / self.temperature  # (B, N, N)
#             # only consider teacher patches from different classes
#             mem_mask_cross = mem_mask  # same mask shape as S (we assume same label layout)
#             cross_neg_mask = 1.0 - mem_mask_cross            # (B, N, N) negatives only

#             # mask out non-negatives with big negative number
#             S_cross_masked = S_cross * cross_neg_mask + (-1e9) * (1.0 - cross_neg_mask)
#             # flatten teacher dim for top-k selection
#             S_cross_flat = S_cross_masked.view(B * N, N)
#             k = min(self.topk, N)
#             topk_vals, _ = torch.topk(S_cross_flat, k=k, dim=-1)
#             topk_vals = topk_vals.view(B, N, k)
#             # mean exp(S_il) over top-k
#             teacher_term = torch.exp(topk_vals).mean(dim=-1, keepdim=True)  # (B, N, 1)
#         else:
#             teacher_term = 0.0

#         # ---------- Dual focal weights (Eq. 4) ----------

#         # base focal weights using similarity S_ij
#         if self.use_focal:
#             # clamp to avoid negative weirdness
#             S_clamped = S.clamp(min=0.0, max=1.0)

#             # negatives: F_q^- = (S_iq)^gamma
#             F_neg = (S_clamped ** self.gamma) * neg_mask          # (B, N, N)

#             # positives: F_k^+ = (1 - S_ik)^gamma * exp(H_gs(i))
#             one_minus_S = (1.0 - S_clamped).clamp(min=0.0)
#             F_pos = (one_minus_S ** self.gamma) * pos_mask        # (B, N, N)

#             if gambling_uncertainty is not None:
#                 # gambling_uncertainty: (B, N) -> exp(H_gs) per anchor i
#                 H_exp = torch.exp(gambling_uncertainty).unsqueeze(-1)  # (B, N, 1)
#                 F_pos = F_pos * H_exp                                  # broadcast over j
#         else:
#             F_pos = pos_mask
#             F_neg = neg_mask

#         # ---------- Build Eq. 3-style denominator ----------
#         # sum over negatives with focal weights and teacher term
#         # D(i) = sum_k Z_pos_ik + sum_q F_q^- [ Z_neg_iq + mean_teacher_exp ]
#         # Note: teacher_term is per-i; broadcast over j
#         neg_contrib = (Z_neg + teacher_term) * F_neg       # (B, N, N)
#         sum_neg = neg_contrib.sum(dim=-1)                  # (B, N)
#         sum_pos = Z_pos.sum(dim=-1)                        # (B, N)

#         D_i = sum_pos + sum_neg + EPS                      # (B, N)
#         D_i = D_i.unsqueeze(-1)                            # (B, N, 1)

#         # probability for each positive pair
#         P_pos = Z_pos / (D_i + EPS)                        # (B, N, N)

#         # ---------- Final FeCL loss ----------
#         loss_matrix = -torch.log(P_pos + EPS) * F_pos      # (B, N, N)

#         # normalize by number of positives per anchor
#         pos_count = pos_mask.sum(dim=-1) + EPS             # (B, N)
#         loss_per_anchor = loss_matrix.sum(dim=-1) / pos_count  # (B, N)

#         loss_student = loss_per_anchor.mean()              # scalar

#         # no separate lambda_cross teacher loss: already integrated in D_i
#         total_loss = loss_student
#         return total_loss


# if __name__ == "__main__":
#     # Test the UnCLoss
#     s_logits = torch.randn(8, 2, 16, 16, 16)
#     t_logits = torch.randn(8, 2, 16, 16, 16)
#     beta = 0.8
#     uncl = UnCLoss()
#     loss = uncl(s_logits, t_logits, beta)
#     print(f"uncl_loss: {loss}")
    
#     # Test the FeCLoss
#     feat = torch.randn(8, 128, 128).cuda()
#     mask = torch.randint(0, 2, (8, 1, 128)).cuda()
#     decoded_logits = torch.randn(8, 128).cuda()
    
#     fecl = FeCLoss(device='cuda:0', use_focal=True)
#     loss = fecl(feat=feat, mask=mask, teacher_feat=None, gambling_uncertainty=decoded_logits)
#     print(f"fecl_loss: {loss}")
