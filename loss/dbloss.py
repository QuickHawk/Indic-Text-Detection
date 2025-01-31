from torch.nn import Module, BCELoss, L1Loss

class DBLoss(Module):
    def __init__(self, alpha=1.0, beta=10.0):
        super(DBLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce_loss = BCELoss()
        self.l1_loss = L1Loss()

    def forward(self, pred_prob_map, pred_thresh_map, pred_approx_bin_map, gt_prob_map, gt_thresh_map):
        L_s = self.bce_loss(pred_prob_map, gt_prob_map)
        
        L_t = self.l1_loss(pred_thresh_map, gt_thresh_map)
        
        L_b = self.bce_loss(pred_approx_bin_map, gt_prob_map)
        
        L = L_s + self.alpha * L_t + self.beta * L_b
        
        return L
