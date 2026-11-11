import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskGenerator(nn.Module):
    """Base class for mask generators"""
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        
    def forward(self, mask_logits, batch_size):
        """Generate mask for training
        
        Args:
            mask_logits: External mask_logits parameter
            batch_size: Batch size
            
        Returns:
            torch.Tensor: Mask with shape [batch_size, num_features, num_features]
        """
        raise NotImplementedError
        
    def get_causal_mask(self, mask_logits):
        """Get final causal structure mask
        
        Args:
            mask_logits: External mask_logits parameter
            
        Returns:
            numpy.ndarray: Causal structure matrix with shape [num_features, num_features]
        """
        raise NotImplementedError
        
    def update_parameters(self):
        """Update generator parameters (e.g., temperature)"""
        raise NotImplementedError
        
    def update_parameters_for_early_stopping(self):
        """Update parameters (called in early stopping mechanism)"""
        raise NotImplementedError
        
    def get_parameters(self):
        """Get current parameter state
        
        Returns:
            dict: Dictionary containing current parameter state
        """
        raise NotImplementedError
        
    def initialize_mask_logits(self, device='cpu'):
        """Initialize mask_logits
        
        Args:
            device: Device
            
        Returns:
            torch.Tensor: Initialized mask_logits
        """
        raise NotImplementedError

class GumbelSoftmaxMaskGenerator(MaskGenerator):
    """Mask generator using Gumbel-Softmax"""
    def __init__(self, num_features, initial_temperature=0.5, final_temperature=2.0, temperature_multiplier=1.1):
        super().__init__(num_features)
        
        # Temperature parameter settings
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.temperature = initial_temperature
        self.temperature_multiplier = temperature_multiplier
        
        # Store current Gumbel noise
        self.current_gumbel_noise = None
        
    def _prepare_logits(self, mask_logits):
        """Prepare logits, handle diagonal"""
        diag_mask = torch.eye(self.num_features, device=mask_logits.device)
        mask_logits_no_diag = mask_logits.clone()
        mask_logits_no_diag = mask_logits_no_diag * (1 - diag_mask)
        mask_logits_no_diag = mask_logits_no_diag + diag_mask * (-1e9)
        return mask_logits_no_diag, diag_mask
    
    def forward(self, mask_logits, batch_size):
        """Generate mask for training"""
        mask_logits_no_diag, diag_mask = self._prepare_logits(mask_logits)
        
        if self.training:
            # Use Gumbel-Softmax during training
            if self.current_gumbel_noise is None:
                self.current_gumbel_noise = 0.5 * -torch.log(-torch.log(torch.rand_like(mask_logits_no_diag) + 1e-10) + 1e-10)
            noisy_logits = (mask_logits_no_diag + self.current_gumbel_noise) / self.temperature
        else:
            # Use logits directly during evaluation
            noisy_logits = mask_logits_no_diag / self.temperature
            
        # Apply softmax
        mask = F.softmax(noisy_logits, dim=-1)
        
        # Force diagonal to 0
        mask = mask * (1 - diag_mask)
        
        # Expand mask to batch dimension
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        return mask
    
    def update_temperature(self):
        """Update temperature parameter, starting from 0.5, multiply by a number greater than 1 each time until reaching maximum value"""
        self.temperature = min(self.temperature * self.temperature_multiplier, self.final_temperature)
        return self.temperature
        
    def get_causal_mask(self, mask_logits):
        """Generate mask for evaluation"""
        mask_logits_no_diag, diag_mask = self._prepare_logits(mask_logits)
        mask = F.softmax(mask_logits_no_diag / self.temperature, dim=-1)
        mask = mask * (1 - diag_mask)
        return mask
        
    def update_parameters(self):
        """Update parameters"""
        # Update temperature parameter
        self.temperature = min(
            self.temperature * self.temperature_multiplier,
            self.final_temperature
        )
        
        # Update Gumbel noise
        self.current_gumbel_noise = None
        
    def get_parameters(self):
        """Get current parameters"""
        return {
            'temperature': self.temperature,
            'initial_temperature': self.initial_temperature,
            'final_temperature': self.final_temperature,
            'temperature_multiplier': self.temperature_multiplier
        }
        
    def initialize_mask_logits(self, device='cpu'):
        """Initialize mask_logits
        
        Args:
            device: Device
            
        Returns:
            torch.Tensor: Initialized mask_logits
        """
        random_values = torch.rand(self.num_features, self.num_features, device=device) * 5 + 5
        random_values.fill_diagonal_(-10.0)
        return random_values

class SigmoidMaskGenerator(MaskGenerator):
    """Mask generator using Sigmoid function, controlling sparsity through threshold"""
    def __init__(self, num_features, initial_threshold=0.1, final_threshold=0.3, threshold_multiplier=1.1):
        super().__init__(num_features)
        
        # Threshold parameter settings
        self.initial_threshold = initial_threshold
        self.final_threshold = final_threshold
        self.threshold = initial_threshold
        self.threshold_multiplier = threshold_multiplier
        
    def _get_diag_mask(self, device):
        """Get mask with diagonal as 0"""
        return 1 - torch.eye(self.num_features, device=device)
    
    def forward(self, mask_logits, batch_size):
        """Generate mask for training
        
        Args:
            mask_logits: External mask_logits parameter
            batch_size: Batch size
            
        Returns:
            torch.Tensor: Mask with shape [batch_size, num_features, num_features]
        """
        # Apply sigmoid function
        mask = torch.sigmoid(mask_logits)
        # Force diagonal to 0
        mask = mask * self._get_diag_mask(mask_logits.device)
        
        # Expand mask to batch dimension
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        return mask
    
    def get_causal_mask(self, mask_logits):
        """Generate mask for evaluation
        
        Args:
            mask_logits: External mask_logits parameter
            
        Returns:
            torch.Tensor: Causal structure matrix with shape [num_features, num_features]
        """
        mask = torch.sigmoid(mask_logits)
        # During evaluation, use threshold, set values below threshold to 0, keep others as original
        mask = torch.where(mask > self.threshold, mask, torch.zeros_like(mask))
        # Force diagonal to 0
        mask = mask * self._get_diag_mask(mask_logits.device)
        return mask
    
    def update_parameters(self):
        """Update parameters (called each epoch)"""
        pass
    
    def update_parameters_for_early_stopping(self):
        """Update parameters (called in early stopping mechanism)"""
        # Update threshold parameter
        self.threshold = min(
            self.threshold * self.threshold_multiplier,
            self.final_threshold
        )
    
    def get_parameters(self):
        """Get current parameters"""
        return {
            'threshold': self.threshold,
            'initial_threshold': self.initial_threshold,
            'final_threshold': self.final_threshold,
            'threshold_multiplier': self.threshold_multiplier
        }
    
    def initialize_mask_logits(self, device='cpu'):
        """Initialize mask_logits
        
        Args:
            device: Device
            
        Returns:
            torch.Tensor: Initialized mask_logits
        """
        # Use small random values for initialization, range [-0.1, 0.1]
        random_values = torch.rand(self.num_features, self.num_features, device=device) * 0.2 - 0.1
        return random_values 