<<<<<<< HEAD
import numpy as np
=======
>>>>>>> origin/MarieAngeBDev
import torch.nn as nn
import segmentation_models_pytorch as smp

class MultiHeadUnet(nn.Module):
<<<<<<< HEAD
    def __init__(self, base_model, num_heads, domain_ids_list,training=True, validation=True):
        super(MultiHeadUnet, self).__init__()
        self.base_model = base_model
        self.num_heads = num_heads
        self.domain_ids_list = domain_ids_list
=======
    def __init__(self, base_model, num_heads, training=True, validation=True):
        super(MultiHeadUnet, self).__init__()
        self.base_model = base_model
        self.num_heads = num_heads
        
>>>>>>> origin/MarieAngeBDev
        # Create multiple segmentation heads
        self.segmentation_heads = nn.ModuleList()
        for _ in range(num_heads):          
            self.segmentation_heads.append(self.base_model.segmentation_head)
            
        # Remove the segmentation_head layer
        del self.base_model.segmentation_head
        # Set the encoder attribute
        self.encoder = self.base_model.encoder
        self.decoder = self.base_model.decoder
        self.shared_layers = [self.encoder, self.decoder]
        self.training = training
<<<<<<< HEAD
        self.validation = validation         
        
    def forward(self, x, domain_ids_list):
        batch_size = x.size(0)
        features = self.encoder(x)
        decoder_output = self.base_model.decoder(*features)

        segmentation_outputs = []
        
        for head_idx in np.unique(domain_ids_list):
            indexes = [index for index, element in enumerate(domain_ids_list) if element == head_idx]
            image = x[indexes]

            # Forward pass through the selected segmentation heads based on the domain IDs
            segmentation_output = self.segmentation_heads[head_idx](decoder_output)
        
        return segmentation_output, decoder_output
=======
        self.validation = validation        
        
        
    def forward(self, x, domain_ids_list):
                
        features = self.encoder(x)
        decoder_output = self.base_model.decoder(*features)
        
        # Forward pass through the selected segmentation heads based on the domain IDs
        segmentation_outputs = []
        for i, domain_ids in enumerate(domain_ids_list):
            for domain_id in domain_ids:
                head_idx = domain_id
                segmentation_output = self.segmentation_heads[i](decoder_output)
                segmentation_outputs.append(segmentation_output)
               
        return segmentation_outputs, decoder_output
>>>>>>> origin/MarieAngeBDev
