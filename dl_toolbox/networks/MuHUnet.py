import torch.nn as nn
import segmentation_models_pytorch as smp

class MultiHeadUnet(nn.Module):
    def __init__(self, base_model, num_heads, training=True, validation=True):
        super(MultiHeadUnet, self).__init__()
        self.base_model = base_model
        self.num_heads = num_heads
        
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
